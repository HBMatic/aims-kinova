"""Offline tests for the bounded routine, chunked rotation and motion server (no robot, no kortex_api)."""
import io
import json
import os
import socket
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kinova import motion, safety  # noqa: E402
from kinova.arms import FakeArm  # noqa: E402
from kinova.envelope import Envelope, Routine, RoutineStopped, load_envelope, tilt_from_down_deg  # noqa: E402
from kinova.server import MotionServer  # noqa: E402

ENV = load_envelope(os.path.join(os.path.dirname(__file__), "..", "..", "envelopes", "etch_draw.json"))
DOWN = (180.0, 0.0, 90.0)


def routine(arm=None, env=ENV):
    return Routine(arm or FakeArm(), env, cap_lin=0.10, cap_ang=30.0, assume_yes=True)


class TestEnvelope(unittest.TestCase):
    def test_tilt(self):
        self.assertAlmostEqual(tilt_from_down_deg(180, 0), 0.0, places=5)
        self.assertAlmostEqual(tilt_from_down_deg(170, 0), 10.0, places=5)
        self.assertAlmostEqual(tilt_from_down_deg(90, 0), 90.0, places=5)

    def test_envelope_cannot_loosen_caps(self):
        fast = Envelope(**dict(ENV.__dict__, max_lin_speed=0.5))
        with self.assertRaises(safety.SafetyError):
            routine(env=fast)

    def test_must_start_inside_and_pointing_down(self):
        home = {"x": 0.574, "y": 0.0, "z": 0.426, "theta_x": 90.0, "theta_y": 0.0, "theta_z": 90.0}
        with self.assertRaises(safety.SafetyError):
            routine(FakeArm(pose=home))

    def test_needs_human_when_not_a_tty(self):
        real, sys.stdin = sys.stdin, io.StringIO("")           # a script or agent has no terminal to type "yes" into
        try:
            with self.assertRaises(safety.SafetyError):
                Routine(FakeArm(), ENV, 0.10, 30.0, assume_yes=False)
        finally:
            sys.stdin = real


class TestRoutine(unittest.TestCase):
    def test_normal_sequence(self):
        arm = FakeArm(stall_at=0.62)
        r = routine(arm)
        self.assertTrue(r.move_to_pose(0.38, 0.09, 0.10, *DOWN)["ok"])
        self.assertIn("STALLED", r.gripper(1.0)["event"])
        self.assertTrue(r.rotate(470.0)["ok"])
        self.assertEqual(arm.total_rotation, 470.0)
        self.assertIsNone(r.stopped)

    def test_out_of_box_stops_the_routine_for_good(self):
        arm = FakeArm()
        r = routine(arm)
        with self.assertRaises(RoutineStopped):
            r.move_to_pose(0.38, 0.09, 0.40, *DOWN)            # above the box
        self.assertEqual(arm.stops, 1)
        with self.assertRaises(RoutineStopped):
            r.gripper(0.0)                                     # nothing runs after a stop
        self.assertEqual(arm.pose["z"], 0.20)

    def test_tilted_target_and_long_rotation_rejected(self):
        with self.assertRaises(RoutineStopped):
            routine().move_to_pose(0.38, 0.0, 0.15, 150.0, 0.0, 90.0)
        with self.assertRaises(RoutineStopped):
            routine().rotate(5000.0)

    def test_failed_motion_stops_without_retry(self):
        arm = FakeArm()
        r = routine(arm)
        arm.fail_next = "ACTION_ABORT"
        self.assertFalse(r.rotate(90.0)["ok"])
        self.assertIn("ACTION_ABORT", r.stopped)
        self.assertEqual(arm.stops, 1)
        with self.assertRaises(RoutineStopped):
            r.rotate(90.0)

    def test_expiry(self):
        now = [0.0]
        r = Routine(FakeArm(), ENV, 0.10, 30.0, assume_yes=True, clock=lambda: now[0])
        now[0] = ENV.max_duration_s + 1
        with self.assertRaises(RoutineStopped):
            r.gripper(0.0)


class TestRotateChunks(unittest.TestCase):
    def setUp(self):
        self.q, self.t, self.sent = 170.0, 0.0, []

    def send(self, speed=30.0, ok=True):
        def f(target, timeout):
            self.sent.append(target)
            self.t += abs(safety.wrap_deg(target - self.q)) / speed
            self.q = target
            return {"ok": ok, "event": "ACTION_END" if ok else "ACTION_ABORT"}
        return f

    def run_chunks(self, delta, **kw):
        return motion.rotate_chunks(delta, 30.0, lambda: self.q, self.send(**kw), clock=lambda: self.t)

    def test_multi_turn_across_the_wrap(self):
        res = self.run_chunks(470.0)
        self.assertTrue(res["ok"])
        self.assertEqual(res["chunks"], 4)
        self.assertAlmostEqual(res["rotated_deg"], 470.0)
        self.assertTrue(all(-180 < s <= 180 for s in self.sent))

    def test_negative(self):
        self.assertAlmostEqual(self.run_chunks(-250.0)["rotated_deg"], -250.0)

    def test_detects_ignored_speed_limit(self):
        res = self.run_chunks(240.0, speed=200.0)              # arm moved far faster than asked
        self.assertFalse(res["ok"])
        self.assertIn("SPEED_LIMIT_NOT_HONOURED", res["event"])
        self.assertEqual(res["chunks"], 1)                     # stopped after the first chunk

    def test_abort_reports_partial_rotation(self):
        res = self.run_chunks(240.0, ok=False)
        self.assertFalse(res["ok"])
        self.assertEqual(res["chunks"], 1)


class TestServer(unittest.TestCase):
    def test_json_round_trip_and_refusal_after_stop(self):
        arm = FakeArm(stall_at=0.62)
        srv = MotionServer(("127.0.0.1", 0), arm, routine(arm))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        sock = socket.create_connection(srv.server_address)
        f = sock.makefile("rw")

        def call(**req):
            f.write(json.dumps(req) + "\n")
            f.flush()
            return json.loads(f.readline())

        try:
            self.assertEqual(call(cmd="status")["routine"], "etch_draw")
            self.assertTrue(call(cmd="rotate", deg=360)["ok"])
            bad = call(cmd="move_pose", x=0.38, y=0.5, z=0.1, theta_x=180, theta_y=0, theta_z=90)
            self.assertFalse(bad["ok"])
            self.assertTrue(bad["routine_stopped"])
            self.assertFalse(call(cmd="rotate", deg=10)["ok"])
            self.assertFalse(call(cmd="approve")["ok"])       # clients cannot approve anything
        finally:
            f.close()
            sock.close()
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
