"""Motion server: holds one Kortex session and runs ONE human-approved bounded routine.

A person starts it in a terminal, reads the envelope summary and types `yes` (or passes --yes
after approving). Clients then send JSON lines over TCP; each motion is checked against the
envelope. Clients cannot approve, widen or restart a routine: when it ends for any reason, motion
commands are refused until a person starts the server again.

    python3 -m kinova.server --envelope envelopes/etch_draw.json          # inside the container
    python3 -m kinova.server --envelope envelopes/etch_draw.json --fake   # no arm, for client development

Protocol: one JSON object per line, e.g. {"cmd": "rotate", "deg": 90}; reply {"ok": true, ...}.
Commands: status, move_pose (x y z theta_x theta_y theta_z [lin_speed] [ang_speed]),
rotate (deg [speed]), gripper (position), stop. There is no authentication: bind to localhost only.
"""
import argparse
import json
import socketserver
import sys
import threading
import time

from . import safety
from .arms import FakeArm, RealArm
from .config import load_config
from .envelope import Routine, RoutineStopped, load_envelope

KEEPALIVE_S = 10.0            # the Kortex session expires after 60 s without traffic


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        for raw in self.rfile:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                reply = srv.dispatch(json.loads(line))
            except RoutineStopped as e:
                reply = {"ok": False, "error": str(e), "routine_stopped": True}
            except safety.SafetyError as e:
                reply = {"ok": False, "error": "SAFETY: %s" % e}
            except (ValueError, KeyError, TypeError) as e:
                reply = {"ok": False, "error": "bad request: %s" % e}
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


class MotionServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, arm, routine):
        super().__init__(addr, _Handler)
        self.arm, self.routine = arm, routine
        self.last_use = time.time()
        self._one_at_a_time = threading.Lock()

    def dispatch(self, req: dict) -> dict:
        cmd = req["cmd"]
        r = self.routine
        if cmd == "stop":                             # never waits behind a running motion
            r.stop("stop requested by client")
            return {"ok": True, "stopped": r.stopped}
        with self._one_at_a_time:
            return self._dispatch_locked(cmd, req, r)

    def _dispatch_locked(self, cmd: str, req: dict, r) -> dict:
        self.last_use = time.time()
        if cmd == "status":
            return {"ok": True, "state": self.arm.snapshot(), "routine": r.env.name, "stopped": r.stopped,
                    "commands": r.n_commands, "elapsed_s": round(r.clock() - r.t0, 1)}
        if cmd == "move_pose":
            res = r.move_to_pose(*(float(req[k]) for k in ("x", "y", "z", "theta_x", "theta_y", "theta_z")),
                                 lin_speed=req.get("lin_speed"), ang_speed=req.get("ang_speed"))
        elif cmd == "rotate":
            res = r.rotate(float(req["deg"]), req.get("speed"))
        elif cmd == "gripper":
            res = r.gripper(float(req["position"]))
        else:
            raise ValueError("unknown cmd %r" % cmd)
        return dict(res, routine_stopped=r.stopped is not None, tool_pose=self.arm.tool_pose())


def _keepalive(server: MotionServer):
    while True:
        time.sleep(KEEPALIVE_S)
        if time.time() - server.last_use >= KEEPALIVE_S and server.routine.stopped is None:
            try:
                server.arm.arm_state()
            except Exception as e:                    # link or session lost: end the routine, do not reconnect
                server.routine.stop("keep-alive failed: %s" % e)


def serve(arm, env, cap_lin, cap_ang, host, port, assume_yes):
    routine = Routine(arm, env, cap_lin, cap_ang, assume_yes=assume_yes)
    server = MotionServer((host, port), arm, routine)
    threading.Thread(target=_keepalive, args=(server,), daemon=True).start()
    print("Routine %r approved. Serving on %s:%d. Ctrl-C stops the arm and ends the routine." % (env.name, host, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        routine.stop("server shut down")
        server.server_close()
        print("Routine ended: %s (%d commands)" % (routine.stopped, routine.n_commands), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envelope", required=True, help="JSON file describing the bounded routine")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside the container (the port is published to the Mac's localhost only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fake", action="store_true", help="no arm: instant, perfect FakeArm")
    ap.add_argument("--yes", action="store_true", help="only after a human has approved this envelope")
    args = ap.parse_args(argv)
    env = load_envelope(args.envelope)
    cfg = load_config(username="fake", password="fake") if args.fake else load_config()
    try:
        if args.fake:
            serve(FakeArm(stall_at=0.62), env, cfg.max_lin_speed, cfg.max_ang_speed, args.host, args.port, args.yes)
        else:
            from .connection import KortexConnection
            with KortexConnection(cfg) as k:
                serve(RealArm(k), env, cfg.max_lin_speed, cfg.max_ang_speed, args.host, args.port, args.yes)
    except safety.SafetyError as e:
        print("SAFETY: %s" % e, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
