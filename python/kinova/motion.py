"""High-level (single-level servoing) motion commands. Every call blocks until the action
ENDs/ABORTs, and a Ctrl-C during motion sends Base.Stop()."""
import threading
import time
from contextlib import contextmanager

from . import safety, state

ACTION_TIMEOUT_S = 30.0


def ensure_high_level(k):
    from kortex_api.autogen.messages import Base_pb2
    mode = Base_pb2.ServoingModeInformation()
    mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    k.base.SetServoingMode(mode)


def stop(k):
    """Stop the current motion (soft stop, the arm stays powered)."""
    k.base.Stop()


def clear_faults(k):
    k.base.ClearFaults()
    time.sleep(1.0)


@contextmanager
def _action_watch(k):
    """Subscribe to ActionTopic; yield (event, result). ExecuteAction only *accepts* a command,
    so completion must be observed through notifications."""
    from kortex_api.autogen.messages import Base_pb2
    done, result = threading.Event(), {}

    def cb(n):
        if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            result["event"] = Base_pb2.ActionEvent.Name(n.action_event)
            if n.action_event == Base_pb2.ACTION_ABORT:
                result["abort_details"] = str(n.abort_details)
            done.set()

    handle = k.base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
    try:
        yield done, result
    finally:
        k.base.Unsubscribe(handle)


def _run(k, send, timeout=ACTION_TIMEOUT_S) -> dict:
    ensure_high_level(k)
    with _action_watch(k) as (done, result):
        send()
        try:
            finished = done.wait(timeout)
        except KeyboardInterrupt:
            stop(k)
            return {"ok": False, "event": "STOPPED_BY_USER"}
    if not finished:
        stop(k)
        return {"ok": False, "event": "TIMEOUT"}
    return {"ok": result.get("event") == "ACTION_END", **result}


def run_stored_action(k, name: str, action_type="REACH_JOINT_ANGLES") -> dict:
    """e.g. 'Home', 'Retract', 'Zero', 'Packaging' or a pose you saved in the web app."""
    from kortex_api.autogen.messages import Base_pb2
    req = Base_pb2.RequestedActionType()
    req.action_type = getattr(Base_pb2, action_type)
    handle = next((a.handle for a in k.base.ReadAllActions(req).action_list if a.name == name), None)
    if handle is None:
        raise ValueError(f"No stored {action_type} action named {name!r}: {state.stored_actions(k, action_type)}")
    return _run(k, lambda: k.base.ExecuteActionFromReference(handle))


def move_to_pose(k, x, y, z, theta_x, theta_y, theta_z, lin_speed=None, ang_speed=None) -> dict:
    """Absolute Cartesian goal in the base frame (m, deg) with speed caps."""
    from kortex_api.autogen.messages import Base_pb2
    cur = state.tool_pose(k)
    safety.check_cartesian(x, y, z, current=(cur["x"], cur["y"], cur["z"]))

    action = Base_pb2.Action()
    action.name = "kinova_cli_reach_pose"
    action.application_data = ""
    tp = action.reach_pose.target_pose
    tp.x, tp.y, tp.z = x, y, z
    tp.theta_x, tp.theta_y, tp.theta_z = theta_x, theta_y, theta_z
    sp = action.reach_pose.constraint.speed
    sp.translation = safety.clamp_speed(lin_speed or k.cfg.max_lin_speed, k.cfg.max_lin_speed)
    sp.orientation = safety.clamp_speed(ang_speed or k.cfg.max_ang_speed, k.cfg.max_ang_speed)
    return _run(k, lambda: k.base.ExecuteAction(action))


def move_relative(k, dx=0.0, dy=0.0, dz=0.0, dthx=0.0, dthy=0.0, dthz=0.0, **kw) -> dict:
    """Translate/rotate the tool relative to its current pose (base-frame deltas)."""
    p = state.tool_pose(k)
    return move_to_pose(k, p["x"] + dx, p["y"] + dy, p["z"] + dz,
                        p["theta_x"] + dthx, p["theta_y"] + dthy, p["theta_z"] + dthz, **kw)


def move_joints(k, angles_deg) -> dict:
    from kortex_api.autogen.messages import Base_pb2
    safety.check_joints(angles_deg)
    action = Base_pb2.Action()
    action.name = "kinova_cli_reach_joints"
    action.application_data = ""
    for i, a in enumerate(angles_deg):
        ja = action.reach_joint_angles.joint_angles.joint_angles.add()
        ja.joint_identifier = i
        ja.value = a
    return _run(k, lambda: k.base.ExecuteAction(action))


def gripper(k, position: float, timeout=5.0, tol=0.01) -> dict:
    """Position mode, 0 = open, 1 = closed. Returns when the target is reached or the fingers stall
    (stall = object grasped: position stops changing before reaching the target)."""
    from kortex_api.autogen.messages import Base_pb2
    position = min(max(float(position), 0.0), 1.0)
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    f = cmd.gripper.finger.add()
    f.finger_identifier = 1
    f.value = position
    k.base.SendGripperCommand(cmd)

    t0, last, still_since = time.time(), None, None
    while time.time() - t0 < timeout:
        cur = state.gripper_position(k)
        if abs(cur - position) <= tol:
            return {"ok": True, "event": "REACHED", "position": cur}
        if last is not None and abs(cur - last) < 1e-3:
            still_since = still_since or time.time()
            if time.time() - still_since > 0.5 and time.time() - t0 > 0.5:
                return {"ok": True, "event": "STALLED (object?)", "position": cur}
        else:
            still_since = None
        last = cur
        time.sleep(0.05)
    return {"ok": False, "event": "TIMEOUT", "position": state.gripper_position(k)}


# ---- multi-turn rotation of a continuous joint ---------------------------------------------------
ROTATE_CHUNK_DEG = 120.0      # well under 180 so each chunk's direction and read-back are unambiguous
ROTATE_TOL_DEG = 1.5
ROTATE_MIN_TIME_FRAC = 0.6    # a chunk that finishes much faster than commanded means the speed limit was ignored


def rotate_chunks(delta_deg, speed_dps, read_joint, send_chunk, others_moved=lambda: 0.0,
                  chunk_deg=ROTATE_CHUNK_DEG, tol_deg=ROTATE_TOL_DEG, clock=time.time) -> dict:
    """Turn one joint by delta_deg (any size, signed) as a series of absolute-angle moves.

    read_joint() -> current angle in (-180, 180];  send_chunk(target_deg, timeout_s) -> result dict.
    Position targets fail safe: if the caller dies mid-way the arm finishes the current chunk and
    stops. Each chunk is verified (angle reached, took about as long as the speed implies, other
    joints still); the first anomaly ends the rotation and nothing is retried.
    """
    n = max(1, int(-(-abs(delta_deg) // chunk_deg)))
    step = delta_deg / n
    done = 0.0
    for i in range(n):
        before = read_joint()
        expect_s = abs(step) / speed_dps
        t0 = clock()
        res = send_chunk(safety.wrap_deg(before + step), 2.0 * expect_s + 5.0)
        took = clock() - t0
        moved = safety.wrap_deg(read_joint() - before)
        done += moved
        out = {"rotated_deg": done, "chunks": i + 1, "of": n}
        if not res.get("ok"):
            return dict(res, **out)
        if abs(moved - step) > tol_deg:
            return dict(out, ok=False, event="ROTATION_MISMATCH (asked %.1f, got %.1f deg)" % (step, moved))
        if abs(step) > 20.0 and took < ROTATE_MIN_TIME_FRAC * expect_s:
            return dict(out, ok=False, event="SPEED_LIMIT_NOT_HONOURED (%.1f deg in %.2f s, expected >= %.2f s)"
                        % (abs(step), took, expect_s))
        if others_moved() > tol_deg:
            return dict(out, ok=False, event="OTHER_JOINTS_MOVED")
    return {"ok": True, "event": "ACTION_END", "rotated_deg": done, "chunks": n, "of": n}


def rotate_joint(k, joint_index: int, delta_deg: float, speed_dps=None) -> dict:
    """Spin a continuous joint (0-based index; J1, J4 or J6) by delta_deg at a capped speed.

    UNVERIFIED ON THE ARM (2026-11-09): that firmware 2.5.2 applies JOINT_CONSTRAINT_SPEED to a
    reach_joint_angles action, and that it takes the short way to a wrapped target. rotate_chunks
    detects both failures on the first chunk; test with a small angle first.
    """
    from kortex_api.autogen.messages import Base_pb2
    if safety.JOINT_LIMITS_DEG[joint_index] is not None:
        raise safety.SafetyError("J%d is not a continuous joint" % (joint_index + 1))
    speed = safety.clamp_speed(speed_dps or k.cfg.max_ang_speed, k.cfg.max_ang_speed)
    start = state.joint_angles(k)

    def send_chunk(target_deg, timeout_s):
        angles = state.joint_angles(k)
        angles[joint_index] = target_deg
        safety.check_joints(angles)
        action = Base_pb2.Action()
        action.name = "kinova_cli_rotate_joint"
        action.application_data = ""
        for i, a in enumerate(angles):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = a
        action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_SPEED
        action.reach_joint_angles.constraint.value = speed
        return _run(k, lambda: k.base.ExecuteAction(action), timeout=timeout_s)

    def others_moved():
        now = state.joint_angles(k)
        return max(abs(safety.wrap_deg(a - b)) for i, (a, b) in enumerate(zip(now, start)) if i != joint_index)

    return rotate_chunks(delta_deg, speed, lambda: state.joint_angles(k)[joint_index], send_chunk, others_moved)
