"""Read-only RealSense check (D435 verified, L515 attempted): list device + modes, try depth / colour / both, save previews."""
import sys, time
import numpy as np
import pyrealsense2 as rs

OUT = sys.argv[1] if len(sys.argv) > 1 else None
ctx = rs.context()
devs = ctx.query_devices()
print("devices:", len(devs))
if not len(devs):
    sys.exit(2)
d = devs[0]
print("  ", d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number),
      "fw", d.get_info(rs.camera_info.firmware_version), "usb", d.get_info(rs.camera_info.usb_type_descriptor))
for s in d.query_sensors():
    modes = sorted({(p.stream_name(), p.format().name, p.as_video_stream_profile().width(),
                     p.as_video_stream_profile().height(), p.fps())
                    for p in s.get_stream_profiles() if p.is_video_stream_profile()})
    print("  sensor:", s.get_info(rs.camera_info.name), "| modes:", modes[:12])

CONFIGS = [
    ("depth 640x480@30", [(rs.stream.depth, 640, 480, rs.format.z16, 30)]),
    ("color 1280x720@15", [(rs.stream.color, 1280, 720, rs.format.bgr8, 15)]),
    ("depth 640x480@30 + color 1280x720@30", [(rs.stream.depth, 640, 480, rs.format.z16, 30),
                                             (rs.stream.color, 1280, 720, rs.format.bgr8, 30)]),
]
ok = 0
for label, streams in CONFIGS:
    pipe, cfg = rs.pipeline(ctx), rs.config()
    for st in streams:
        cfg.enable_stream(*st)
    print(f"\n== {label}")
    try:
        prof = pipe.start(cfg)
    except Exception as e:
        print("   start FAILED:", e)
        continue
    try:
        t0, n, fs = time.time(), 0, None
        while time.time() - t0 < 6:
            got, f = pipe.try_wait_for_frames(1000)
            if got:
                n, fs = n + 1, f
        print(f"   frames in 6 s: {n}")
        if fs is None:
            continue
        ok += 1
        dep, col = fs.get_depth_frame(), fs.get_color_frame()
        if dep:
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            z = np.asanyarray(dep.get_data()).astype(np.float32) * scale
            v = z[z > 0]
            i = dep.profile.as_video_stream_profile().get_intrinsics()
            print(f"   depth {z.shape} valid={v.size / z.size:.0%} min={v.min():.3f} med={np.median(v):.3f} max={v.max():.3f} m"
                  f" | K fx={i.fx:.1f} fy={i.fy:.1f} cx={i.ppx:.1f} cy={i.ppy:.1f}")
        if col:
            c = np.asanyarray(col.get_data())
            i = col.profile.as_video_stream_profile().get_intrinsics()
            print(f"   color {c.shape} mean={c.mean():.1f} | K fx={i.fx:.1f} fy={i.fy:.1f} cx={i.ppx:.1f} cy={i.ppy:.1f}")
        if OUT:
            import cv2
            tag = label.split()[0] if len(streams) == 1 else "both"
            if dep:
                cv2.imwrite(f"{OUT}_{tag}_depth.png", cv2.applyColorMap(
                    cv2.convertScaleAbs(z, alpha=255 / max(1e-3, np.percentile(v, 99))), cv2.COLORMAP_TURBO))
            if col:
                cv2.imwrite(f"{OUT}_{tag}_color.png", c)
    finally:
        pipe.stop()
        time.sleep(1)
print(f"\n{ok}/{len(CONFIGS)} configs delivered frames")
sys.exit(0 if ok else 1)
