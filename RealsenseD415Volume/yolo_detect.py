import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- YOLO11 nano ---
model = YOLO("yolo11n.pt")

# Smooth distance per track ID over last N frames
from collections import defaultdict, deque
dist_history = defaultdict(lambda: deque(maxlen=10))

# --- RealSense Setup (same as camera_test.py) ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

print("Warming up camera...")
for _ in range(60):
    pipeline.wait_for_frames()

color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 150)
color_sensor.set_option(rs.option.gain, 64)
profile.get_device().query_sensors()[0].set_option(rs.option.enable_auto_exposure, 0)
print("Camera locked. YOLO running. Press Q to quit.")

align         = rs.align(rs.stream.color)
decimation    = rs.decimation_filter()
depth_to_disp = rs.disparity_transform(True)
spatial       = rs.spatial_filter()
temporal      = rs.temporal_filter()
disp_to_depth = rs.disparity_transform(False)
hole_filling  = rs.hole_filling_filter()
decimation.set_option(rs.option.filter_magnitude, 2)

CLIP_MIN, CLIP_MAX = 0.4, 3.0
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

while True:
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    depth_frame = decimation.process(depth_frame)
    depth_frame = depth_to_disp.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = disp_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    depth   = np.asanyarray(depth_frame.get_data())
    color   = np.asanyarray(color_frame.get_data())
    depth_m = depth * depth_scale

    # Depth colormap
    depth_clipped = np.clip(depth_m, CLIP_MIN, CLIP_MAX)
    depth_vis     = ((depth_clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * 255).astype(np.uint8)
    depth_color   = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_JET)
    depth_color[depth_m == 0] = 0

    dh, dw        = depth_color.shape[:2]
    color_resized = cv2.resize(color, (dw, dh))

    # --- YOLO tracking (stable IDs, no flicker) ---
    results = model.track(color_resized, persist=True, verbose=False, conf=0.5)[0]
    for box in results.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf    = float(box.conf[0])
        label   = model.names[int(box.cls[0])]
        track_id = int(box.id[0])

        # Distance at center of box — smoothed over last 10 frames
        bx, by = (x1 + x2) // 2, (y1 + y2) // 2
        raw_dist = depth_frame.as_depth_frame().get_distance(bx, by)
        if raw_dist > 0:
            dist_history[track_id].append(raw_dist)
        dist = np.mean(dist_history[track_id]) if dist_history[track_id] else 0

        dist_text = f"#{track_id} {label} {conf:.0%} | {dist*100:.0f}cm" if dist > 0 else f"#{track_id} {label} {conf:.0%} | --"

        if dist <= 0:
            bcolor = (128, 128, 128)
        elif dist < 0.8:
            bcolor = (0, 0, 255)     # red = close
        elif dist < 2.0:
            bcolor = (0, 165, 255)   # orange = mid
        else:
            bcolor = (255, 0, 0)     # blue = far

        cv2.rectangle(color_resized, (x1, y1), (x2, y2), bcolor, 2)
        cv2.rectangle(color_resized, (x1, y1 - 20), (x1 + len(dist_text) * 9, y1), bcolor, -1)
        cv2.putText(color_resized, dist_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 1)

    combined = np.hstack([color_resized, depth_color])
    cv2.imshow("YOLO + Depth  |  Press Q", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
