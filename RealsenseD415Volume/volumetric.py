import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

model        = YOLO("yolo11n.pt")
dist_history = defaultdict(lambda: deque(maxlen=15))
dim_stable   = {}   # track_id -> [L, W, H] EMA stable value
EMA_ALPHA    = 0.08 # lower = smoother but slower to update

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile  = pipeline.start(config)

print("Warming up...")
for _ in range(60):
    pipeline.wait_for_frames()

color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 150)
color_sensor.set_option(rs.option.gain, 64)
profile.get_device().query_sensors()[0].set_option(rs.option.enable_auto_exposure, 0)
print("Ready. Press Q to quit.")

align         = rs.align(rs.stream.color)
decimation    = rs.decimation_filter()
depth_to_disp = rs.disparity_transform(True)
spatial       = rs.spatial_filter()
temporal      = rs.temporal_filter()
disp_to_depth = rs.disparity_transform(False)
hole_filling  = rs.hole_filling_filter()
decimation.set_option(rs.option.filter_magnitude, 2)

CLIP_MIN, CLIP_MAX = 0.4, 3.0
DIM_FACTOR = 5000

depth_scale      = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()


def get_dimensions_and_fill(depth_m, x1, y1, x2, y2):
    fh, fw = depth_m.shape
    # Shrink box 5% to reduce YOLO padding
    px = max(3, int((x2 - x1) * 0.05))
    py = max(3, int((y2 - y1) * 0.05))
    x1, y1 = max(1, x1 + px), max(1, y1 + py)
    x2, y2 = min(fw-2, x2 - px), min(fh-2, y2 - py)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None, None, "UNKNOWN"

    roi = depth_m[y1:y2, x1:x2].astype(float)

    # Object depth from center patch
    cy_r, cx_r = (y2-y1)//2, (x2-x1)//2
    m = max(4, min(cy_r//3, cx_r//3))
    patch = roi[cy_r-m:cy_r+m, cx_r-m:cx_r+m]
    pv = patch[patch > 0]
    if len(pv) < 5:
        return None, None, None, "UNKNOWN"
    obj_depth = float(np.median(pv))

    # Keep only object pixels ±10cm
    obj_mask = (roi > 0) & (np.abs(roi - obj_depth) < 0.10)
    if obj_mask.sum() < 20:
        return None, None, None, "UNKNOWN"

    ys_idx, xs_idx = np.where(obj_mask)
    xs_abs = (xs_idx + x1).astype(float)
    ys_abs = (ys_idx + y1).astype(float)
    ds     = roi[obj_mask]

    # Deproject to 3D
    X = (xs_abs - depth_intrinsics.ppx) * ds / depth_intrinsics.fx
    Y = (ys_abs - depth_intrinsics.ppy) * ds / depth_intrinsics.fy

    # Use 10th-90th percentile — tighter than before to cut outliers
    xlo, xhi = np.percentile(X, 10), np.percentile(X, 90)
    ylo, yhi = np.percentile(Y, 10), np.percentile(Y, 90)
    W = round((xhi - xlo) * 100, 1)
    H = round((yhi - ylo) * 100, 1)

    # Hollow / Solid
    cx1 = x1 + (x2-x1)//3;  cy1 = y1 + (y2-y1)//3
    cx2 = x2 - (x2-x1)//3;  cy2 = y2 - (y2-y1)//3
    inner = depth_m[cy1:cy2, cx1:cx2]
    iv = inner[inner > 0]
    fill_status = "UNKNOWN"
    if len(iv) > 5:
        diff_cm = (float(np.mean(iv)) - obj_depth) * 100
        fill_status = "EMPTY/HOLLOW" if diff_cm > 3 else "SOLID"

    return None, W, H, fill_status


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

    depth_clipped = np.clip(depth_m, CLIP_MIN, CLIP_MAX)
    depth_vis     = ((depth_clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * 255).astype(np.uint8)
    depth_color   = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_JET)
    depth_color[depth_m == 0] = 0

    dh, dw        = depth_color.shape[:2]
    color_resized = cv2.resize(color, (dw, dh))

    results     = model.track(color_resized, persist=True, verbose=False, conf=0.5)[0]
    valid_boxes = [b for b in results.boxes if b.id is not None]

    header = np.zeros((80, 1280, 3), dtype=np.uint8)
    cv2.putText(header, f"Objects: {len(valid_boxes)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    h_xpos = 10
    for box in valid_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf     = float(box.conf[0])
        label    = model.names[int(box.cls[0])]
        track_id = int(box.id[0])

        # Smoothed center distance
        bx, by   = (x1 + x2) // 2, (y1 + y2) // 2
        raw_dist = depth_frame.as_depth_frame().get_distance(bx, by)
        if raw_dist > 0:
            dist_history[track_id].append(raw_dist)
        dist = float(np.median(dist_history[track_id])) if dist_history[track_id] else 0

        # Dimensions + fill — EMA smoothing with outlier rejection
        L, W, H, fill = get_dimensions_and_fill(depth_m, x1, y1, x2, y2)

        dim_text  = "Measuring..."
        fill_text = fill
        dim_weight = 0

        if W and H:
            if track_id not in dim_stable:
                dim_stable[track_id] = [W, H]  # initialise
            else:
                sW, sH = dim_stable[track_id]
                if abs(W - sW) / max(sW, 1) < 0.3:
                    sW = sW * (1 - EMA_ALPHA) + W * EMA_ALPHA
                if abs(H - sH) / max(sH, 1) < 0.3:
                    sH = sH * (1 - EMA_ALPHA) + H * EMA_ALPHA
                dim_stable[track_id] = [sW, sH]

        if track_id in dim_stable:
            W, H       = [round(v, 1) for v in dim_stable[track_id]]
            dim_text   = f"W:{W} H:{H}cm  (L=needs top-down camera)"

        # Box color by distance
        if dist <= 0:    bcolor = (128, 128, 128)
        elif dist < 0.8: bcolor = (0, 0, 255)
        elif dist < 2.0: bcolor = (0, 165, 255)
        else:            bcolor = (255, 0, 0)

        # Fill status color
        fcolor = (0, 255, 255) if fill_text == "SOLID" else (0, 100, 255) if fill_text == "EMPTY/HOLLOW" else (180, 180, 180)

        # Draw on color panel
        cv2.rectangle(color_resized, (x1, y1), (x2, y2), bcolor, 2)
        cv2.putText(color_resized, f"#{track_id} {label} {conf:.0%} {dist*100:.0f}cm",
                    (x1, max(y1 - 35, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(color_resized, dim_text,
                    (x1, max(y1 - 20, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        cv2.putText(color_resized, fill_text,
                    (x1, max(y1 - 6, 38)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fcolor, 2)

        # Mirror on depth panel
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(depth_color, fill_text, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fcolor, 2)

        # Header row
        h_info = f"#{track_id} {label}: W:{W} H:{H}cm [{fill_text}]"
        cv2.putText(header, h_info, (h_xpos, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        h_xpos += 430

    combined = np.hstack([color_resized, depth_color])
    combined = cv2.resize(combined, (1280, 540))
    combined = np.vstack([header, combined])
    cv2.imshow("Volumetric  |  Press Q", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
