import pyrealsense2 as rs
import numpy as np
import cv2

# --- Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Warm up
print("Warming up camera...")
for _ in range(60):
    pipeline.wait_for_frames()

# Lock color sensor
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 150)
color_sensor.set_option(rs.option.gain, 64)
profile.get_device().query_sensors()[0].set_option(rs.option.enable_auto_exposure, 0)
print("Camera locked. Press Q to quit.")

align = rs.align(rs.stream.color)

decimation       = rs.decimation_filter()
depth_to_disp    = rs.disparity_transform(True)
spatial          = rs.spatial_filter()
temporal         = rs.temporal_filter()
disp_to_depth    = rs.disparity_transform(False)
hole_filling     = rs.hole_filling_filter()
decimation.set_option(rs.option.filter_magnitude, 2)

CLIP_MIN, CLIP_MAX = 0.1, 3.0

depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

while True:
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Filter pipeline
    depth_frame = decimation.process(depth_frame)
    depth_frame = depth_to_disp.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = disp_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    depth   = np.asanyarray(depth_frame.get_data())
    color   = np.asanyarray(color_frame.get_data())
    depth_m = depth * depth_scale

    # Depth colormap (red=close, blue=far)
    depth_clipped = np.clip(depth_m, CLIP_MIN, CLIP_MAX)
    depth_vis     = ((depth_clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * 255).astype(np.uint8)
    depth_color   = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_JET)
    depth_color[depth_m == 0] = 0

    # Resize color to match depth size
    dh, dw = depth_color.shape[:2]
    color_resized = cv2.resize(color, (dw, dh))

    # Center distance
    cx, cy      = dw // 2, dh // 2
    center_dist = depth_frame.as_depth_frame().get_distance(cx, cy)

    # Close/Far label
    if center_dist <= 0:
        status, color_box = "NO DATA", (128, 128, 128)
    elif center_dist < 0.5:
        status, color_box = "VERY CLOSE", (0, 0, 255)
    elif center_dist < 1.5:
        status, color_box = "CLOSE", (0, 165, 255)
    elif center_dist < 2.5:
        status, color_box = "MID", (0, 255, 0)
    else:
        status, color_box = "FAR", (255, 0, 0)

    dist_text = f"{center_dist*100:.1f} cm  [{status}]"

    # Draw on depth panel
    cv2.putText(depth_color, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.circle(depth_color, (cx, cy), 6, (255,255,255), -1)

    # Draw on color panel
    cv2.putText(color_resized, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box, 2)
    cv2.circle(color_resized, (cx, cy), 6, color_box, -1)

    # Single window: color left | depth right
    combined = np.hstack([color_resized, depth_color])
    cv2.imshow("Color  |  Depth Map  --  Press Q", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
