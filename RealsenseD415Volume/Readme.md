# RealSense D415 Volumetric Dimension System

## Project Goal
Static camera + weighing machine setup that measures package dimensions and bills by actual weight vs DIM weight (whichever is higher).

---

## Setup Steps

### Step 1 — Verified Camera is Connected
Checked USB connection via terminal. D415 detected at USB port.
```bash
system_profiler SPUSBDataType | grep -A5 "RealSense"
```

### Step 2 — Checked Python Version
Default `python3` on the Mac was 3.14 (Homebrew). `pyrealsense2` does not support 3.14.
```bash
python3 --version
```

### Step 3 — Found pyrealsense2 was Broken
The library was installed but compiled for Python 3.12, causing an import error on 3.14.

### Step 4 — Used Conda (Python 3.12)
Conda was already installed with Python 3.12 in the base environment.
```bash
conda --version
```

### Step 5 — Created a Dedicated Conda Environment
```bash
conda create -n realsense python=3.12 -y
```

### Step 6 — Installed All Required Libraries
```bash
conda activate realsense
conda install -c conda-forge pyrealsense2 numpy opencv -y
pip install open3d
```

### Step 7 — Verified Camera Works (1 device found)
```bash
conda run -n realsense python -c "import pyrealsense2 as rs; ctx = rs.context(); print(len(ctx.devices), 'device found')"
```
Output: `1 device found`

### Step 8 — Hit USB Permission Error
Running `python camera_test.py` caused a segmentation fault. Root cause was macOS blocking USB access to the camera.
```
RS2_USB_STATUS_ACCESS — failed to claim usb interface
```

### Step 9 — Confirmed Camera Works with sudo
Used homebrew's CLI tool with sudo to verify the camera is fully functional.
```bash
sudo rs-enumerate-devices
```
Output: D415 detected, all depth and color stream profiles listed.

### Step 10 — Run Python with sudo Using Full Conda Path
Since macOS blocks USB without sudo, run scripts using the full conda Python path:
```bash
sudo /opt/anaconda3/envs/realsense/bin/python camera_test.py
```

### Step 11 — Live Camera Feed Working
Got first live feed showing color and depth in separate windows. Fixed depth visualization using `cv2.normalize` so near/far colors are accurate.

### Step 12 — Added Depth Accuracy Filters
Applied Intel's recommended filter pipeline in the correct order for stereo cameras:
1. Decimation filter (reduces noise, magnitude = 2)
2. Depth → Disparity transform
3. Spatial filter (edge-preserving smoothing)
4. Temporal filter (reduces flicker between frames)
5. Disparity → Depth transform
6. Hole filling (fills missing pixels)

Also aligned depth frame to color frame using `rs.align` so both frames match pixel-for-pixel.
Clipped depth range to **0.1m – 3.0m** (D415 optimal range) for accurate color mapping.

### Step 13 — Fixed Unstable/Flickering Color Feed
Camera auto-exposure and auto white balance were causing constant flickering. Fixed by:
- Warming up camera for 60 frames
- Locking color sensor: disabled auto exposure + auto white balance, set fixed exposure (150µs) and gain (64)
- Locking depth sensor: disabled auto exposure

### Step 14 — Single Window with Two Panels
Combined everything into one window:
- **Left panel** — normal color feed with distance label and close/far status
- **Right panel** — depth heatmap (red = close, blue = far, black = no data)

Distance status zones:
- VERY CLOSE = < 50 cm (red)
- CLOSE = < 1.5 m (orange)
- MID = < 2.5 m (green)
- FAR = > 2.5 m (blue)

### How to Run
```bash
sudo /opt/anaconda3/envs/realsense/bin/python camera_test.py
```

---

### Step 15 — Created yolo_detect.py
Added YOLO11 nano model on top of camera feed for real-time object detection.
```bash
pip install ultralytics
```
- Model: `yolo11n.pt` (auto-downloaded on first run)
- Used `model.track(persist=True)` instead of `model()` for stable object IDs across frames
- Confidence threshold set to 0.5 to reduce false detections
- Distance smoothed with median over 10 frames per track ID
- Box color changes by distance: red = close, orange = mid, blue = far

### Step 16 — Created volumetric.py
New file for dimension measurement. Key decisions made:

**What W and H mean from the camera:**
- **W** = horizontal width (left → right) — measurable from front camera
- **H** = vertical height (top → bottom) — measurable from front camera
- **L** = depth front-to-back — NOT measurable from front-facing camera (needs top-down setup)

**Measurement approach:**
1. Get object surface depth from center patch of bounding box
2. Create ±10cm depth mask to isolate only object pixels (removes background)
3. Deproject masked pixels to real 3D coordinates using camera intrinsics
4. Use 10th–90th percentile of X and Y extents to cut outliers
5. Smooth W and H using EMA (alpha=0.08) with 30% jump rejection — prevents flickering

**Hollow/Solid detection:**
- Compares inner center depth vs object surface depth
- If center is >3cm deeper than surface → EMPTY/HOLLOW, else SOLID

### Step 17 — Tuned for Accuracy
- YOLO bounding box shrunk 5% inward on all sides to remove padding inflation
- Switched from median smoothing to EMA for smoother real-time values
- Outlier rejection: updates only if new value is within 30% of current stable value
- Clipped depth minimum raised from 0.1m to 0.4m (D415 reliable range)

### Step 18 — Single Object Mode
Limited detection to one object at a time — the largest bounding box in frame.
Prevents confusion when multiple objects are visible.

---

## Files

| File | Purpose |
|---|---|
| `camera_test.py` | Basic camera test: color + depth feed, close/far label |
| `yolo_detect.py` | YOLO object detection + distance per object |
| `volumetric.py` | Full pipeline: YOLO + W/H measurement + hollow detection |

## How to Run Each File
```bash
sudo /opt/anaconda3/envs/realsense/bin/python camera_test.py
sudo /opt/anaconda3/envs/realsense/bin/python yolo_detect.py
sudo /opt/anaconda3/envs/realsense/bin/python volumetric.py
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pyrealsense2` | Talk to the D415 camera |
| `numpy` | Math/array operations on depth data |
| `opencv` (cv2) | Frame visualization and image processing |
| `ultralytics` | YOLO11 object detection and tracking |

---

## Environment Info
- Machine: M1 MacBook Air
- Python: 3.12 (conda env: `realsense`)
- Conda path: `/opt/anaconda3/envs/realsense/`
- pyrealsense2 path: `/opt/anaconda3/envs/realsense/lib/python3.12/site-packages/pyrealsense2/`
- Camera: Intel RealSense D415 (firmware 50.c7, USB)
