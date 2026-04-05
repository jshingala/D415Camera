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

## What camera_test.py Does (Current State)
1. Starts the D415 pipeline with color + depth streams at 640x480 @ 30fps
2. Warms up for 60 frames then locks exposure and white balance for stable image
3. Every frame: aligns depth to color, applies full filter pipeline
4. Displays one window with color on the left and depth heatmap on the right
5. Shows center distance in cm and CLOSE/MID/FAR status on both panels
6. Press Q to quit

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pyrealsense2` | Talk to the D415 camera |
| `numpy` | Math/array operations on depth data |
| `opencv` (cv2) | Frame visualization and image processing |
| `open3d` | 3D point cloud and dimension extraction |

---

## Environment Info
- Machine: M1 MacBook Air
- Python: 3.12 (conda env: `realsense`)
- Conda path: `/opt/anaconda3/envs/realsense/`
- pyrealsense2 path: `/opt/anaconda3/envs/realsense/lib/python3.12/site-packages/pyrealsense2/`
- Camera: Intel RealSense D415 (firmware 50.c7, USB)
