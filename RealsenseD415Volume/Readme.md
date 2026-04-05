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
Updated `camera_test.py` to show two real-time windows:
- **Color** — normal RGB camera feed
- **Depth** — heatmap where red = close, blue = far (normalized per frame)

Fixed depth visualization by using `cv2.normalize` instead of fixed alpha scaling so near/far colors are accurate.

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
