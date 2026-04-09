# MHSF Triangle Detection

Real-time AI object detection and inspection system built for **NVIDIA Jetson Orin Nano**. Uses YOLO deep learning models with a PySide6 GUI for live video inspection, class filtering, center-line distance measurement, and automated relay control via Modbus TCP.

---

## Features

| Category | Details |
|---|---|
| **AI Detection** | Real-time YOLO inference with bounding boxes, confidence scores, and per-class colour coding |
| **Multi-Camera** | Auto-scan and select from connected USB/CSI cameras |
| **Filtering** | Adjustable confidence threshold (0–100%) and per-class on/off toggles |
| **Center Lines** | Frame center & detection center overlay with pixel-distance readout |
| **Relay Control** | 8-channel Modbus TCP relay mapping — trigger relays based on detected class + distance range |
| **Auto-Retry** | Relay connection with configurable retry count and delay |
| **Zoom** | Mouse-wheel zoom on the video feed; double-click for fullscreen (ESC button overlay to exit) |
| **Frame Capture** | Save the current annotated frame as PNG / JPG / BMP |
| **Config Persistence** | Save / load all settings to JSON; auto-loads on startup |
| **Dark Theme** | Consistent dark palette via Qt Fusion style |
| **Auto-Start** | Systemd service for headless boot on Jetson |

---

## UI Layout

```
┌─ Status Bar ──────────────────────────────────────────────────┐
│  FPS: 30.0    Detections: 2    Ready                          │
├───────────────┬───────────────────────────────────────────────┤
│  📹 Monitor   │  ⚡ Relay                                     │
├───────┬───────┴───────────────────────────────────────────────┤
│       │                                                       │
│ 🤖    │                                                       │
│ Model │              Video Feed                               │
│ & Cfg │         (zoom / fullscreen)                           │
│       │                                                       │
│ 📹    │                                                       │
│Camera │     🟡 Frame Center  🔵 Detection Center              │
│       │          ↕ distance (px)                              │
│ 🎛️    │                                                       │
│Filters│                                                       │
│       │                                                       │
│ 📊    │                                                       │
│Detect.│                                                       │
│ Table │                                                       │
└───────┴───────────────────────────────────────────────────────┘
```

The **Monitor** tab has a scrollable left sidebar (280 px) with four cards (Model, Camera, Filters, Detections) and the video feed filling the rest. The **Relay** tab provides host/port connection, an 8-row class→channel mapping table with min/max distance, and a match status banner.

---

## Project Structure

```
MHSF_TriangleDetection/
├── main.py              # MainWindow UI, app entry point
├── camera.py            # Camera discovery, backend selection, FPS normalisation
├── detection.py         # YOLO model loading, inference, detection extraction
├── workers.py           # VideoWorker QThread (capture → inference pipeline)
├── widgets.py           # ZoomableLabel, fullscreen window with ESC overlay
├── relay_control.py     # Relay connection worker, mapping evaluation
├── Relay_B.py           # Waveshare Modbus POE relay driver (8-ch TCP)
├── requirements.txt     # Python package dependencies
├── setup.sh             # Automated Jetson setup & auto-start installer
├── app_config.json      # Persisted application settings (auto-generated)
├── 2D_Shape.pt          # Custom trained model
├── yolo11n.pt           # YOLO11 Nano model
└── README.md            # This file
```

### Module Responsibilities

| Module | Role |
|---|---|
| `main.py` | Builds the PySide6 GUI, wires signals/slots, handles config save/load |
| `camera.py` | Scans `/dev/video*`, picks the right OpenCV backend per OS, normalises FPS |
| `detection.py` | Wraps Ultralytics YOLO — CUDA check, model load, class list, inference |
| `workers.py` | `VideoWorker` QThread — reads frames, runs inference, emits signals |
| `widgets.py` | `ZoomableLabel` (wheel zoom, double-click fullscreen, ESC overlay) |
| `relay_control.py` | `RelayConnectionWorker` QThread, default mappings, `evaluate_mappings` |
| `Relay_B.py` | Low-level Modbus TCP socket driver for Waveshare 8-channel relay |

---

## System Requirements

| Component | Requirement |
|---|---|
| **Hardware** | NVIDIA Jetson Orin Nano (or any Jetson with JetPack) |
| **OS** | JetPack 5.x / 6.x (Ubuntu 20.04 / 22.04 based) |
| **Python** | 3.8+ (ships with JetPack) |
| **GPU** | Jetson integrated GPU with CUDA (auto-detected) |
| **Camera** | USB webcam or CSI camera |
| **Relay** *(optional)* | Waveshare Modbus POE Ethernet Relay (8-ch) |

> The app also runs on standard x86 Linux / Windows with a discrete NVIDIA GPU or CPU-only mode.

---

## Installation

### Option A — Automated Setup (Recommended for Jetson)

The included `setup.sh` script handles everything in one command:

```bash
cd /home/orin_nano/Dev_AI/MHSF_TriangleDetection

# Make executable (first time only)
chmod +x setup.sh

# Run with sudo
sudo ./setup.sh
```

**What the script does:**

| Step | Action |
|---|---|
| 1 | Detects JetPack version (dpkg, L4T, nv_jetson_release, jtop) |
| 2 | Installs system libs (`libgl1`, `libxcb-*`, `libxkbcommon-x11-0`, etc.) and pip packages from `requirements.txt` |
| 3 | Verifies each Python module imports correctly (`cv2`, `numpy`, `PySide6`, `ultralytics`, `torch`) |
| 4 | Creates and enables a **systemd service** (`mhsf-triangle-detection`) so the app auto-starts after every boot |

After setup, the app will start automatically on the next reboot. To manage it manually:

```bash
# Start / stop / restart
sudo systemctl start   mhsf-triangle-detection
sudo systemctl stop    mhsf-triangle-detection
sudo systemctl restart mhsf-triangle-detection

# Check status
sudo systemctl status  mhsf-triangle-detection

# View live logs
journalctl -u mhsf-triangle-detection -f

# Disable auto-start
sudo systemctl disable mhsf-triangle-detection
```

---

### Option B — Manual Installation

#### 1. Install System Dependencies

```bash
sudo apt update && sudo apt install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-shape0 \
    libdbus-1-3
```

#### 2. Install Python Packages

```bash
cd /home/orin_nano/Dev_AI/MHSF_TriangleDetection
pip install -r requirements.txt
```

**`requirements.txt` contents:**

| Package | Purpose |
|---|---|
| `PySide6 >= 6.5` | Qt GUI framework |
| `ultralytics >= 8.0` | YOLO model training & inference |
| `opencv-python >= 4.8` | Video capture & image processing |
| `numpy >= 1.24` | Numerical arrays |

> **PyTorch on Jetson:** `torch` and `torchvision` are typically pre-installed by JetPack. If missing, install the official NVIDIA wheel:
> https://forums.developer.nvidia.com/t/pytorch-for-jetson/

#### 3. Verify Installation

```bash
python3 -c "
import cv2, numpy, PySide6, ultralytics
print('cv2       :', cv2.__version__)
print('numpy     :', numpy.__version__)
print('PySide6   :', PySide6.__version__)
print('ultralytics:', ultralytics.__version__)

try:
    import torch
    print('torch     :', torch.__version__)
    print('CUDA      :', torch.cuda.is_available())
except ImportError:
    print('torch     : NOT INSTALLED')

print('All OK!')
"
```

#### 4. Prepare a YOLO Model

The project includes `2D_Shape.pt` and `yolo11n.pt`. To download a general-purpose model:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Available sizes: `yolov8n` (nano) → `yolov8s` → `yolov8m` → `yolov8l` → `yolov8x` (largest).

#### 5. Run the Application

```bash
cd /home/orin_nano/Dev_AI/MHSF_TriangleDetection
python3 main.py
```

---

## Quick Start Guide

1. **Load a Model** — Click **📁 Load Model** in the sidebar and select a `.pt` file
2. **Select Camera** — Pick your camera from the dropdown, or click **🔄 Scan Cameras**
3. **Start Stream** — Click **▶ Start** to begin live detection
4. **Adjust Filters** — Slide the confidence threshold; toggle class checkboxes
5. **Enable Center Lines** — Check **🎯 Show Center Lines** to see distance overlays
6. **Configure Relays** *(optional)* — Switch to the **⚡ Relay** tab, enter relay host/port, connect, and map classes to channels with distance ranges
7. **Capture Frame** — Click **📷 Capture Frame** to save a screenshot
8. **Save Config** — Click **💾 Save** to persist all settings; they auto-load on next launch
9. **Fullscreen** — Double-click the video feed; press **ESC** or click the ESC button to exit

---

## Configuration

Settings are saved to `app_config.json` in the project directory. Example:

```json
{
  "model_path": "/home/orin_nano/Dev_AI/MHSF_TriangleDetection/2D_Shape.pt",
  "camera_index": 0,
  "confidence_threshold": 25,
  "selected_classes": ["triangle", "circle"],
  "compute_device": "cuda",
  "show_center_overlay": true,
  "relay_host": "192.168.1.201",
  "relay_port": 502,
  "relay_mappings": [
    { "class": "triangle", "channel": 1, "distance_min": 0, "distance_max": 200 }
  ],
  "auto_connect_relay": false
}
```

The config auto-loads on startup. If the saved camera and model are available, the stream starts automatically.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **`Could not load the Qt platform plugin "xcb"`** | Install XCB libs: `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0` |
| **No cameras detected** | Check `ls /dev/video*`. Ensure user is in the `video` group: `sudo usermod -aG video $USER` |
| **PyTorch not found** | On Jetson, install the NVIDIA `.whl` from the developer forum (not from PyPI) |
| **CUDA not detected** | Verify with `nvcc --version`. Ensure JetPack is fully installed |
| **Slow FPS** | Use a smaller model (`yolov8n.pt`). Ensure `cuda` is selected in Compute dropdown |
| **Relay won't connect** | Verify the relay's IP/port. Check network with `ping 192.168.1.201` |
| **App won't auto-start** | Check service status: `sudo systemctl status mhsf-triangle-detection`. Check logs: `journalctl -u mhsf-triangle-detection` |
| **Fullscreen stuck** | Press `Escape` key or click the ESC button in the top-right corner. Double-click also exits. |

---

## Service Management

After running `setup.sh`, the app is registered as a systemd service:

| Command | Action |
|---|---|
| `sudo systemctl start mhsf-triangle-detection` | Start the app now |
| `sudo systemctl stop mhsf-triangle-detection` | Stop the app |
| `sudo systemctl restart mhsf-triangle-detection` | Restart |
| `sudo systemctl enable mhsf-triangle-detection` | Enable auto-start on boot |
| `sudo systemctl disable mhsf-triangle-detection` | Disable auto-start |
| `journalctl -u mhsf-triangle-detection -f` | View live logs |

---

## License and Credits

This project uses:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection
- [PySide6 / Qt](https://www.qt.io/) — GUI framework
- [OpenCV](https://opencv.org/) — video capture & image processing
- [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) — Jetson platform SDK
