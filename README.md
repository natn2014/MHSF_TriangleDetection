# Universal AI Inspection

A real-time YOLO object detection application with an intuitive GUI for AI-powered video inspection, industrial monitoring, and quality control analysis.

## Concept

Universal AI Inspection is a desktop application that leverages YOLOv8 deep learning models to perform real-time object detection on video streams from webcams or connected cameras. Built with PySide6, it provides a user-friendly interface for:

- Loading custom YOLO detection models
- Monitoring live video feeds with AI-powered object detection
- Filtering detections by confidence threshold and class type
- Setting up alerts for specific object classes
- Capturing and analyzing detection statistics
- Saving screenshots of detections for documentation

The application is designed for industrial inspection, quality control, surveillance, and research applications where real-time AI analysis of video streams is needed.

## Features

### Core Detection
- **Real-time YOLO Detection** - Live video analysis with YOLOv8 models
- **Multi-Camera Support** - Auto-detection of available cameras
- **Bounding Box Visualization** - Color-coded detection boxes with confidence scores

### Customization & Control
- **Custom Model Loading** - Load any YOLOv8 .pt model files
- **Confidence Filtering** - Adjustable threshold slider (0-100%)
- **Class Selection** - Toggle individual object classes on/off
- **Dynamic FPS Adjustment** - Auto-normalize camera FPS (24, 30, or 60 fps)

### Monitoring & Alerts
- **Match Alerts** - Real-time notifications when specific classes are detected
- **Detection Statistics** - Live table showing class counts per frame
- **Color-Coded Row Matching** - Visual correlation between detections and statistics

### Utilities
- **Frame Capture** - Screenshot detections to PNG, JPG, or BMP
- **Performance Optimization** - Automatic frame skipping for smooth real-time performance
- **Status Display** - Real-time monitoring of detection status and statistics

## Installation

### System Requirements

- **OS**: Windows 10+ or Linux/macOS with Python 3.8+
- **Python**: 3.8 or higher
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster inference
- **Linux Dependencies**: Qt/XCB libraries for GUI support

## Quick Setup (Ubuntu/Debian)

🚀 **Automated Setup Script** - One command to set up everything:

```bash
# Clone the repository
git clone <repository-url>
cd Universal_AI_Inspection

# Run the automated setup script
./setup.sh
```

The setup script will:
- Install all required system dependencies (Qt/XCB libraries)
- Create and configure a Python virtual environment
- Install all Python packages from requirements.txt
- Verify the installation

## Manual Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd Universal_AI_Inspection

# Or extract the project folder
```

### Step 2: Install System Dependencies (Linux Only)

For Ubuntu/Debian systems, install Qt/XCB libraries:

```bash
sudo apt update && sudo apt install -y \
    libxcb-cursor0 \
    libxcb-cursor-dev \
    libxcb1-dev \
    libxkbcommon-x11-0 \
    libxkbcommon-x11-dev \
    python3-venv \
    python3-pip
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `PySide6==6.8.0.2` - GUI framework
- `opencv-python==4.13.0.92` - Video processing
- `ultralytics==8.4.14` - YOLO implementation
- `numpy==1.26.4` - Numerical computing
- `torch` & `torchvision` - Deep learning framework

### Step 5: Verify Installation

Test that all packages are properly installed:

```bash
python -c "
import cv2, numpy, PySide6, ultralytics
print('✓ All dependencies installed successfully!')
print('✓ PySide6 version:', PySide6.__version__)
print('✓ OpenCV version:', cv2.__version__)
"
```

### Step 5: Prepare YOLO Models

Download a YOLOv8 model file (.pt):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This will download and cache the nano model. You can also download other sizes:
- `yolov8n.pt` - Nano (smallest, fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

Alternatively, use custom-trained models in the same format.

### Step 6: Run the Application

```bash
# If using automated setup
./venv/bin/python main.py

# Or if virtual environment is activated
python main.py
```

The GUI window should open. You're ready to start inspecting!

## Project Structure

The codebase is organized into **modular files by function**:

```
Universal_AI_Inspection/
├── main.py              # UI Display & Controls – MainWindow, app entry point
├── camera.py            # Camera Input – camera discovery, video backend, FPS normalization
├── detection.py         # AI Object Detection – YOLO model loading, inference, detection extraction
├── relay_control.py     # Relay Control Logic – connection worker, mapping evaluation, channel control
├── workers.py           # Frame Capture & Processing – VideoWorker QThread (camera + inference pipeline)
├── widgets.py           # Custom Qt Widgets – ZoomableLabel with mouse-wheel zoom
├── Relay_B.py           # Hardware Output – Waveshare Modbus POE relay driver (8-channel TCP)
├── requirements.txt     # Python dependencies
├── setup.sh             # Automated setup script for Ubuntu/Debian
├── yolov8n.pt           # YOLOv8 Nano model
├── yolo11n.pt           # YOLO11 Nano model
├── yolo26n.pt           # YOLO26 Nano model
└── README.md            # This documentation
```

### Module Descriptions

| Module | Responsibility |
|---|---|
| **main.py** | Application entry point and `MainWindow` class. Builds the entire PySide6 GUI (tabs, controls, tables) and wires signals/slots between modules. |
| **camera.py** | Discovers available cameras (`find_cameras`), selects the correct video backend per OS, normalizes FPS to allowed values, and opens camera captures via OpenCV. |
| **detection.py** | Wraps Ultralytics YOLO: checks CUDA availability, loads `.pt` models, retrieves class names, runs inference, and extracts bounding-box detections. |
| **relay_control.py** | Manages relay hardware interaction: `RelayConnectionWorker` (QThread) for non-blocking connection, mapping evaluation (class → channel), and safe channel on/off. |
| **workers.py** | `VideoWorker` QThread that continuously reads frames from the camera, runs AI inference, and emits `frame_ready` / `status` signals back to the UI. |
| **widgets.py** | `ZoomableLabel` – a QLabel subclass with mouse-wheel zoom support and a `zoom_changed` signal. |
| **Relay_B.py** | Low-level Waveshare Modbus POE Ethernet relay driver. Communicates over TCP sockets to control up to 8 relay channels. |

## Quick Start

1. **Load a Model** - Click "Load Model" and select a YOLOv8 .pt file
2. **Select Camera** - Choose your camera from the dropdown
3. **Click Start** - Begin real-time detection
4. **Adjust Filters** - Use the confidence slider and class checkboxes to refine results
5. **Set Alert Class** - Select a class to monitor for alerts
6. **Capture Frames** - Click "Capture Screenshot" to save detection images

## Troubleshooting

**Qt XCB Platform Plugin Error (Linux):**
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```
- **Solution**: Run the automated setup script `./setup.sh` which installs required system dependencies
- **Manual fix**: Install Qt/XCB libraries: `sudo apt install -y libxcb-cursor0 libxcb-cursor-dev libxcb1-dev libxkbcommon-x11-0 libxkbcommon-x11-dev`

**No cameras detected:**
- Ensure your camera/webcam is properly connected
- Check camera permissions in Windows Settings
- Try restarting the application
- On Linux, check if user has access to `/dev/video*` devices

**YOLO import error:**
- Verify ultralytics is installed: `pip install --upgrade ultralytics`
- Update CUDA drivers if using GPU acceleration
- Check Python version compatibility (3.8+ required)

**Performance issues:**
- Use a smaller YOLO model (nano or small)
- Lower the FPS target
- Reduce video resolution if possible
- Ensure adequate GPU/CPU resources
- Close other resource-intensive applications

**Virtual Environment Issues:**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Recreate environment if corrupted: `rm -rf venv && python3 -m venv venv`
- Use the automated setup script for a fresh installation

## License and Credits

This project uses:
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics
- [PySide6](https://www.qt.io/) - Qt Framework
- [OpenCV](https://opencv.org/) - Computer Vision Library
