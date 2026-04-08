# YOLO Object Detection Inspection Application - Summary

## Overview
A real-time YOLO object detection application built with PySide6 that enables AI-powered video inspection with customizable filtering, alerting, and analysis capabilities.

## Key Features

### 1. Model Management
- **Load Custom YOLO Models** - File dialog to select .pt model files
- **Automatic Class Extraction** - Extracts and displays all classes from loaded model
- **Model Status Display** - Shows current model name and class count

### 2. Camera & Stream Control
- **Multi-Camera Support** - Auto-scans available cameras (0-9)
- **Dynamic Camera Selection** - Dropdown to select active camera
- **FPS Normalization** - Automatically detects and matches camera FPS to 24/30/60 fps
- **Real-time Frame Skipping** - Skips frames when inference slower than target FPS to maintain real-time performance
- **Start/Stop Controls** - Easy toggle for video stream

### 3. Object Detection Visualization
- **Bounding Box Drawing** - Uses QPainter to draw detection rectangles on video feed
- **Per-Class Color Coding** - Each class gets unique color (consistent across boxes and table)
- **Confidence Display** - Shows confidence percentage on each detection
- **Class Labels** - Displays class name with confidence score

### 4. Filtering System
- **Confidence Threshold Slider** - Adjustable 0-100% to filter low-confidence detections
- **Class Selection Checkboxes** - Scrollable panel to toggle each class visibility
- **All Classes Listed** - Dynamically populated from model on load
- **Real-time Filter Updates** - Bounding boxes and table update instantly

### 5. Match Alert System
- **Dropdown Class Selection** - Select specific class to monitor for matches
- **Live Match Detection** - Displays "Matched = ClassName (N detected)" when class is found
- **Status Updates** - Shows "Checking for matches..." when no detection
- **Dynamic Confidence Display** - Status bar includes current confidence threshold

### 6. Detection Statistics Table
- **Real-time Class Counts** - Shows count of each detected class per frame
- **Color-coded Rows** - Each row matches bounding box color for visual correlation
- **Filter-aware** - Only shows classes that pass confidence threshold
- **Dynamic Updates** - Updates every frame with latest detections

### 7. Frame Capture
- **Screenshot Button** - Captures current frame with all painted detections
- **File Dialog** - Save to PNG, JPG, or BMP
- **Status Feedback** - Displays save success/failure message

### 8. Real-time Performance Optimization
- **Frame Skipping Logic** - When inference speed exceeds target FPS, automatically skips frames
- **Multi-threaded Processing** - Video stream runs in separate QThread
- **Smooth UI** - Non-blocking inference and display updates
- **FPS Matching** - Maintains consistent playback speed regardless of inference speed

## Technical Stack
- **Frontend**: PySide6 (Qt for Python)
- **AI/ML**: Ultralytics YOLO
- **Computer Vision**: OpenCV
- **Video Processing**: Multi-threaded QThread
- **Drawing**: QPainter for custom graphics

## UI Components Layout
```
┌─────────────────────────────────────────────┐
│ [Load .pt Model] Model Name                 │
├─────────────────────────────────────────────┤
│     🟢 Matched = ClassName (3 detected)     │
├─────────────────────────────────────────────┤
│ Camera: [Dropdown] [Scan] [Start] [Stop] [Capture]
├──────────────────────────┬───────────────────┤
│                          │  Filters          │
│                          │  ──────           │
│                          │  Confidence: 0%   │
│    Video Feed (640x360)  │  [Slider]         │
│                          │                   │
│                          │  Classes to Show: │
│                          │  [x] Class1       │
│                          │  [x] Class2       │
│                          │  [x] Class3       │
│                          │  [x] Class4       │
│                          │                   │
│                          │  Match Alert:     │
│                          │  [Dropdown]       │
│                          ├───────────────────┤
│                          │ Class  | Count    │
│                          │ Class1 │    3     │
│                          │ Class2 │    1     │
└──────────────────────────┴───────────────────┘
│ Status: Ready                                │
└─────────────────────────────────────────────┘
```

## Data Flow
1. **Video Capture** → Frame grabbed from camera
2. **YOLO Inference** → Model processes frame
3. **Detection Extraction** → Boxes/classes/confidence extracted
4. **Filtering** → Applied based on user settings (confidence + class selection)
5. **Visualization** → Rectangles drawn with QPainter
6. **Analysis** → Class counts updated in table
7. **Alerting** → Match status updated based on detections
8. **Display** → Pixmap rendered to UI

## Performance Features
- **FPS Targeting**: Snaps camera FPS to nearest supported value (24/30/60)
- **Frame Skipping**: Drops frames when inference is slow to stay real-time
- **Threading**: Inference runs in background thread, UI stays responsive
- **Lazy Color Generation**: Colors assigned only to detected classes

## Current Color Scheme
- **Background**: White (match status bar)
- **Text**: Bright Green (#00ff00)
- **Bounding Boxes**: HSV-generated unique colors per class
- **Table**: Alternating rows with class-matched text colors

## Future Enhancement Opportunities
- Dark theme option
- Modern card-based layout
- Animated transitions
- Advanced metrics dashboard
- Recording capability
- Export detection logs
- Multi-model comparison
- Batch processing mode
