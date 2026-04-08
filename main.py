"""Universal AI Inspection – Main UI & Application Entry Point.

This module contains the MainWindow (UI Display & Controls) and the
application entry point.  All functional logic is delegated to:

- camera.py          – Camera Input
- workers.py         – Frame Capture & Processing
- detection.py       – AI Object Detection
- relay_control.py   – Relay Control Logic
- Relay_B.py         – Hardware Output (Modbus relay driver)
- widgets.py         – Reusable custom Qt widgets
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, cast

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ── project modules ─────────────────────────────────────────────
from camera import find_cameras
from detection import cuda_available, load_model, get_model_classes, YOLO
from relay_control import (
    RELAY_AVAILABLE,
    RelayConnectionWorker,
    create_default_mappings,
    evaluate_mappings,
    disconnect_relay as _disconnect_relay_hw,
)
from widgets import ZoomableLabel
from workers import VideoWorker


# ── config path ─────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "app_config.json"


# ═══════════════════════════════════════════════════════════════
#  MainWindow – UI Display & Controls
# ═══════════════════════════════════════════════════════════════

class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Object Detection Inspection")

        # ── workers ─────────────────────────────────────────────
        self._worker = VideoWorker()
        self._worker.frame_ready.connect(self.on_frame)
        self._worker.status.connect(self.on_status)

        # ── detection state ─────────────────────────────────────
        self._class_counts: dict = {}
        self._class_colors: dict = {}
        self._current_pixmap: Optional[QPixmap] = None
        self._selected_classes: set = set()
        self._confidence_threshold: float = 0.0
        self._model_classes: List[str] = []
        self._match_class: Optional[str] = None
        self._match_detected: bool = False
        self._zoom_level: float = 1.0

        # ── relay state ─────────────────────────────────────────
        self._relay: Any = None
        self._relay_worker: Optional[RelayConnectionWorker] = None
        self._relay_connected: bool = False
        self._relay_host: str = "192.168.1.201"
        self._relay_port: int = 502
        self._relay_min_on_seconds: float = 1.0
        self._relay_mappings = create_default_mappings()
        self._relay_retry_count: int = 0
        self._relay_max_retries: int = 5
        self._relay_retry_delay_ms: int = 3000  # 3 seconds between retries
        self._relay_auto_connect: bool = False

        # ── metrics ─────────────────────────────────────────────
        self._fps_counter: int = 0
        self._fps_timer: float = 0
        self._current_fps: float = 0
        self._total_detections: int = 0
        self._inference_time: float = 0

        # ── overlay settings ────────────────────────────────────
        self._show_center_overlay: bool = True
        self._center_lines_data: List[dict] = []

        # ────────────────────────────────────────────────────────
        #  Build UI
        # ────────────────────────────────────────────────────────
        self._build_header()
        self._build_metrics()
        self._build_camera_controls()
        self._build_video_display()
        self._build_filter_panel()
        self._build_detection_table()
        self._build_relay_tab()
        self._assemble_layout()

        self.scan_cameras()
        self._auto_load_config()

    # ═══════════════════════════════════════════════════════════
    #  UI BUILDING HELPERS
    # ═══════════════════════════════════════════════════════════

    def _build_header(self) -> None:
        self._header_frame = QFrame()
        header_layout = QHBoxLayout()

        self.model_label = QLabel("No model selected")
        self.model_label.setStyleSheet("font-size: 12pt; font-weight: bold")
        self.load_model_button = QPushButton("📁 Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setMinimumHeight(40)

        self.compute_combo = QComboBox()
        self.compute_combo.addItem("cpu")
        self.compute_combo.addItem("cuda")
        self.compute_combo.addItem("cuda:0")
        self.compute_combo.setMaximumWidth(140)

        try:
            if cuda_available():
                idx = self.compute_combo.findText("cuda")
                if idx >= 0:
                    self.compute_combo.setCurrentIndex(idx)
                    self._worker.set_device("cuda")
            else:
                self._worker.set_device("cpu")
        except Exception:
            self._worker.set_device("cpu")

        self.save_config_button = QPushButton("💾 Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        self.save_config_button.setMinimumHeight(40)
        self.save_config_button.setMaximumWidth(140)

        self.load_config_button = QPushButton("📂 Load Config")
        self.load_config_button.clicked.connect(self.load_config)
        self.load_config_button.setMinimumHeight(40)
        self.load_config_button.setMaximumWidth(140)

        header_layout.addWidget(QLabel("Model:"), 0)
        header_layout.addWidget(self.model_label, 1)
        header_layout.addWidget(self.load_model_button, 0)
        header_layout.addWidget(QLabel("Compute:"), 0)
        header_layout.addWidget(self.compute_combo, 0)
        header_layout.addWidget(self.save_config_button, 0)
        header_layout.addWidget(self.load_config_button, 0)
        self._header_frame.setLayout(header_layout)

    def _build_metrics(self) -> None:
        self._metrics_frame = QFrame()
        metrics_layout = QHBoxLayout()

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 11pt; font-weight: bold")

        self.detections_label = QLabel("Detections: 0")
        self.detections_label.setStyleSheet("font-size: 11pt; font-weight: bold")

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 10pt")

        self.compute_combo.currentTextChanged.connect(self.on_compute_changed)

        metrics_layout.addWidget(self.fps_label, 0)
        metrics_layout.addWidget(self.detections_label, 0)
        metrics_layout.addWidget(self.status_label, 1)
        self._metrics_frame.setLayout(metrics_layout)

    def _build_camera_controls(self) -> None:
        self._cam_frame = QFrame()
        cam_layout = QHBoxLayout()

        self.camera_combo = QComboBox()
        self.camera_combo.setMaximumWidth(240)

        self.scan_button = QPushButton("🔄 Scan")
        self.scan_button.clicked.connect(self.scan_cameras)
        self.scan_button.setMaximumWidth(120)

        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.start_stream)
        self.start_button.setMaximumWidth(120)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setMaximumWidth(120)

        self.capture_button = QPushButton("📷 Capture")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setMaximumWidth(120)

        cam_layout.addWidget(QLabel("Camera:"), 0)
        cam_layout.addWidget(self.camera_combo, 1)
        cam_layout.addWidget(self.scan_button, 0)
        cam_layout.addWidget(self.start_button, 0)
        cam_layout.addWidget(self.stop_button, 0)
        cam_layout.addWidget(self.capture_button, 0)
        self._cam_frame.setLayout(cam_layout)

    def _build_video_display(self) -> None:
        self._video_frame = QFrame()
        video_layout = QVBoxLayout()

        zoom_control_layout = QHBoxLayout()
        self.zoom_out_button = QPushButton("🔍➖")
        self.zoom_out_button.setMaximumWidth(50)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

        self.zoom_in_button = QPushButton("🔍➕")
        self.zoom_in_button.setMaximumWidth(50)
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.reset_zoom_button = QPushButton("Reset")
        self.reset_zoom_button.setMaximumWidth(70)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        zoom_control_layout.addWidget(self.zoom_out_button, 0)
        zoom_control_layout.addWidget(self.zoom_label, 1)
        zoom_control_layout.addWidget(self.zoom_in_button, 0)
        zoom_control_layout.addWidget(self.reset_zoom_button, 0)

        self.video_label = ZoomableLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_label.zoom_changed.connect(self.on_zoom_changed)

        video_layout.addLayout(zoom_control_layout, 0)
        video_layout.addWidget(self.video_label, 1)
        self._video_frame.setLayout(video_layout)

    def _build_filter_panel(self) -> None:
        self._filter_frame = QFrame()
        filter_layout = QVBoxLayout()

        filter_title = QLabel("🎛️ Filters & Settings")
        filter_title.setStyleSheet("font-size: 12pt; font-weight: bold")
        filter_layout.addWidget(filter_title)

        conf_label = QLabel("Confidence Threshold")
        conf_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        filter_layout.addWidget(conf_label)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        filter_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0%")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filter_layout.addWidget(self.confidence_label)

        filter_layout.addSpacing(15)

        classes_label = QLabel("Classes to Show")
        classes_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        filter_layout.addWidget(classes_label)

        self.class_filters_scroll = QScrollArea()
        self.class_filters_scroll.setWidgetResizable(True)
        self.class_filters_container = QWidget()
        self.class_filters_layout = QVBoxLayout()
        self.class_filters_container.setLayout(self.class_filters_layout)
        self.class_filters_scroll.setWidget(self.class_filters_container)
        self.class_filters_scroll.setMinimumHeight(180)
        filter_layout.addWidget(self.class_filters_scroll, 1)

        filter_layout.addSpacing(15)

        self.center_overlay_checkbox = QCheckBox("🎯 Show Center Lines")
        self.center_overlay_checkbox.setChecked(True)
        self.center_overlay_checkbox.toggled.connect(self.on_overlay_toggled)
        filter_layout.addWidget(self.center_overlay_checkbox)

        self._filter_frame.setLayout(filter_layout)
        self._filter_frame.setMaximumWidth(240)

    def _build_detection_table(self) -> None:
        self._table_frame = QFrame()
        table_layout = QVBoxLayout()

        table_title = QLabel("📊 Detections")
        table_title.setStyleSheet("font-size: 10pt; font-weight: bold; color: #00d9ff;")
        table_layout.addWidget(table_title)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)
        self._table_frame.setLayout(table_layout)
        self._table_frame.setMaximumWidth(240)

    def _build_relay_tab(self) -> None:
        self.match_status_label = QLabel("")
        self.match_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_status_label.setStyleSheet(
            "background-color: #00d9ff; color: #000000; font-size: 14pt; font-weight: bold; "
            "padding: 12px; border-radius: 6px;"
        )
        self.match_status_label.setMinimumHeight(50)

        self._relay_tab = QWidget()
        relay_tab_layout = QVBoxLayout()

        # Connection section
        relay_conn_frame = QFrame()
        relay_conn_layout = QHBoxLayout()

        relay_conn_layout.addWidget(QLabel("Host:"), 0)
        self.relay_host_input = QLineEdit()
        self.relay_host_input.setText(self._relay_host)
        self.relay_host_input.setPlaceholderText("192.168.1.201")
        self.relay_host_input.setMaximumWidth(180)
        relay_conn_layout.addWidget(self.relay_host_input, 0)

        relay_conn_layout.addWidget(QLabel("Port:"), 0)
        self.relay_port_spin = QSpinBox()
        self.relay_port_spin.setRange(1, 65535)
        self.relay_port_spin.setValue(self._relay_port)
        self.relay_port_spin.setMaximumWidth(100)
        relay_conn_layout.addWidget(self.relay_port_spin, 0)

        self.relay_connect_button = QPushButton("🔌 Connect Relay")
        self.relay_connect_button.clicked.connect(self.connect_relay)
        self.relay_connect_button.setMinimumHeight(36)
        relay_conn_layout.addWidget(self.relay_connect_button, 0)

        self.relay_status_label = QLabel("Status: Disconnected")
        self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
        relay_conn_layout.addWidget(self.relay_status_label, 1)

        relay_conn_frame.setLayout(relay_conn_layout)
        relay_tab_layout.addWidget(relay_conn_frame, 0)

        # Mapping table
        mapping_label = QLabel("🎯 Class → Relay Channel Mapping")
        mapping_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        relay_tab_layout.addWidget(mapping_label)

        self.relay_mapping_table = QTableWidget(8, 5)
        self.relay_mapping_table.setHorizontalHeaderLabels(["Class", "Relay Channel", "Distance Min (px)", "Distance Max (px)", "Relay Status"])
        self.relay_mapping_table.verticalHeader().setVisible(False)
        self.relay_mapping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.relay_mapping_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.relay_mapping_table.setAlternatingRowColors(True)
        header = self.relay_mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        for row in range(8):
            class_combo = QComboBox()
            class_combo.addItem("None", None)
            class_combo.currentIndexChanged.connect(
                lambda idx, r=row: self._on_relay_mapping_class_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 0, class_combo)

            ch_combo = QComboBox()
            for ch in range(1, 9):
                ch_combo.addItem(f"CH {ch}", ch)
            ch_combo.setCurrentIndex(row)
            ch_combo.currentIndexChanged.connect(
                lambda idx, r=row: self._on_relay_mapping_channel_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 1, ch_combo)

            dist_min_spin = QSpinBox()
            dist_min_spin.setRange(0, 10000)
            dist_min_spin.setValue(0)
            dist_min_spin.setSuffix(" px")
            dist_min_spin.valueChanged.connect(
                lambda val, r=row: self._on_relay_mapping_distance_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 2, dist_min_spin)

            dist_max_spin = QSpinBox()
            dist_max_spin.setRange(0, 10000)
            dist_max_spin.setValue(10000)
            dist_max_spin.setSuffix(" px")
            dist_max_spin.valueChanged.connect(
                lambda val, r=row: self._on_relay_mapping_distance_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 3, dist_max_spin)

            status_item = QTableWidgetItem("OFF")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setForeground(QBrush(QColor("#ff6b6b")))
            self.relay_mapping_table.setItem(row, 4, status_item)

        relay_tab_layout.addWidget(self.relay_mapping_table, 1)
        relay_tab_layout.addWidget(self.match_status_label, 0)
        self._relay_tab.setLayout(relay_tab_layout)

    def _assemble_layout(self) -> None:
        self.tab_widget = QTabWidget()

        monitor_tab = QWidget()
        monitor_layout = QHBoxLayout()
        monitor_layout.addWidget(self._video_frame, 2)
        monitor_layout.addWidget(self._filter_frame, 0)
        monitor_layout.addWidget(self._table_frame, 0)
        monitor_tab.setLayout(monitor_layout)

        self.tab_widget.addTab(monitor_tab, "📹 Monitor")
        self.tab_widget.addTab(self._relay_tab, "⚡ Relay")

        layout = QVBoxLayout()
        layout.addWidget(self._header_frame, 0)
        layout.addWidget(self._metrics_frame, 0)
        layout.addWidget(self._cam_frame, 0)
        layout.addWidget(self.tab_widget, 1)

        self.setLayout(layout)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

    # ═══════════════════════════════════════════════════════════
    #  CAMERA SLOTS
    # ═══════════════════════════════════════════════════════════

    @Slot()
    def scan_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = find_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera", None)
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

    @Slot()
    def start_stream(self) -> None:
        if self._worker.isRunning():
            self.status_label.setText("Already running.")
            return
        index = self.camera_combo.currentData()
        if index is None:
            self.status_label.setText("Select a valid camera.")
            return
        self._worker.set_camera_index(int(index))
        self._worker.set_device(self.compute_combo.currentText())
        self._worker.start()

    @Slot()
    def stop_stream(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

    @Slot()
    def capture_frame(self) -> None:
        if self._current_pixmap is None:
            self.status_label.setText("No frame to capture.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Captured Frame", "", "Images (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return
        if self._current_pixmap.save(file_path):
            self.status_label.setText(f"Frame saved: {Path(file_path).name}")
        else:
            self.status_label.setText("Frame save failed.")

    # ═══════════════════════════════════════════════════════════
    #  AI MODEL SLOTS
    # ═══════════════════════════════════════════════════════════

    @Slot()
    def load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO .pt Model", "", "PyTorch Model (*.pt)"
        )
        if not file_path:
            return
        model_path = Path(file_path)
        self.model_label.setText(model_path.name)
        self._worker.set_model_path(model_path)
        self.status_label.setText("Loading model...")

        if YOLO is not None:
            try:
                device = self.compute_combo.currentText()
                model = load_model(model_path, device)
                self._model_classes = get_model_classes(model)
                self._populate_class_filters()
                self.status_label.setText(f"Model loaded: {len(self._model_classes)} classes")
            except Exception as exc:
                self.status_label.setText(f"Failed to load model: {exc}")
                self._model_classes = []
        else:
            self.status_label.setText("Ultralytics not available.")
            self._model_classes = []

    def _populate_class_filters(self) -> None:
        while self.class_filters_layout.count() > 0:
            item = self.class_filters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._selected_classes.clear()
        for class_name in sorted(self._model_classes):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, cn=class_name: self._toggle_class_filter(cn, checked)
            )
            self.class_filters_layout.addWidget(checkbox)
            self._selected_classes.add(class_name)

        self._update_relay_mapping_classes()

    # ═══════════════════════════════════════════════════════════
    #  FILTER / CONFIDENCE SLOTS
    # ═══════════════════════════════════════════════════════════

    @Slot(int)
    def on_confidence_changed(self, value: int) -> None:
        self._confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence: {value}%")
        self._update_match_status_label()

    def _toggle_class_filter(self, class_name: str, checked: bool) -> None:
        if checked:
            self._selected_classes.add(class_name)
        else:
            self._selected_classes.discard(class_name)

    @Slot(bool)
    def on_overlay_toggled(self, checked: bool) -> None:
        self._show_center_overlay = checked

    # ═══════════════════════════════════════════════════════════
    #  RELAY CONTROL SLOTS
    # ═══════════════════════════════════════════════════════════

    def _update_relay_mapping_classes(self) -> None:
        sorted_classes = sorted(self._model_classes)
        for row in range(8):
            combo_widget = self.relay_mapping_table.cellWidget(row, 0)
            if not isinstance(combo_widget, QComboBox):
                continue
            combo = cast(QComboBox, combo_widget)
            combo.blockSignals(True)
            current_data = combo.currentData()
            combo.clear()
            combo.addItem("None", None)
            for class_name in sorted_classes:
                combo.addItem(class_name, class_name)
            if current_data:
                idx = combo.findData(current_data)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            self._relay_mappings[row]["class"] = combo.currentData()
            
            # Initialize distance parameters if not present
            if "distance_min" not in self._relay_mappings[row]:
                self._relay_mappings[row]["distance_min"] = 0
            if "distance_max" not in self._relay_mappings[row]:
                self._relay_mappings[row]["distance_max"] = 10000

    def _on_relay_mapping_class_changed(self, row: int) -> None:
        combo_widget = self.relay_mapping_table.cellWidget(row, 0)
        if isinstance(combo_widget, QComboBox):
            combo = cast(QComboBox, combo_widget)
            self._relay_mappings[row]["class"] = combo.currentData()

    def _on_relay_mapping_channel_changed(self, row: int) -> None:
        combo_widget = self.relay_mapping_table.cellWidget(row, 1)
        if isinstance(combo_widget, QComboBox):
            combo = cast(QComboBox, combo_widget)
            self._relay_mappings[row]["channel"] = combo.currentData()

    def _on_relay_mapping_distance_changed(self, row: int) -> None:
        dist_min_widget = self.relay_mapping_table.cellWidget(row, 2)
        dist_max_widget = self.relay_mapping_table.cellWidget(row, 3)
        if isinstance(dist_min_widget, QSpinBox) and isinstance(dist_max_widget, QSpinBox):
            self._relay_mappings[row]["distance_min"] = dist_min_widget.value()
            self._relay_mappings[row]["distance_max"] = dist_max_widget.value()

    def _update_match_status_label(self) -> None:
        matched = self._evaluate_mappings_with_distance()

        for row, mapping in enumerate(self._relay_mappings):
            self._set_relay_status_cell(row, mapping["last_state"])

        if matched:
            self.match_status_label.setText(f"Matched: {', '.join(matched)}")
        else:
            active = [m["class"] for m in self._relay_mappings if m["class"] is not None]
            if active:
                self.match_status_label.setText("Checking for matches...")
            else:
                self.match_status_label.setText("No class-to-relay mappings configured")

    def _evaluate_mappings_with_distance(self) -> List[str]:
        """Evaluate relay mappings considering both class and distance."""
        matched = []
        now = time.time()

        for row, mapping in enumerate(self._relay_mappings):
            channel = mapping["channel"]

            if mapping["class"] is None:
                # Turn off relay if it was previously on
                if mapping["last_state"]:
                    self._relay_set_channel(channel, False)
                    mapping["last_state"] = False
                continue

            class_name = mapping["class"]
            distance_min = mapping.get("distance_min", 0)
            distance_max = mapping.get("distance_max", 10000)

            # Check if class is detected
            detected = class_name in self._class_counts

            # Check if any detection of this class is within distance range
            matched_distance = False
            if detected:
                for det_data in self._center_lines_data:
                    if det_data["class_name"] == class_name:
                        distance = det_data["distance_px"]
                        if distance_min <= distance <= distance_max:
                            matched_distance = True
                            break

            if matched_distance and not mapping["last_state"]:
                # Turn ON: detection matched and relay was off
                self._relay_set_channel(channel, True)
                mapping["last_state"] = True
                mapping["last_on_time"] = now
                matched.append(class_name)
            elif matched_distance and mapping["last_state"]:
                # Still matched, keep on
                matched.append(class_name)
            elif not matched_distance and mapping["last_state"]:
                # No longer matched — turn off after min-on time
                elapsed_on = now - mapping.get("last_on_time", 0)
                if elapsed_on >= self._relay_min_on_seconds:
                    self._relay_set_channel(channel, False)
                    mapping["last_state"] = False
                else:
                    # Keep it on until min-on time elapses
                    matched.append(class_name)

        return matched

    def _relay_set_channel(self, channel: int, on: bool) -> None:
        """Send relay on/off command to hardware."""
        if not self._relay_connected or self._relay is None:
            return
        try:
            if on:
                self._relay.on(channel)
            else:
                self._relay.off(channel)
        except Exception as e:
            self.status_label.setText(f"Relay CH{channel} error: {e}")

    def _set_relay_status_cell(self, row: int, is_on: bool) -> None:
        item = self.relay_mapping_table.item(row, 4)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.relay_mapping_table.setItem(row, 4, item)
        if is_on:
            item.setText("ON")
            item.setForeground(QBrush(QColor("#51cf66")))
        else:
            item.setText("OFF")
            item.setForeground(QBrush(QColor("#ff6b6b")))

    @Slot()
    def connect_relay(self) -> None:
        if not RELAY_AVAILABLE:
            self.relay_status_label.setText("Status: Relay library not available")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
            return

        if self._relay_connected:
            self.disconnect_relay()
            return

        self._relay_host = self.relay_host_input.text().strip()
        self._relay_port = self.relay_port_spin.value()
        self._relay_retry_count = 0

        self.relay_status_label.setText("Status: Connecting...")
        self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ffd43b; font-weight: bold;")
        self.relay_connect_button.setText("⏳ Connecting...")
        self.relay_connect_button.setEnabled(False)

        self._relay_worker = RelayConnectionWorker(self._relay_host, self._relay_port)
        self._relay_worker.connection_result.connect(self.on_relay_connection_result)
        self._relay_worker.start()

    @Slot(bool, str)
    def on_relay_connection_result(self, success: bool, message: str) -> None:
        self.relay_connect_button.setEnabled(True)

        if success and self._relay_worker is not None:
            self._relay = self._relay_worker.relay
            self._relay_connected = True
            self._relay_retry_count = 0
            self.relay_status_label.setText(f"Status: Connected ({self._relay_host}:{self._relay_port})")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #51cf66; font-weight: bold;")
            self.relay_connect_button.setText("🔌 Disconnect Relay")
            self.relay_connect_button.clicked.disconnect()
            self.relay_connect_button.clicked.connect(self.disconnect_relay)
            self.status_label.setText(f"Relay connected to {self._relay_host}:{self._relay_port}")
        else:
            self._relay_connected = False
            self._relay = None

            # Retry logic
            self._relay_retry_count += 1
            if self._relay_retry_count < self._relay_max_retries:
                remaining = self._relay_max_retries - self._relay_retry_count
                self.relay_status_label.setText(
                    f"Status: Retry {self._relay_retry_count}/{self._relay_max_retries} in {self._relay_retry_delay_ms // 1000}s..."
                )
                self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ffd43b; font-weight: bold;")
                self.relay_connect_button.setText(f"⏳ Retrying ({remaining} left)")
                self.relay_connect_button.setEnabled(False)
                self.status_label.setText(f"Relay connect failed: {message} — retrying...")
                if self._relay_worker is not None:
                    self._relay_worker.deleteLater()
                self._relay_worker = None
                QTimer.singleShot(self._relay_retry_delay_ms, self._retry_connect_relay)
                return
            else:
                self._relay_retry_count = 0
                self.relay_status_label.setText("Status: Connection failed (retries exhausted)")
                self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
                self.relay_connect_button.setText("🔌 Connect Relay")
                self.status_label.setText(f"Relay connection error: {message}")

        if self._relay_worker is not None:
            self._relay_worker.deleteLater()
        self._relay_worker = None

    def _retry_connect_relay(self) -> None:
        """Retry relay connection (called by QTimer after failure)."""
        if self._relay_connected:
            return
        self.relay_status_label.setText(
            f"Status: Connecting (attempt {self._relay_retry_count + 1}/{self._relay_max_retries})..."
        )
        self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ffd43b; font-weight: bold;")
        self.relay_connect_button.setText("⏳ Connecting...")
        self.relay_connect_button.setEnabled(False)

        self._relay_worker = RelayConnectionWorker(self._relay_host, self._relay_port)
        self._relay_worker.connection_result.connect(self.on_relay_connection_result)
        self._relay_worker.start()

    @Slot()
    def disconnect_relay(self) -> None:
        self._relay_retry_count = self._relay_max_retries  # stop any pending retries
        try:
            if self._relay is not None and self._relay_connected:
                _disconnect_relay_hw(self._relay, self._relay_mappings)
            self._relay_connected = False
            self._relay = None

            for row in range(8):
                self._set_relay_status_cell(row, False)

            self.relay_status_label.setText("Status: Disconnected")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
            self.relay_connect_button.setText("🔌 Connect Relay")
            self.relay_connect_button.clicked.disconnect()
            self.relay_connect_button.clicked.connect(self.connect_relay)
            self.status_label.setText("Relay disconnected")
        except Exception as e:
            self.status_label.setText(f"Relay disconnection error: {e}")

    # ═══════════════════════════════════════════════════════════
    #  COMPUTE / ZOOM SLOTS
    # ═══════════════════════════════════════════════════════════

    @Slot(str)
    def on_compute_changed(self, device: str) -> None:
        self.status_label.setText(f"Compute: {device}")
        self._worker.set_device(device)

    @Slot(float)
    def on_zoom_changed(self, zoom_level: float) -> None:
        self._zoom_level = zoom_level
        self.zoom_label.setText(f"Zoom: {zoom_level * 100:.0f}%")

    @Slot()
    def zoom_in(self) -> None:
        self.video_label.set_zoom_level(self._zoom_level * 1.1)

    @Slot()
    def zoom_out(self) -> None:
        self.video_label.set_zoom_level(self._zoom_level / 1.1)

    @Slot()
    def reset_zoom(self) -> None:
        self.video_label.set_zoom_level(1.0)

    # ═══════════════════════════════════════════════════════════
    #  FRAME PROCESSING SLOT
    # ═══════════════════════════════════════════════════════════

    @Slot(QImage, list)
    def on_frame(self, image: QImage, detections: list) -> None:
        # FPS tracking
        current_time = time.time()
        if self._fps_timer == 0:
            self._fps_timer = current_time

        self._fps_counter += 1
        elapsed = current_time - self._fps_timer
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self.fps_label.setText(f"FPS: {self._current_fps:.1f}")
            self._fps_counter = 0
            self._fps_timer = current_time

        pixmap = QPixmap.fromImage(image)

        # Filter detections
        filtered_detections = []
        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = float(det["label"].split()[-1]) if " " in det["label"] else 0.0
            if class_name in self._selected_classes and conf >= self._confidence_threshold:
                filtered_detections.append(det)
        detections = filtered_detections

        # Draw bounding boxes
        if detections:
            for det in detections:
                class_name = det.get("class_name", "unknown")
                if class_name not in self._class_colors:
                    hue = (len(self._class_colors) * 37) % 360
                    self._class_colors[class_name] = QColor.fromHsv(hue, 200, 255)
                det["color"] = self._class_colors[class_name]

            painter = QPainter(pixmap)
            pen = QPen()
            pen.setWidth(2)
            for det in detections:
                color = det.get("color", Qt.GlobalColor.green)
                pen.setColor(color)
                painter.setPen(pen)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1, max(0, y1 - 6), det["label"])
            painter.end()

        self._total_detections = len(detections)
        self.detections_label.setText(f"Detections: {self._total_detections}")

        # Draw center lines overlay
        if self._show_center_overlay and detections and pixmap.width() > 0:
            self._center_lines_data = self._draw_center_lines(pixmap, detections)

        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)
        self._update_class_table(detections)

    def _update_class_table(self, detections: list) -> None:
        self._class_counts.clear()
        for det in detections:
            class_name = det.get("class_name", "unknown")
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + 1

        self.table.setRowCount(len(self._class_counts))
        for row, (class_name, count) in enumerate(sorted(self._class_counts.items())):
            color = self._class_colors.get(class_name, Qt.GlobalColor.white)
            class_item = QTableWidgetItem(class_name)
            count_item = QTableWidgetItem(str(count))
            brush = QBrush(color)
            class_item.setForeground(brush)
            count_item.setForeground(brush)
            self.table.setItem(row, 0, class_item)
            self.table.setItem(row, 1, count_item)

        self._update_match_status_label()

    def _draw_center_lines(self, pixmap: QPixmap, detections: list) -> List[dict]:
        """Draw vertical center lines for detections and frame center."""
        frame_width = pixmap.width()
        frame_height = pixmap.height()
        frame_center_x = frame_width / 2

        # Prepare data for each detection
        center_lines_data = []
        for det in detections:
            x1, x2 = det["x1"], det["x2"]
            det_center_x = (x1 + x2) / 2
            distance_px = abs(det_center_x - frame_center_x)
            center_lines_data.append({
                "det_center_x": det_center_x,
                "distance_px": distance_px,
                "class_name": det.get("class_name", "unknown"),
                "color": det.get("color", Qt.GlobalColor.cyan)
            })

        # Draw on pixmap
        painter = QPainter(pixmap)

        # Draw frame center line (yellow)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(frame_center_x), 0, int(frame_center_x), frame_height)

        # Draw detection center lines (cyan) with distance labels
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for data in center_lines_data:
            det_center_x = data["det_center_x"]
            color = data["color"]
            distance = data["distance_px"]
            class_name = data["class_name"]

            # Draw vertical line at detection center
            painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
            painter.drawLine(int(det_center_x), 0, int(det_center_x), frame_height)

            # Draw distance label
            label_text = f"{distance:.0f}px"
            label_y = 20
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.fillRect(int(det_center_x) - 30, label_y - 12, 60, 16, QColor(0, 0, 0, 200))
            painter.drawText(int(det_center_x) - 25, label_y, label_text)

        # Draw legend at top-left
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        legend_y = 20
        painter.drawText(10, legend_y, "🟡 Frame Center")
        painter.drawText(10, legend_y + 15, "🔵 Detection Center")

        painter.end()

        return center_lines_data

    @Slot(str)
    def on_status(self, message: str) -> None:
        self.status_label.setText(message)

    # ═══════════════════════════════════════════════════════════
    #  CONFIG SAVE / LOAD
    # ═══════════════════════════════════════════════════════════

    def _build_config_dict(self) -> dict:
        """Collect current app settings into a serialisable dict."""
        # Selected camera index
        cam_index = self.camera_combo.currentData()

        # Relay mappings
        relay_mappings = []
        for m in self._relay_mappings:
            relay_mappings.append({
                "class": m["class"],
                "channel": m["channel"],
                "distance_min": m.get("distance_min", 0),
                "distance_max": m.get("distance_max", 10000),
            })

        return {
            "model_path": self._worker._model_path_str if hasattr(self._worker, '_model_path_str') else
                          str(self._worker._model_path) if hasattr(self._worker, '_model_path') and self._worker._model_path else "",
            "camera_index": cam_index,
            "confidence_threshold": self.confidence_slider.value(),
            "selected_classes": sorted(self._selected_classes),
            "relay_host": self.relay_host_input.text().strip(),
            "relay_port": self.relay_port_spin.value(),
            "relay_mappings": relay_mappings,
            "compute_device": self.compute_combo.currentText(),
            "show_center_overlay": self.center_overlay_checkbox.isChecked(),
            "auto_connect_relay": self._relay_auto_connect or self._relay_connected,
        }

    @Slot()
    def save_config(self) -> None:
        """Save current settings to JSON config file."""
        try:
            cfg = self._build_config_dict()
            # Resolve model path from the label text
            model_text = self.model_label.text()
            if model_text and model_text != "No model selected":
                # Try to keep the full path if available from the worker
                worker_path = getattr(self._worker, '_model_path', None)
                if worker_path:
                    cfg["model_path"] = str(worker_path)
                else:
                    cfg["model_path"] = model_text
            _CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            self.status_label.setText(f"Config saved → {_CONFIG_PATH.name}")
        except Exception as e:
            self.status_label.setText(f"Config save error: {e}")

    @Slot()
    def load_config(self) -> None:
        """Load settings from JSON config file via file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Config File", str(_CONFIG_PATH.parent), "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            cfg = json.loads(Path(file_path).read_text(encoding="utf-8"))
            self._apply_config(cfg)
            self.status_label.setText(f"Config loaded ← {Path(file_path).name}")
        except Exception as e:
            self.status_label.setText(f"Config load error: {e}")

    def _auto_load_config(self) -> None:
        """Automatically load config from default path on startup."""
        if not _CONFIG_PATH.exists():
            return
        try:
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            self._apply_config(cfg)
            self.status_label.setText(f"Config auto-loaded ← {_CONFIG_PATH.name}")
        except Exception as e:
            self.status_label.setText(f"Auto-load config error: {e}")

    def _apply_config(self, cfg: dict) -> None:
        """Apply a config dict to the UI and internal state."""
        # ── Model ───────────────────────────────────────────────
        model_path_str = cfg.get("model_path", "")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                self.model_label.setText(model_path.name)
                self._worker.set_model_path(model_path)
                if YOLO is not None:
                    try:
                        device = cfg.get("compute_device", self.compute_combo.currentText())
                        model = load_model(model_path, device)
                        self._model_classes = get_model_classes(model)
                        self._populate_class_filters()
                    except Exception:
                        pass

        # ── Compute device ──────────────────────────────────────
        compute = cfg.get("compute_device", "")
        if compute:
            idx = self.compute_combo.findText(compute)
            if idx >= 0:
                self.compute_combo.setCurrentIndex(idx)

        # ── Camera ──────────────────────────────────────────────
        cam_index = cfg.get("camera_index")
        if cam_index is not None:
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == cam_index:
                    self.camera_combo.setCurrentIndex(i)
                    break

        # ── Confidence threshold ────────────────────────────────
        conf = cfg.get("confidence_threshold")
        if conf is not None:
            self.confidence_slider.setValue(int(conf))

        # ── Selected classes ────────────────────────────────────
        saved_classes = cfg.get("selected_classes")
        if saved_classes is not None and self._model_classes:
            saved_set = set(saved_classes)
            for i in range(self.class_filters_layout.count()):
                widget = self.class_filters_layout.itemAt(i).widget()
                if isinstance(widget, QCheckBox):
                    widget.setChecked(widget.text() in saved_set)

        # ── Center overlay ──────────────────────────────────────
        show_overlay = cfg.get("show_center_overlay")
        if show_overlay is not None:
            self.center_overlay_checkbox.setChecked(bool(show_overlay))

        # ── Relay host / port ───────────────────────────────────
        relay_host = cfg.get("relay_host")
        if relay_host:
            self.relay_host_input.setText(relay_host)
        relay_port = cfg.get("relay_port")
        if relay_port:
            self.relay_port_spin.setValue(int(relay_port))

        # ── Relay mappings ──────────────────────────────────────
        saved_mappings = cfg.get("relay_mappings")
        if saved_mappings:
            for row, sm in enumerate(saved_mappings[:8]):
                # Class combo
                combo_widget = self.relay_mapping_table.cellWidget(row, 0)
                if isinstance(combo_widget, QComboBox):
                    combo = cast(QComboBox, combo_widget)
                    cls = sm.get("class")
                    if cls:
                        idx = combo.findData(cls)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    else:
                        combo.setCurrentIndex(0)  # None

                # Channel combo
                ch_widget = self.relay_mapping_table.cellWidget(row, 1)
                if isinstance(ch_widget, QComboBox):
                    ch_combo = cast(QComboBox, ch_widget)
                    ch = sm.get("channel", row + 1)
                    ch_idx = ch_combo.findData(ch)
                    if ch_idx >= 0:
                        ch_combo.setCurrentIndex(ch_idx)

                # Distance min
                dist_min_w = self.relay_mapping_table.cellWidget(row, 2)
                if isinstance(dist_min_w, QSpinBox):
                    dist_min_w.setValue(sm.get("distance_min", 0))

                # Distance max
                dist_max_w = self.relay_mapping_table.cellWidget(row, 3)
                if isinstance(dist_max_w, QSpinBox):
                    dist_max_w.setValue(sm.get("distance_max", 10000))

                # Update internal mapping
                self._relay_mappings[row]["class"] = sm.get("class")
                self._relay_mappings[row]["channel"] = sm.get("channel", row + 1)
                self._relay_mappings[row]["distance_min"] = sm.get("distance_min", 0)
                self._relay_mappings[row]["distance_max"] = sm.get("distance_max", 10000)

        # ── Auto-connect relay ──────────────────────────────────
        auto_relay = cfg.get("auto_connect_relay", False)
        if auto_relay and not self._relay_connected:
            self._relay_retry_count = 0
            self.connect_relay()

        # ── Auto-start stream if model and camera are ready ─────
        if model_path_str and self._model_classes and self.camera_combo.currentData() is not None:
            self.start_stream()

    # ═══════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        self.stop_stream()
        self.disconnect_relay()
        if self._relay_worker is not None:
            self._relay_worker.quit()
            self._relay_worker.wait(1000)
            self._relay_worker = None
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
#  Application Entry Point
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
