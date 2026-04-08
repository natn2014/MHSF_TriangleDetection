"""Frame Capture & Processing Worker Module.

Background QThread that captures camera frames, runs AI detection,
and emits processed frames back to the UI.
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from camera import open_camera, nearest_allowed_fps
from detection import cuda_available, extract_detections, load_model


class VideoWorker(QThread):
    """Captures frames from a camera, runs YOLO inference, emits results."""

    frame_ready = Signal(QImage, list)
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._camera_index: Optional[int] = None
        self._model_path: Optional[Path] = None
        self._model = None
        self._target_fps: int = 30
        self._device: str = "cuda" if cuda_available() else "cpu"

    # ── configuration setters ───────────────────────────────────

    def set_camera_index(self, index: Optional[int]) -> None:
        self._camera_index = index

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self._model = None

    def set_device(self, device: str) -> None:
        self._device = device

    def stop(self) -> None:
        self._running = False

    # ── model helpers ───────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model_path is None:
            return
        try:
            self._model = load_model(self._model_path, self._device)
            self.status.emit(f"Model loaded: {self._model_path.name}")
        except RuntimeError as exc:
            self.status.emit(str(exc))
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            self._model = None

    # ── main loop ───────────────────────────────────────────────

    def run(self) -> None:
        if self._camera_index is None:
            self.status.emit("Select a camera.")
            return

        cap = open_camera(self._camera_index)

        if not cap.isOpened():
            self.status.emit("Camera open failed.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self._target_fps = nearest_allowed_fps(fps)
        self.status.emit(f"Camera FPS: {fps:.2f} -> Target FPS: {self._target_fps}")

        if self._model_path is not None and self._model is None:
            self._load_model()

        self._running = True
        frame_period = 1.0 / self._target_fps

        while self._running:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                self.status.emit("Frame grab failed.")
                break

            detections: List[dict] = []
            if self._model is not None:
                try:
                    results = self._model(frame, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    results = None
                detections = extract_detections(results)

            elapsed = time.perf_counter() - start_time

            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            else:
                frames_to_skip = int(elapsed / frame_period) - 1
                for _ in range(max(0, frames_to_skip)):
                    cap.grab()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(image, detections)

        cap.release()
        self.status.emit("Stopped.")
