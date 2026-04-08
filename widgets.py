"""Custom Qt Widgets Module.

Reusable widgets used by the main UI.
"""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel


class ZoomableLabel(QLabel):
    """QLabel that supports mouse-wheel zooming."""

    zoom_changed = Signal(float)  # Emits new zoom level

    def __init__(self) -> None:
        super().__init__()
        self._zoom_level = 1.0
        self._original_pixmap: QPixmap | None = None

    # ── public API ──────────────────────────────────────────────

    def set_zoom_level(self, zoom: float) -> None:
        """Set zoom level (1.0 = 100%). Clamped to 0.1x – 5.0x."""
        self._zoom_level = max(0.1, min(zoom, 5.0))
        self.zoom_changed.emit(self._zoom_level)
        self._update_display()

    def get_zoom_level(self) -> float:
        return self._zoom_level

    # ── overrides ───────────────────────────────────────────────

    def setPixmap(self, pixmap: QPixmap) -> None:
        """Store the original pixmap and display it at current zoom."""
        self._original_pixmap = pixmap
        self._update_display()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out on mouse wheel."""
        if event.angleDelta().y() > 0:
            self.set_zoom_level(self._zoom_level * 1.1)
        else:
            self.set_zoom_level(self._zoom_level / 1.1)
        event.accept()

    # ── internals ───────────────────────────────────────────────

    def _update_display(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        zoomed_size = QSize(
            int(self._original_pixmap.width() * self._zoom_level),
            int(self._original_pixmap.height() * self._zoom_level),
        )
        zoomed_pixmap = self._original_pixmap.scaledToWidth(
            zoomed_size.width(),
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(zoomed_pixmap)
