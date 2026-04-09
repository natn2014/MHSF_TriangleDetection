"""Custom Qt Widgets Module.

Reusable widgets used by the main UI.
"""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QKeyEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class _FullscreenWindow(QWidget):
    """Borderless fullscreen window that hosts the video label temporarily."""

    closed = Signal()  # emitted when ESC is pressed

    def __init__(self, label: "ZoomableLabel") -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #000000;")
        self._label = label
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)

        # ── ESC overlay button (top-right) ──────────────────────
        self._esc_btn = QPushButton("ESC", self)
        self._esc_btn.setFixedSize(60, 36)
        self._esc_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(40, 40, 40, 180);"
            "  color: #ffffff;"
            "  border: 1px solid rgba(255, 255, 255, 100);"
            "  border-radius: 6px;"
            "  font-size: 14px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(200, 50, 50, 200);"
            "}"
        )
        self._esc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._esc_btn.clicked.connect(self.closed.emit)
        # Raise above video; position is set in resizeEvent
        self._esc_btn.raise_()

        self.showFullScreen()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        margin = 12
        self._esc_btn.move(self.width() - self._esc_btn.width() - margin, margin)
        self._esc_btn.raise_()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.closed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.closed.emit()
        event.accept()


class ZoomableLabel(QLabel):
    """QLabel that supports mouse-wheel zooming and double-click fullscreen."""

    zoom_changed = Signal(float)  # Emits new zoom level

    def __init__(self) -> None:
        super().__init__()
        self._zoom_level = 1.0
        self._original_pixmap: QPixmap | None = None
        self._is_fullscreen = False
        self._fullscreen_window: _FullscreenWindow | None = None
        self._parent_layout = None
        self._layout_index: int = 0
        self._layout_stretch: int = 0

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

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Toggle fullscreen on double-click."""
        if self._is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Exit fullscreen on Escape key."""
        if event.key() == Qt.Key.Key_Escape and self._is_fullscreen:
            self._exit_fullscreen()
            event.accept()
        else:
            super().keyPressEvent(event)

    # ── internals ───────────────────────────────────────────────

    def _update_display(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        if self._is_fullscreen:
            # In fullscreen, scale to fit the screen
            super().setPixmap(self._original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            zoomed_size = QSize(
                int(self._original_pixmap.width() * self._zoom_level),
                int(self._original_pixmap.height() * self._zoom_level),
            )
            zoomed_pixmap = self._original_pixmap.scaledToWidth(
                zoomed_size.width(),
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(zoomed_pixmap)

    def _enter_fullscreen(self) -> None:
        """Move label into a fullscreen window."""
        parent = self.parentWidget()
        if parent is not None:
            self._parent_layout = parent.layout()
        self._is_fullscreen = True
        self._fullscreen_window = _FullscreenWindow(self)
        self._fullscreen_window.closed.connect(self._exit_fullscreen)
        self.setFocus()

    def _exit_fullscreen(self) -> None:
        """Restore label back into its parent layout."""
        self._is_fullscreen = False
        if self._fullscreen_window is not None:
            self._fullscreen_window.closed.disconnect(self._exit_fullscreen)
            self._fullscreen_window.layout().removeWidget(self)
            self._fullscreen_window.close()
            self._fullscreen_window.deleteLater()
            self._fullscreen_window = None
        # Re-insert into the saved parent layout
        if self._parent_layout is not None:
            self._parent_layout.addWidget(self, 1)
            self._parent_layout = None
        self.show()
        self._update_display()
