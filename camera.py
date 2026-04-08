"""Camera Input Module.

Handles camera discovery, video backend selection, and FPS normalization.
"""

import sys
from typing import List

import cv2


ALLOWED_FPS = [24, 30, 60]


def video_backend_for_platform() -> int:
    """Return the best OpenCV video backend for the current OS."""
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("darwin"):
        return cv2.CAP_AVFOUNDATION
    return 0


def find_cameras(max_index: int = 10) -> List[int]:
    """Scan for available camera indices."""
    available: List[int] = []
    backend = video_backend_for_platform()
    for idx in range(max_index):
        cap = None
        try:
            cap = (
                cv2.VideoCapture(idx, backend)
                if backend != 0
                else cv2.VideoCapture(idx)
            )
            if cap is None:
                continue
            if cap.isOpened():
                available.append(idx)
            cap.release()
        except Exception:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return available


def nearest_allowed_fps(value: float) -> int:
    """Snap a raw FPS value to the nearest allowed FPS."""
    if value <= 1:
        return 30
    return min(ALLOWED_FPS, key=lambda x: abs(x - value))


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera with platform-specific backend, with fallback.

    Returns an opened ``cv2.VideoCapture`` or one that is *not* opened
    (caller must check ``cap.isOpened()``).
    """
    backend = video_backend_for_platform()
    cap = (
        cv2.VideoCapture(index, backend)
        if backend != 0
        else cv2.VideoCapture(index)
    )
    # Fallback: if the specific backend fails, try default constructor
    if not cap.isOpened() and backend != 0:
        try:
            cap.release()
        except Exception:
            pass
        cap = cv2.VideoCapture(index)
    return cap
