"""AI Object Detection Module.

Handles YOLO model loading, device management, and inference.
"""

from pathlib import Path
from typing import Any, List, Optional, cast

try:
    from ultralytics import YOLO  # type: ignore

    YOLO("yolo11n.pt")
except Exception:
    YOLO = None


def cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


def load_model(model_path: Path, device: str = "cpu") -> Any:
    """Load a YOLO model and move it to *device*.

    Returns the model object, or ``None`` on failure.
    Raises ``RuntimeError`` when ultralytics is not installed.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics is not available. Install requirements.")

    model = YOLO(str(model_path))

    # Move model to the specified device
    try:
        if hasattr(model, "to"):
            cast(Any, model).to(device)
        else:
            inner_model = getattr(model, "model", None)
            if device != "cpu" and inner_model is not None and hasattr(inner_model, "to"):
                cast(Any, inner_model).to(device)
    except Exception:
        pass  # Fallback to CPU if device setting fails

    return model


def get_model_classes(model: Any) -> List[str]:
    """Extract the list of class names from a loaded YOLO model."""
    if model is None:
        return []
    try:
        return list(model.names.values())
    except Exception:
        return []


def extract_detections(results: Any) -> List[dict]:
    """Parse YOLO results into a list of detection dicts.

    Each dict contains keys: x1, y1, x2, y2, label, class_name.
    """
    detections: List[dict] = []
    if results is None:
        return detections
    boxes = results.boxes
    names = results.names
    if boxes is None:
        return detections
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        class_name = str(names.get(cls_id, cls_id))
        label = f"{class_name} {conf:.2f}"
        x1, y1, x2, y2 = xyxy.tolist()
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "class_name": class_name,
            }
        )
    return detections


def run_inference(model: Any, frame: Any) -> List[dict]:
    """Run YOLO inference on a single frame and return detections."""
    if model is None:
        return []
    try:
        results = model(frame, verbose=False)[0]
    except Exception:
        return []
    return extract_detections(results)
