"""YOLO detector wrapper for selected road-scene classes."""

from __future__ import annotations

from typing import Any

from .utils import get_risk_zone


CLASSES_OF_INTEREST = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


class RoadSceneDetector:
    """Small wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: ultralytics. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        risk_x1, risk_y1, risk_x2, risk_y2 = get_risk_zone(width, height)
        results = self.model.predict(frame, conf=self.confidence, verbose=False)

        detections: list[dict[str, Any]] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                if class_name not in CLASSES_OF_INTEREST:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "in_risk_zone": risk_x1 <= center_x <= risk_x2
                        and risk_y1 <= center_y <= risk_y2,
                    }
                )

        return detections
