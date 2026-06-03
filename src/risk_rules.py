"""Rule-based risk evaluation for road-scene test logging."""

from __future__ import annotations

from typing import Any


VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
VULNERABLE_CLASSES = {"bicycle", "motorcycle"}
TRAFFIC_CONTROL_CLASSES = {"traffic light", "stop sign"}
LARGE_VEHICLE_CLASSES = {"bus", "truck"}


def _event(event_type: str, risk_level: str, tester_note: str) -> dict[str, str]:
    return {
        "event_type": event_type,
        "risk_level": risk_level,
        "tester_note": tester_note,
    }


def _has_class_in_zone(detections: list[dict[str, Any]], class_names: set[str]) -> bool:
    return any(
        detection.get("class_name") in class_names and bool(detection.get("in_risk_zone"))
        for detection in detections
    )


def evaluate_risks(
    detections: list[dict[str, Any]],
    brightness: float,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> list[dict[str, str]]:
    """Evaluate one analyzed frame and return tester-style risk events."""
    del frame_width, frame_height

    events: list[dict[str, str]] = []

    if _has_class_in_zone(detections, {"person"}):
        events.append(
            _event(
                "Pedestrian in driving path",
                "High",
                "Pedestrian detected inside the predefined driving risk zone.",
            )
        )

    if _has_class_in_zone(detections, VULNERABLE_CLASSES):
        events.append(
            _event(
                "Vulnerable road user nearby",
                "High",
                "Bicycle or motorcycle detected inside the driving risk zone.",
            )
        )

    vehicle_count = sum(1 for detection in detections if detection.get("class_name") in VEHICLE_CLASSES)
    if vehicle_count > 8:
        events.append(
            _event(
                "Dense traffic",
                "Medium",
                "Vehicle count exceeded the dense-traffic threshold.",
            )
        )

    if brightness < 60:
        events.append(
            _event(
                "Low visibility",
                "Medium",
                "Average frame brightness is below the visibility threshold.",
            )
        )

    if any(detection.get("class_name") in TRAFFIC_CONTROL_CLASSES for detection in detections):
        events.append(
            _event(
                "Traffic control scene",
                "Low",
                "Traffic control object detected.",
            )
        )

    if _has_class_in_zone(detections, LARGE_VEHICLE_CLASSES):
        events.append(
            _event(
                "Large vehicle nearby",
                "Medium",
                "Large vehicle detected inside the driving risk zone.",
            )
        )

    if len(events) >= 2:
        events.append(
            _event(
                "Multi-risk frame",
                "High",
                "Multiple risk indicators were detected in the same frame.",
            )
        )

    return events
