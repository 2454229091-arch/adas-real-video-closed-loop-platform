"""Adapt real-video YOLO detections into ADAS scenario-mining records."""

from __future__ import annotations

from math import inf
from typing import Any

from .risk_metrics import calculate_distance


SUPPORTED_SCENARIO_CLASSES = {"person", "bicycle", "motorcycle", "car", "truck", "bus"}


def build_real_video_detection_data(
    video_name: str,
    fps: float,
    ego_speed_kmh: float,
    frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert sampled YOLO frame detections into ScenarioMiner input."""
    frames: list[dict[str, Any]] = []
    for record in frame_records:
        width = int(record.get("frame_width", 1280))
        height = int(record.get("frame_height", 720))
        objects = []
        for index, detection in enumerate(record.get("detections", [])):
            class_name = str(detection.get("class_name", "object"))
            if class_name not in SUPPORTED_SCENARIO_CLASSES:
                continue

            bbox = [float(value) for value in detection.get("bbox", [])]
            center_x = float(detection.get("center_x", _bbox_center_x(bbox)))
            lane_id = _estimate_lane_id(center_x, width)
            objects.append(
                {
                    "track_id": _build_track_id(class_name, lane_id, index),
                    "class_name": class_name,
                    "bbox": bbox,
                    "distance_m": _estimate_distance(class_name, bbox, width, height),
                    "speed_kmh": None,
                    "lane_id": lane_id,
                    "confidence": float(detection.get("confidence", 0.0)),
                }
            )

        frames.append(
            {
                "frame_id": int(record["frame_number"]),
                "timestamp_s": float(record.get("timestamp_sec", int(record["frame_number"]) / fps)),
                "ego_speed_kmh": ego_speed_kmh,
                "ego_lane_id": 1,
                "ego_lane_offset_m": 0.0,
                "objects": objects,
            }
        )

    return {
        "metadata": {
            "source": "real_video",
            "video_name": video_name,
            "fps": fps,
            "ego_speed_kmh": ego_speed_kmh,
            "adapter": "bbox_geometry_v1",
        },
        "frames": frames,
    }


def link_real_video_evidence(events: list[dict[str, Any]], evidence_by_frame: dict[int, str]) -> None:
    """Attach nearest available real-video evidence image paths to mined events."""
    if not evidence_by_frame:
        return
    available_frames = sorted(evidence_by_frame)
    for event in events:
        frame_id = int(event.get("frame_id", 0))
        nearest_frame = min(available_frames, key=lambda candidate: abs(candidate - frame_id))
        event["evidence_frame"] = evidence_by_frame[nearest_frame]


def _bbox_center_x(bbox: list[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return (bbox[0] + bbox[2]) / 2.0


def _estimate_lane_id(center_x: float, frame_width: int) -> int:
    if frame_width <= 0:
        return 1
    normalized_x = center_x / frame_width
    if 0.35 <= normalized_x <= 0.65:
        return 1
    return 0 if normalized_x < 0.35 else 2


def _build_track_id(class_name: str, lane_id: int, index: int) -> str:
    return f"real_{class_name.replace(' ', '_')}_{lane_id}_{index}"


def _estimate_distance(class_name: str, bbox: list[float], frame_width: int, frame_height: int) -> float:
    if len(bbox) < 4:
        return inf
    distance = calculate_distance(bbox, frame_width=frame_width, frame_height=frame_height)
    if class_name in {"person", "bicycle", "motorcycle"}:
        return round(distance * 0.8, 2)
    return distance
