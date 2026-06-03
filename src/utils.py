"""Utility helpers for video analysis, annotation, and file handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


RISK_ZONE_COLOR = (0, 180, 255)
BOX_COLOR = (70, 220, 80)
HIGH_COLOR = (30, 30, 230)
TEXT_COLOR = (255, 255, 255)


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_risk_zone(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    return (
        int(frame_width * 0.35),
        int(frame_height * 0.45),
        int(frame_width * 0.65),
        int(frame_height * 0.95),
    )


def calculate_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def draw_risk_zone(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = get_risk_zone(width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), RISK_ZONE_COLOR, 2)
    cv2.putText(
        frame,
        "Driving risk zone",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        RISK_ZONE_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        color = HIGH_COLOR if detection.get("in_risk_zone") else BOX_COLOR
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_event_text(frame: np.ndarray, events: list[dict[str, str]]) -> np.ndarray:
    for index, event in enumerate(events[:4]):
        text = f'{event["risk_level"]}: {event["event_type"]}'
        y = 30 + index * 28
        cv2.rectangle(frame, (8, y - 22), (620, y + 6), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return frame


def save_annotated_frame(frame: np.ndarray, output_dir: Path, frame_number: int) -> Path:
    ensure_dir(output_dir)
    path = output_dir / f"frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def summarize_detected_objects(detections: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for detection in detections:
        class_name = str(detection["class_name"])
        counts[class_name] = counts.get(class_name, 0) + 1
    return ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
