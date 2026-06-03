"""Render annotated object detections and risk events into a video file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2

from .utils import draw_detections, draw_event_text, draw_risk_zone, ensure_dir


def render_frames_to_video(
    frames: Iterable[Any],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    events_by_frame: dict[int, list[dict[str, str]]],
    output_path: Path,
    fps: float,
) -> Path:
    """Render already-loaded frames into an annotated MP4 file."""
    ensure_dir(output_path.parent)
    writer = None
    try:
        for frame_index, frame in enumerate(frames):
            annotated = frame.copy()
            draw_risk_zone(annotated)
            draw_detections(annotated, detections_by_frame.get(frame_index, []))
            draw_event_text(annotated, events_by_frame.get(frame_index, []))

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()
    return output_path


def render_video_with_detector(
    video_path: Path,
    output_path: Path,
    detector: Any,
    evaluate_risks: Any,
    calculate_brightness: Any,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Run detection over a video and save an annotated video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    last_detections: list[dict[str, Any]] = []
    last_events: list[dict[str, str]] = []
    processed_frames = 0
    risk_frames = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % max(1, frame_stride) == 0:
                brightness = calculate_brightness(frame)
                last_detections = detector.detect(frame)
                last_events = evaluate_risks(last_detections, brightness)

            annotated = frame.copy()
            draw_risk_zone(annotated)
            draw_detections(annotated, last_detections)
            draw_event_text(annotated, last_events)
            writer.write(annotated)

            processed_frames += 1
            if last_events:
                risk_frames += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "output_path": output_path,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "risk_frames": risk_frames,
    }
