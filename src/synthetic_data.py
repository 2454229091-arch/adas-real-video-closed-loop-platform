"""Generate deterministic synthetic ADAS video and detection records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bbox(cx: int, cy: int, width: int, height: int) -> list[int]:
    return [cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2]


def build_synthetic_detection_data(frame_count: int = 120, fps: int = 10, ego_speed_kmh: float = 50) -> dict[str, Any]:
    """Build deterministic frame-level detections that trigger platform scenarios."""
    frames = []
    for frame_id in range(frame_count):
        timestamp = round(frame_id / fps, 2)
        lane_offset = 0.0
        objects: list[dict[str, Any]] = []

        if 5 <= frame_id <= 24:
            distance = 18.0 - (frame_id - 5) * 0.45
            objects.append(
                {
                    "track_id": "veh_close",
                    "class_name": "car",
                    "bbox": _bbox(640, 470, 130, 90),
                    "distance_m": round(distance, 2),
                    "speed_kmh": 35,
                    "lane_id": 1,
                    "confidence": 0.96,
                }
            )

        if 28 <= frame_id <= 44:
            distance = 28.0 - (frame_id - 28) * 1.25
            objects.append(
                {
                    "track_id": "veh_brake",
                    "class_name": "car",
                    "bbox": _bbox(635, 455, 120, 82),
                    "distance_m": round(distance, 2),
                    "speed_kmh": 20,
                    "lane_id": 1,
                    "confidence": 0.94,
                }
            )

        if 48 <= frame_id <= 66:
            cx = 360 + (frame_id - 48) * 16
            objects.append(
                {
                    "track_id": "ped_cross",
                    "class_name": "person",
                    "bbox": _bbox(cx, 485, 38, 100),
                    "distance_m": 12.0,
                    "speed_kmh": 5,
                    "lane_id": 1,
                    "confidence": 0.91,
                }
            )

        if 70 <= frame_id <= 88:
            cx = 330 + (frame_id - 70) * 20
            lane_id = 0 if frame_id < 78 else 1
            objects.append(
                {
                    "track_id": "veh_cutin",
                    "class_name": "car",
                    "bbox": _bbox(cx, 465, 120, 86),
                    "distance_m": 16.0,
                    "speed_kmh": 42,
                    "lane_id": lane_id,
                    "confidence": 0.93,
                }
            )

        if 82 <= frame_id <= 108:
            lane_offset = 0.95
            objects.append(
                {
                    "track_id": "lane_proxy",
                    "class_name": "lane_marker",
                    "bbox": _bbox(640, 600, 20, 20),
                    "distance_m": 0.0,
                    "speed_kmh": 0,
                    "lane_id": 1,
                    "confidence": 1.0,
                }
            )

        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_s": timestamp,
                "ego_speed_kmh": ego_speed_kmh,
                "ego_lane_id": 1,
                "ego_lane_offset_m": lane_offset,
                "objects": objects,
            }
        )

    return {
        "metadata": {"fps": fps, "ego_speed_kmh": ego_speed_kmh, "source": "synthetic"},
        "frames": frames,
    }


def write_synthetic_video(path: Path, frame_count: int = 120, fps: int = 10) -> Path:
    """Create a simple annotated synthetic driving video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for frame_id in range(frame_count):
        frame = np.full((720, 1280, 3), (42, 48, 52), dtype=np.uint8)
        cv2.line(frame, (430, 720), (565, 360), (180, 180, 180), 4)
        cv2.line(frame, (850, 720), (715, 360), (180, 180, 180), 4)
        cv2.rectangle(frame, (565, 360), (715, 685), (80, 120, 140), 2)
        cv2.putText(frame, f"Synthetic ADAS Frame {frame_id:03d}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        writer.write(frame)
    writer.release()
    return path


def generate_synthetic_assets(output_dir: Path = Path("data/synthetic"), fps: int = 10, ego_speed_kmh: float = 50) -> dict[str, Path]:
    """Generate synthetic video and matching detections JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = write_synthetic_video(output_dir / "synthetic_drive.mp4", fps=fps)
    detections = build_synthetic_detection_data(fps=fps, ego_speed_kmh=ego_speed_kmh)
    detections_path = output_dir / "synthetic_detections.json"
    detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return {"video": video_path, "detections": detections_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic ADAS test data.")
    parser.add_argument("--output", default="data/synthetic/synthetic_drive.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output)
    assets = generate_synthetic_assets(output.parent, fps=args.fps)
    if output != assets["video"]:
        assets["video"].replace(output)
    print(f"Synthetic video saved to: {output}")
    print(f"Synthetic detections saved to: {output.parent / 'synthetic_detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
