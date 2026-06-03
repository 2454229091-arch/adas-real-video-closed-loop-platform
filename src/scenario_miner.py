"""Scenario mining from deterministic detection records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .risk_metrics import compute_object_metrics


@dataclass
class ScenarioMiner:
    """Mine ADAS testing scenarios from frame-level detections."""

    fps: int = 10
    ego_speed_kmh: float = 50
    cooldown_frames: dict[str, int] = field(default_factory=lambda: {
        "close_following": 30,
        "front_vehicle_deceleration": 30,
        "pedestrian_crossing": 20,
        "cut_in": 25,
        "lane_departure_risk": 30,
    })

    def mine(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return scenario events with stable IDs and cooldown control."""
        events: list[dict[str, Any]] = []
        last_event_frame: dict[str, int] = {}
        previous_by_track: dict[str, dict[str, Any]] = {}

        for frame in frames:
            frame_id = int(frame["frame_id"])
            timestamp = float(frame.get("timestamp_s", frame_id / self.fps))
            ego_speed = float(frame.get("ego_speed_kmh", self.ego_speed_kmh))

            if abs(float(frame.get("ego_lane_offset_m", 0.0))) >= 0.8:
                self._maybe_add_event(
                    events,
                    last_event_frame,
                    "lane_departure_risk",
                    frame_id,
                    timestamp,
                    "ego_vehicle",
                    {},
                    {"risk_level": "HIGH", "risk_score": 78, "distance_m": None, "relative_speed_mps": 0, "ttc": None, "thw": None},
                    "Ego lane offset exceeds lane departure risk threshold.",
                )

            for obj in frame.get("objects", []):
                track_id = str(obj.get("track_id", "unknown"))
                prev_obj = previous_by_track.get(track_id)
                previous_timestamp = float(prev_obj.get("_timestamp_s", timestamp)) if prev_obj else timestamp
                delta_time_s = max(1.0 / self.fps, timestamp - previous_timestamp)
                metrics = compute_object_metrics(obj, prev_obj, delta_time_s, ego_speed)
                class_name = str(obj.get("class_name", "object"))
                lane_id = obj.get("lane_id")
                bbox = obj.get("bbox", [0, 0, 0, 0])
                cx = (float(bbox[0]) + float(bbox[2])) / 2 if len(bbox) >= 4 else 0

                if class_name in {"car", "truck", "bus"} and lane_id == 1:
                    if metrics["distance_m"] <= 14 and metrics["thw"] <= 1.2:
                        self._maybe_add_event(
                            events,
                            last_event_frame,
                            "close_following",
                            frame_id,
                            timestamp,
                            class_name,
                            obj,
                            metrics,
                            "Lead vehicle distance and headway are below safe following thresholds.",
                        )
                    if metrics["relative_speed_mps"] <= -8 or metrics["ttc"] <= 3:
                        self._maybe_add_event(
                            events,
                            last_event_frame,
                            "front_vehicle_deceleration",
                            frame_id,
                            timestamp,
                            class_name,
                            obj,
                            metrics,
                            "Front vehicle is closing rapidly with low time-to-collision.",
                        )

                if class_name == "person" and 420 <= cx <= 860 and metrics["distance_m"] <= 18:
                    self._maybe_add_event(
                        events,
                        last_event_frame,
                        "pedestrian_crossing",
                        frame_id,
                        timestamp,
                        class_name,
                        obj,
                        metrics,
                        "Pedestrian appears in the ego path region at short range.",
                    )

                if class_name in {"car", "truck", "bus"} and prev_obj:
                    prev_bbox = prev_obj.get("bbox", [0, 0, 0, 0])
                    prev_cx = (float(prev_bbox[0]) + float(prev_bbox[2])) / 2 if len(prev_bbox) >= 4 else cx
                    if prev_cx < 500 <= cx and lane_id == 1 and metrics["distance_m"] <= 18:
                        self._maybe_add_event(
                            events,
                            last_event_frame,
                            "cut_in",
                            frame_id,
                            timestamp,
                            class_name,
                            obj,
                            metrics,
                            "Vehicle moves from adjacent lane into the ego lane region.",
                        )

                previous_by_track[track_id] = {**obj, "_timestamp_s": timestamp}

        return events

    def _maybe_add_event(
        self,
        events: list[dict[str, Any]],
        last_event_frame: dict[str, int],
        scenario_type: str,
        frame_id: int,
        timestamp: float,
        object_type: str,
        obj: dict[str, Any],
        metrics: dict[str, Any],
        description: str,
    ) -> None:
        cooldown = self.cooldown_frames.get(scenario_type, 30)
        if frame_id - last_event_frame.get(scenario_type, -99999) < cooldown:
            return
        event_id = f"EVT-{len(events) + 1:06d}"
        events.append(
            {
                "event_id": event_id,
                "frame_id": frame_id,
                "timestamp": round(timestamp, 2),
                "start_frame": frame_id,
                "end_frame": frame_id,
                "start_time_s": round(timestamp, 2),
                "end_time_s": round(timestamp, 2),
                "scenario_type": scenario_type,
                "object_type": object_type,
                "bbox": obj.get("bbox", []),
                "distance_m": metrics.get("distance_m"),
                "relative_speed_mps": metrics.get("relative_speed_mps"),
                "ttc": metrics.get("ttc"),
                "thw": metrics.get("thw"),
                "risk_level": metrics.get("risk_level", "LOW"),
                "risk_score": metrics.get("risk_score", 0),
                "description": description,
                "evidence_frame": f"outputs/evidence_frames/{event_id}.jpg",
            }
        )
        last_event_frame[scenario_type] = frame_id
