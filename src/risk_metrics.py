"""Risk metric helpers for ADAS scenario mining."""

from __future__ import annotations

from math import inf
from typing import Any


DEFAULT_THRESHOLDS = {
    "ttc": {"critical": 1.5, "high": 3.0, "medium": 5.0},
    "thw": {"critical": 0.8, "high": 1.2, "medium": 2.0},
    "distance_m": {"critical": 5.0, "high": 10.0, "medium": 20.0},
}


def kmh_to_mps(speed_kmh: float | None) -> float:
    """Convert kilometers per hour to meters per second."""
    if speed_kmh is None:
        return 0.0
    return float(speed_kmh) / 3.6


def calculate_distance(bbox: list[float] | tuple[float, ...], frame_width: int = 1280, frame_height: int = 720) -> float:
    """Estimate relative distance from bbox height when explicit distance is unavailable."""
    del frame_width
    if not bbox or len(bbox) < 4:
        return inf
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return round(max(2.0, (frame_height / height) * 4.0), 2)


def calculate_relative_speed(prev_distance_m: float | None, curr_distance_m: float | None, delta_time_s: float) -> float:
    """Return positive speed when the gap increases and negative speed when the object is closing."""
    if prev_distance_m is None or curr_distance_m is None or delta_time_s <= 0:
        return 0.0
    return round((float(curr_distance_m) - float(prev_distance_m)) / delta_time_s, 3)


def calculate_ttc(distance_m: float | None, relative_speed_mps: float | None) -> float:
    """Calculate Time To Collision in seconds for closing objects."""
    if distance_m is None or relative_speed_mps is None:
        return inf
    if distance_m <= 0:
        return 0.0
    if relative_speed_mps >= 0:
        return inf
    return round(float(distance_m) / abs(float(relative_speed_mps)), 3)


def calculate_thw(distance_m: float | None, ego_speed_mps: float | None) -> float:
    """Calculate Time Headway in seconds."""
    if distance_m is None or ego_speed_mps is None or ego_speed_mps <= 0:
        return inf
    return round(float(distance_m) / float(ego_speed_mps), 3)


def calculate_risk_level(
    ttc: float | None,
    thw: float | None,
    distance_m: float | None,
    object_type: str,
    thresholds: dict[str, Any] | None = None,
) -> str:
    """Classify risk with more sensitivity for vulnerable road users."""
    cfg = thresholds or DEFAULT_THRESHOLDS
    ttc_value = inf if ttc is None else float(ttc)
    thw_value = inf if thw is None else float(thw)
    distance_value = inf if distance_m is None else float(distance_m)

    if (
        ttc_value <= cfg["ttc"]["critical"]
        or thw_value <= cfg["thw"]["critical"]
        or distance_value <= cfg["distance_m"]["critical"]
    ):
        return "CRITICAL"
    if (
        ttc_value <= cfg["ttc"]["high"]
        or thw_value <= cfg["thw"]["high"]
        or distance_value <= cfg["distance_m"]["high"]
    ):
        return "HIGH"
    if (
        ttc_value <= cfg["ttc"]["medium"]
        or thw_value <= cfg["thw"]["medium"]
        or distance_value <= cfg["distance_m"]["medium"]
    ):
        return "HIGH" if object_type in {"person", "bicycle", "motorcycle"} else "MEDIUM"
    return "LOW"


def calculate_risk_score(
    ttc: float | None,
    thw: float | None,
    distance_m: float | None,
    object_type: str,
) -> int:
    """Return a deterministic 0-100 risk score for report ranking."""
    score = 0.0
    if ttc is not None and ttc != inf:
        score += max(0.0, 45.0 * (1.0 - min(float(ttc), 6.0) / 6.0))
    if thw is not None and thw != inf:
        score += max(0.0, 25.0 * (1.0 - min(float(thw), 3.0) / 3.0))
    if distance_m is not None and distance_m != inf:
        score += max(0.0, 20.0 * (1.0 - min(float(distance_m), 30.0) / 30.0))
    if object_type in {"person", "bicycle", "motorcycle"}:
        score += 10.0
    return int(round(min(100.0, score)))


def compute_object_metrics(
    obj: dict[str, Any],
    prev_obj: dict[str, Any] | None,
    delta_time_s: float,
    ego_speed_kmh: float,
) -> dict[str, Any]:
    """Compute distance, relative speed, TTC, THW, level, and score for one object."""
    distance_m = float(obj.get("distance_m") or calculate_distance(obj.get("bbox", [])))
    prev_distance = float(prev_obj["distance_m"]) if prev_obj and prev_obj.get("distance_m") is not None else None
    relative_speed = calculate_relative_speed(prev_distance, distance_m, delta_time_s)
    ttc = calculate_ttc(distance_m, relative_speed)
    thw = calculate_thw(distance_m, kmh_to_mps(ego_speed_kmh))
    object_type = str(obj.get("class_name", "object"))
    risk_level = calculate_risk_level(ttc, thw, distance_m, object_type)
    risk_score = calculate_risk_score(ttc, thw, distance_m, object_type)
    return {
        "distance_m": round(distance_m, 2),
        "relative_speed_mps": relative_speed,
        "ttc": ttc,
        "thw": thw,
        "risk_level": risk_level,
        "risk_score": risk_score,
    }
