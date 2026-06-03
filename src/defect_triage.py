"""Convert mined risk events into tester-style defect records."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PRIORITY_BY_RISK = {"CRITICAL": "P0", "HIGH": "P1", "MEDIUM": "P2", "LOW": "P3"}
FEATURE_BY_SCENARIO = {
    "close_following": ["ACC", "AEB", "Longitudinal Control"],
    "front_vehicle_deceleration": ["AEB", "FCW", "Longitudinal Control"],
    "pedestrian_crossing": ["AEB", "VRU", "Perception"],
    "cut_in": ["ACC", "AEB", "Lane Change Handling"],
    "lane_departure_risk": ["LKA", "LDW", "Lateral Control"],
}


class DefectTriage:
    """Build prioritized defect records from ADAS risk events."""

    def triage(self, events: list[dict[str, Any]], source: str = "synthetic") -> list[dict[str, Any]]:
        defects = []
        first_step = (
            "Load the source video and generated detection report."
            if source == "real_video"
            else "Load the synthetic detection log or source video."
        )
        for index, event in enumerate(events, start=1):
            scenario_type = str(event["scenario_type"])
            risk_level = str(event.get("risk_level", "LOW"))
            feature_tags = FEATURE_BY_SCENARIO.get(scenario_type, ["ADAS Test"])
            defect_id = f"BUG-{index:06d}"
            defects.append(
                {
                    "defect_id": defect_id,
                    "linked_event_id": event["event_id"],
                    "priority": PRIORITY_BY_RISK.get(risk_level, "P3"),
                    "feature": " / ".join(feature_tags[:2]),
                    "feature_tags": feature_tags,
                    "scenario_type": scenario_type,
                    "title": f"{risk_level.title()} {scenario_type.replace('_', ' ')} event detected",
                    "symptom": event.get("description", "Risk event detected by scenario miner."),
                    "trigger_condition": f"Scenario={scenario_type}, risk={risk_level}, score={event.get('risk_score', 0)}",
                    "reproduction_steps": [
                        first_step,
                        f"Navigate to frame {event.get('frame_id', '?')} at {event.get('timestamp', '?')} seconds.",
                        "Review the evidence frame and ADAS scenario metrics.",
                    ],
                    "expected_behavior": "ADAS feature should maintain a safe response envelope for the scenario.",
                    "actual_behavior": event.get("description", "Risk threshold was exceeded."),
                    "evidence": event.get("evidence_frame", ""),
                    "suggested_owner": "ADAS Test / Autopilot QA",
                }
            )
        return defects


def write_defect_report(defects: list[dict[str, Any]], output_path: Path) -> Path:
    """Write a Markdown defect report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ADAS Defect Triage Report", ""]
    if not defects:
        lines.append("No defects generated.")
    for defect in defects:
        lines.extend(
            [
                f"## {defect['defect_id']} - {defect['title']}",
                f"- Priority: {defect['priority']}",
                f"- Linked Event: {defect['linked_event_id']}",
                f"- Feature Tags: {', '.join(defect['feature_tags'])}",
                f"- Trigger: {defect['trigger_condition']}",
                f"- Evidence: {defect['evidence']}",
                "- Reproduction Steps:",
                *[f"  {idx}. {step}" for idx, step in enumerate(defect["reproduction_steps"], start=1)],
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
