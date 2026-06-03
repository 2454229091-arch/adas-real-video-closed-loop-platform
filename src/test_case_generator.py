"""Generate regression test cases from ADAS risk events and defects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class TestCaseGenerator:
    """Create deterministic regression test cases for triaged defects."""

    def generate(
        self,
        events: list[dict[str, Any]],
        defects: list[dict[str, Any]],
        source: str = "synthetic",
    ) -> list[dict[str, Any]]:
        event_by_id = {event["event_id"]: event for event in events}
        source_label = "Real-video detections" if source == "real_video" else "Synthetic ADAS detections"
        regression_type = "real_video_regression" if source == "real_video" else "synthetic_regression"
        cases = []
        for index, defect in enumerate(defects, start=1):
            event = event_by_id.get(defect["linked_event_id"], {})
            scenario = defect["scenario_type"]
            cases.append(
                {
                    "test_case_id": f"TC-ADAS-{index:06d}",
                    "scenario_type": scenario,
                    "linked_event_id": defect["linked_event_id"],
                    "linked_defect_id": defect["defect_id"],
                    "priority": defect["priority"],
                    "feature_tags": ", ".join(defect["feature_tags"]),
                    "preconditions": f"{source_label} are available and ego speed is configured.",
                    "test_steps": "Replay the detection sequence, mine scenarios, and verify generated risk metrics.",
                    "expected_result": f"System flags {scenario} and produces a bounded ADAS response requirement.",
                    "evaluation_metrics": f"TTC={event.get('ttc')}, THW={event.get('thw')}, risk_score={event.get('risk_score')}",
                    "regression_type": regression_type,
                    "evidence": defect.get("evidence", ""),
                }
            )
        return cases


def write_test_cases(test_cases: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Write test cases to Excel and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(test_cases)
    csv_path = output_dir / "test_cases.csv"
    xlsx_path = output_dir / "test_cases.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}
