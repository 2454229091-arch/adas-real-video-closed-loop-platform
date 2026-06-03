from pathlib import Path

from src.defect_triage import DefectTriage
from src.report_generator import write_platform_reports
from src.test_case_generator import TestCaseGenerator


def test_platform_report_outputs_created(tmp_path):
    events = [
        {
            "event_id": "EVT-000001",
            "frame_id": 10,
            "timestamp": 1.0,
            "scenario_type": "pedestrian_crossing",
            "object_type": "person",
            "distance_m": 8.0,
            "relative_speed_mps": -2.0,
            "ttc": 4.0,
            "thw": 0.6,
            "risk_level": "HIGH",
            "risk_score": 82,
            "description": "Pedestrian crossing",
            "evidence_frame": str(tmp_path / "EVT-000001.jpg"),
        }
    ]
    defects = DefectTriage().triage(events)
    cases = TestCaseGenerator().generate(events, defects)

    paths = write_platform_reports(events, defects, cases, tmp_path, metadata={"source": "test"})

    assert paths["event_log"].exists()
    assert paths["risk_events"].exists()
    assert paths["test_cases_xlsx"].exists()
    assert paths["defect_report"].exists()
    assert paths["summary"].exists()
    assert paths["html"].exists()
