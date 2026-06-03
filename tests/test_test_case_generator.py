from src.defect_triage import DefectTriage
from src.test_case_generator import TestCaseGenerator


def test_generate_test_cases_from_events():
    events = [
        {
            "event_id": "EVT-000001",
            "scenario_type": "cut_in",
            "risk_level": "HIGH",
            "description": "Cut-in event",
            "evidence_frame": "outputs/evidence_frames/EVT-000001.jpg",
        }
    ]
    defects = DefectTriage().triage(events)
    cases = TestCaseGenerator().generate(events, defects)

    assert len(cases) == 1
    assert cases[0]["test_case_id"] == "TC-ADAS-000001"


def test_test_case_contains_expected_fields():
    events = [
        {
            "event_id": "EVT-000001",
            "scenario_type": "close_following",
            "risk_level": "MEDIUM",
            "description": "Close following",
            "evidence_frame": "",
        }
    ]
    defects = DefectTriage().triage(events)
    case = TestCaseGenerator().generate(events, defects)[0]

    assert case["linked_event_id"] == "EVT-000001"
    assert case["priority"] == "P2"
    assert case["expected_result"]


def test_real_video_test_case_labels_source_correctly():
    events = [
        {
            "event_id": "EVT-000001",
            "scenario_type": "pedestrian_crossing",
            "risk_level": "HIGH",
            "description": "Pedestrian crossing",
            "evidence_frame": "outputs/run/annotated_frames/frame_000010.jpg",
        }
    ]
    defects = DefectTriage().triage(events)
    case = TestCaseGenerator().generate(events, defects, source="real_video")[0]

    assert "Real-video detections" in case["preconditions"]
    assert case["regression_type"] == "real_video_regression"
