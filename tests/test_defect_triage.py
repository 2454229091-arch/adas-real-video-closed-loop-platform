from src.defect_triage import DefectTriage


def event(scenario_type="front_vehicle_deceleration", risk_level="HIGH"):
    return {
        "event_id": "EVT-000001",
        "scenario_type": scenario_type,
        "risk_level": risk_level,
        "description": "Synthetic event",
        "evidence_frame": "outputs/evidence_frames/EVT-000001.jpg",
    }


def test_priority_mapping():
    defects = DefectTriage().triage([event(risk_level="CRITICAL")])

    assert defects[0]["priority"] == "P0"


def test_feature_mapping():
    defects = DefectTriage().triage([event(scenario_type="pedestrian_crossing")])

    assert "VRU" in defects[0]["feature_tags"]


def test_defect_record_contains_reproduction_steps():
    defects = DefectTriage().triage([event()])

    assert defects[0]["reproduction_steps"]
    assert defects[0]["linked_event_id"] == "EVT-000001"


def test_real_video_reproduction_steps_do_not_reference_synthetic_log():
    defects = DefectTriage().triage([event()], source="real_video")

    assert defects[0]["reproduction_steps"][0] == "Load the source video and generated detection report."
