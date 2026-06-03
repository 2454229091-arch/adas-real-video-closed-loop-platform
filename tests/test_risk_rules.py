from src.risk_rules import evaluate_risks


def event_types(events):
    return {event["event_type"]: event["risk_level"] for event in events}


def detection(class_name, in_risk_zone=False):
    return {
        "class_name": class_name,
        "confidence": 0.9,
        "bbox": [100, 100, 200, 200],
        "center_x": 150,
        "center_y": 150,
        "in_risk_zone": in_risk_zone,
    }


def test_person_in_risk_zone_triggers_high_pedestrian_event():
    events = evaluate_risks([detection("person", in_risk_zone=True)], 100)

    assert event_types(events)["Pedestrian in driving path"] == "High"


def test_vulnerable_road_user_in_risk_zone_triggers_high_event():
    events = evaluate_risks([detection("bicycle", in_risk_zone=True)], 100)

    assert event_types(events)["Vulnerable road user nearby"] == "High"


def test_low_brightness_triggers_medium_low_visibility_event():
    events = evaluate_risks([], 45)

    assert event_types(events)["Low visibility"] == "Medium"


def test_vehicle_count_above_threshold_triggers_dense_traffic_event():
    detections = [detection("car") for _ in range(9)]

    events = evaluate_risks(detections, 100)

    assert event_types(events)["Dense traffic"] == "Medium"


def test_traffic_control_object_triggers_low_event():
    events = evaluate_risks([detection("traffic light")], 100)

    assert event_types(events)["Traffic control scene"] == "Low"


def test_large_vehicle_in_risk_zone_triggers_medium_event():
    events = evaluate_risks([detection("truck", in_risk_zone=True)], 100)

    assert event_types(events)["Large vehicle nearby"] == "Medium"


def test_multiple_risks_in_one_frame_adds_multi_risk_event():
    events = evaluate_risks(
        [detection("person", in_risk_zone=True), detection("traffic light")],
        100,
    )

    assert event_types(events)["Multi-risk frame"] == "High"


def test_empty_detections_with_normal_brightness_do_not_trigger_events():
    events = evaluate_risks([], 100)

    assert events == []


def test_single_low_risk_event_does_not_trigger_multi_risk_event():
    events = evaluate_risks([detection("stop sign")], 100)

    assert "Traffic control scene" in event_types(events)
    assert "Multi-risk frame" not in event_types(events)


def test_eight_vehicles_does_not_trigger_dense_traffic_event():
    events = evaluate_risks([detection("car") for _ in range(8)], 100)

    assert "Dense traffic" not in event_types(events)


def test_brightness_at_threshold_does_not_trigger_low_visibility_event():
    events = evaluate_risks([], 60)

    assert "Low visibility" not in event_types(events)


def test_brightness_below_threshold_triggers_low_visibility_event():
    events = evaluate_risks([], 59.9)

    assert event_types(events)["Low visibility"] == "Medium"


def test_large_vehicle_outside_risk_zone_does_not_trigger_large_vehicle_event():
    events = evaluate_risks([detection("bus", in_risk_zone=False)], 100)

    assert "Large vehicle nearby" not in event_types(events)


def test_multiple_people_in_risk_zone_do_not_duplicate_pedestrian_event():
    events = evaluate_risks(
        [detection("person", in_risk_zone=True), detection("person", in_risk_zone=True)],
        100,
        1280,
        720,
    )

    pedestrian_events = [
        event for event in events if event["event_type"] == "Pedestrian in driving path"
    ]
    assert len(pedestrian_events) == 1


def test_person_and_low_visibility_same_frame_triggers_multi_risk_event():
    events = evaluate_risks([detection("person", in_risk_zone=True)], 50)

    assert event_types(events)["Multi-risk frame"] == "High"
