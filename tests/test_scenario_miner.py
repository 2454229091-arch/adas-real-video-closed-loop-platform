from src.scenario_miner import ScenarioMiner
from src.synthetic_data import build_synthetic_detection_data


def mine_events():
    data = build_synthetic_detection_data(frame_count=90, fps=10, ego_speed_kmh=50)
    miner = ScenarioMiner(fps=10, ego_speed_kmh=50)
    return miner.mine(data["frames"])


def test_close_following_event_detected():
    scenario_types = {event["scenario_type"] for event in mine_events()}

    assert "close_following" in scenario_types


def test_pedestrian_crossing_event_detected():
    scenario_types = {event["scenario_type"] for event in mine_events()}

    assert "pedestrian_crossing" in scenario_types


def test_cut_in_event_detected():
    scenario_types = {event["scenario_type"] for event in mine_events()}

    assert "cut_in" in scenario_types


def test_front_vehicle_deceleration_event_detected():
    scenario_types = {event["scenario_type"] for event in mine_events()}

    assert "front_vehicle_deceleration" in scenario_types


def test_lane_departure_event_detected():
    scenario_types = {event["scenario_type"] for event in mine_events()}

    assert "lane_departure_risk" in scenario_types


def test_event_cooldown_prevents_duplicates():
    events = mine_events()
    scenario_counts = {}
    for event in events:
        scenario_counts[event["scenario_type"]] = scenario_counts.get(event["scenario_type"], 0) + 1

    assert max(scenario_counts.values()) <= 3
    assert all(event["event_id"].startswith("EVT-") for event in events)


def test_relative_speed_uses_timestamp_gap_between_observations():
    frames = [
        {
            "frame_id": 0,
            "timestamp_s": 0.0,
            "ego_speed_kmh": 50,
            "objects": [
                {
                    "track_id": "lead",
                    "class_name": "car",
                    "bbox": [575, 430, 705, 520],
                    "distance_m": 20.0,
                    "lane_id": 1,
                    "confidence": 0.9,
                }
            ],
        },
        {
            "frame_id": 30,
            "timestamp_s": 1.0,
            "ego_speed_kmh": 50,
            "objects": [
                {
                    "track_id": "lead",
                    "class_name": "car",
                    "bbox": [575, 430, 705, 520],
                    "distance_m": 10.0,
                    "lane_id": 1,
                    "confidence": 0.9,
                }
            ],
        },
    ]

    events = ScenarioMiner(fps=30, ego_speed_kmh=50).mine(frames)

    assert events[0]["relative_speed_mps"] == -10.0
    assert events[0]["ttc"] == 1.0
