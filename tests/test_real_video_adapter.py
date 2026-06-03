from src.real_video_adapter import (
    build_real_video_detection_data,
    link_real_video_evidence,
)
from src.scenario_miner import ScenarioMiner


def test_real_video_vehicle_detection_maps_to_ego_lane_object():
    frames = build_real_video_detection_data(
        video_name="drive.mp4",
        fps=10,
        ego_speed_kmh=50,
        frame_records=[
            {
                "frame_number": 10,
                "timestamp_sec": 1.0,
                "frame_width": 1280,
                "frame_height": 720,
                "detections": [
                    {
                        "class_name": "car",
                        "confidence": 0.9,
                        "bbox": [575, 430, 705, 520],
                        "center_x": 640,
                        "center_y": 475,
                    }
                ],
            }
        ],
    )

    obj = frames["frames"][0]["objects"][0]

    assert frames["metadata"]["source"] == "real_video"
    assert obj["lane_id"] == 1
    assert obj["track_id"].startswith("real_car_")
    assert obj["distance_m"] > 0


def test_real_video_person_detection_can_mine_pedestrian_crossing_event():
    data = build_real_video_detection_data(
        video_name="drive.mp4",
        fps=10,
        ego_speed_kmh=50,
        frame_records=[
            {
                "frame_number": 20,
                "timestamp_sec": 2.0,
                "frame_width": 1280,
                "frame_height": 720,
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.92,
                        "bbox": [620, 390, 660, 560],
                        "center_x": 640,
                        "center_y": 475,
                    }
                ],
            }
        ],
    )

    events = ScenarioMiner(fps=10, ego_speed_kmh=50).mine(data["frames"])

    assert events
    assert events[0]["scenario_type"] == "pedestrian_crossing"
    assert events[0]["object_type"] == "person"


def test_link_real_video_evidence_uses_nearest_available_frame():
    events = [{"event_id": "EVT-000001", "frame_id": 12, "evidence_frame": ""}]

    link_real_video_evidence(events, {10: "outputs/run/annotated_frames/frame_000010.jpg"})

    assert events[0]["evidence_frame"] == "outputs/run/annotated_frames/frame_000010.jpg"
