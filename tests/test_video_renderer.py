import cv2
import numpy as np

from src.video_renderer import render_frames_to_video


def test_render_frames_to_video_creates_output(tmp_path):
    frames = [np.full((120, 160, 3), 60, dtype=np.uint8) for _ in range(3)]
    detections_by_frame = {
        0: [
            {
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [20, 30, 80, 90],
                "center_x": 50,
                "center_y": 60,
                "in_risk_zone": True,
            }
        ]
    }
    events_by_frame = {
        0: [
            {
                "event_type": "Dense traffic",
                "risk_level": "Medium",
                "tester_note": "Vehicle count exceeded the dense-traffic threshold.",
            }
        ]
    }
    output = tmp_path / "annotated.mp4"

    render_frames_to_video(frames, detections_by_frame, events_by_frame, output, fps=5)

    assert output.exists()
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
