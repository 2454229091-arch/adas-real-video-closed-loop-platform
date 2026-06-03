# Interview Pitch

## 30-Second Summary

I built a Python-based ADAS scenario risk-mining and regression-test platform. It can analyze local driving videos with YOLOv8, generate evidence frames and annotated videos, and also run a deterministic synthetic pipeline that converts ADAS scenarios into risk events, defect reports, and regression test cases.

## 5-8 Minute Walkthrough

### Problem

ADAS testing is not only about detecting objects. A useful test workflow needs to answer these questions:

- What scenario happened?
- Why is it risky?
- What evidence proves it?
- Should it become a defect?
- What regression test should prevent it from recurring?

I built this project to simulate that engineering workflow locally.

### Architecture

The project has two data paths.

The first path handles real videos. It samples video frames, runs YOLOv8 detection, applies road-scene risk rules, saves evidence frames, and generates reports. It also adapts sampled YOLO detections into the same ADAS scenario-mining schema, so real videos can produce defect reports and regression test cases. It can render a new annotated MP4 with detection boxes and risk overlays.

The second path handles deterministic synthetic ADAS detections. It mines scenarios such as close following, front-vehicle deceleration, pedestrian crossing, cut-in, and lane departure risk. It computes TTC, THW, distance, relative speed, and risk score. Then it creates defect records and regression test cases.

### Implementation Details

The project is organized into small modules:

- `detector.py` handles YOLO inference.
- `risk_rules.py` handles frame-level video risk rules.
- `scenario_miner.py` handles ADAS scenario extraction.
- `risk_metrics.py` computes TTC, THW, and score signals.
- `defect_triage.py` turns risk events into defect records.
- `test_case_generator.py` turns defects into regression tests.
- `report_generator.py` exports reports and charts.
- `video_renderer.py` renders annotated videos.
- `dashboard.py` provides a local visual interface.

### Output

The project produces CSV, Excel, JSON, Markdown, HTML, chart images, evidence frames, and annotated MP4 files. This makes the result usable by both engineering and testing roles.

### Verification

I added pytest coverage for the core rule and reporting modules. I also verify the project with compile checks and end-to-end synthetic demo commands.

### Honest Limitation

The real-video closed loop is a first engineering version. It estimates distance from bounding-box geometry and uses sampled-frame continuity, so its TTC and THW values are useful for workflow demonstration but not production validation. The next improvement would be stronger object tracking and calibration.

### Closing

The key value of the project is that it connects perception output to a tester-style workflow. It is not just drawing boxes on a video; it shows how road-scene observations can become risk events, defects, and regression tests.
