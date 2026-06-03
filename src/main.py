"""Command-line entry point for FSD Road Scene Test Logger."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a road video and generate ADAS-style risk reports.")
    parser.add_argument("--video", help="Path to the input road video.")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        default="outputs",
        help="Output directory. Defaults to outputs.",
    )
    parser.add_argument(
        "--frame-interval",
        "--interval",
        dest="interval",
        type=float,
        default=1.0,
        help="Analyze one frame every N seconds.",
    )
    parser.add_argument(
        "--confidence",
        "--conf",
        dest="confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name.")
    parser.add_argument("--render-video", action="store_true", help="Render an annotated video with detection boxes and risk overlays.")
    parser.add_argument("--render-stride", type=int, default=1, help="Run detection every N frames while rendering video. Defaults to every frame.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic video and detections, then run the ADAS platform pipeline.")
    parser.add_argument("--use-synthetic-detections", action="store_true", help="Run the ADAS platform pipeline from data/synthetic/synthetic_detections.json.")
    parser.add_argument("--skip-detection", action="store_true", help="Skip YOLO and read frame detections from JSON.")
    parser.add_argument("--detections", help="Path to a synthetic or external detections JSON file.")
    parser.add_argument("--fps", type=int, default=10, help="Synthetic or detection-log FPS. Defaults to 10.")
    parser.add_argument("--ego-speed-kmh", type=float, default=50.0, help="Ego speed for synthetic metrics.")
    parser.add_argument("--save-evidence", action=argparse.BooleanOptionalAction, default=True, help="Save simple evidence frames for mined events.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.generate_synthetic or args.use_synthetic_detections or args.skip_detection:
        return run_platform_pipeline(args)

    if not args.video:
        print("Error: --video is required unless --generate-synthetic, --use-synthetic-detections, or --skip-detection is used.")
        return 1

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        print("Place a driving video in data/raw/, for example data/raw/sample_drive.mp4.")
        return 1

    if args.render_video:
        return run_video_render(args, video_path)

    try:
        import cv2

        from src.detector import RoadSceneDetector
        from src.real_video_adapter import build_real_video_detection_data, link_real_video_evidence
        from src.report_generator import build_summary, write_reports
        from src.risk_rules import evaluate_risks
        from src.utils import (
            calculate_brightness,
            draw_detections,
            draw_event_text,
            draw_risk_zone,
            ensure_dir,
            save_annotated_frame,
            summarize_detected_objects,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    output_dir = ensure_dir(args.output_dir)
    annotated_dir = ensure_dir(output_dir / "annotated_frames")

    detector = RoadSceneDetector(model_path=args.model, confidence=args.confidence)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Error: unable to open video file: {video_path}")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_step = max(1, int(round(fps * args.interval)))

    events: list[dict[str, object]] = []
    frame_records: list[dict[str, Any]] = []
    evidence_by_frame: dict[int, str] = {}
    detected_object_counts: Counter[str] = Counter()
    total_frames_analyzed = 0
    frame_number = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        total_frames_analyzed += 1
        height, width = frame.shape[:2]
        timestamp_sec = round(frame_number / fps, 2)
        brightness = round(calculate_brightness(frame), 2)
        detections = detector.detect(frame)
        frame_records.append(
            {
                "frame_number": frame_number,
                "timestamp_sec": timestamp_sec,
                "frame_width": width,
                "frame_height": height,
                "detections": detections,
            }
        )
        detected_object_counts.update(str(detection["class_name"]) for detection in detections)
        frame_events = evaluate_risks(detections, brightness, width, height)

        evidence_image = ""
        if frame_events:
            annotated = frame.copy()
            draw_risk_zone(annotated)
            draw_detections(annotated, detections)
            draw_event_text(annotated, frame_events)
            evidence_path = save_annotated_frame(annotated, annotated_dir, frame_number)
            evidence_image = str(evidence_path).replace("\\", "/")
            evidence_by_frame[frame_number] = evidence_image

        detected_objects = summarize_detected_objects(detections)
        for event in frame_events:
            events.append(
                {
                    "video_name": video_path.name,
                    "frame_number": frame_number,
                    "timestamp_sec": timestamp_sec,
                    "event_type": event["event_type"],
                    "risk_level": event["risk_level"],
                    "detected_objects": detected_objects,
                    "object_count": len(detections),
                    "brightness": brightness,
                    "evidence_image": evidence_image,
                    "tester_note": event["tester_note"],
                }
            )

        frame_number += 1

    capture.release()

    summary = build_summary(
        video_path.name,
        total_frames_analyzed,
        events,
        detected_object_counts=dict(sorted(detected_object_counts.items())),
        analysis_interval_sec=args.interval,
        model=args.model,
        confidence_threshold=args.confidence,
    )
    report_paths = write_reports(events, summary, output_dir)
    closed_loop_paths = _write_real_video_closed_loop_reports(
        video_path=video_path,
        output_dir=output_dir,
        frame_records=frame_records,
        evidence_by_frame=evidence_by_frame,
        fps=fps,
        ego_speed_kmh=args.ego_speed_kmh,
    )

    print("Analysis complete")
    print(f"Total frames analyzed: {total_frames_analyzed}")
    print(f"Total risk events: {len(events)}")
    print(f"CSV log saved to: {report_paths['csv']}")
    print(f"Excel report saved to: {report_paths['excel']}")
    print(f"Summary JSON saved to: {report_paths['summary']}")
    print(f"HTML report saved to: {report_paths['html']}")
    print(f"Closed-loop defect report saved to: {closed_loop_paths['defect_report']}")
    print(f"Closed-loop test cases saved to: {closed_loop_paths['test_cases_xlsx']}")
    return 0


def _write_real_video_closed_loop_reports(
    video_path: Path,
    output_dir: Path,
    frame_records: list[dict[str, Any]],
    evidence_by_frame: dict[int, str],
    fps: float,
    ego_speed_kmh: float,
) -> dict[str, Path]:
    """Run real-video detections through the ADAS closed-loop report path."""
    from src.defect_triage import DefectTriage
    from src.real_video_adapter import build_real_video_detection_data, link_real_video_evidence
    from src.report_generator import write_platform_reports
    from src.scenario_miner import ScenarioMiner
    from src.test_case_generator import TestCaseGenerator

    detection_data = build_real_video_detection_data(
        video_name=video_path.name,
        fps=fps,
        ego_speed_kmh=ego_speed_kmh,
        frame_records=frame_records,
    )
    miner = ScenarioMiner(fps=max(1, int(round(fps))), ego_speed_kmh=ego_speed_kmh)
    events = miner.mine(detection_data["frames"])
    link_real_video_evidence(events, evidence_by_frame)
    defects = DefectTriage().triage(events, source="real_video")
    test_cases = TestCaseGenerator().generate(events, defects, source="real_video")
    return write_platform_reports(
        events,
        defects,
        test_cases,
        output_dir / "adas_closed_loop",
        metadata={
            **detection_data["metadata"],
            "video_path": str(video_path).replace("\\", "/"),
            "input_frames": len(frame_records),
        },
    )


def run_video_render(args: argparse.Namespace, video_path: Path) -> int:
    """Render a full annotated video from YOLO detections."""
    try:
        from src.detector import RoadSceneDetector
        from src.risk_rules import evaluate_risks
        from src.utils import calculate_brightness
        from src.video_renderer import render_video_with_detector
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    output_dir = Path(args.output_dir)
    output_path = output_dir / "annotated_video.mp4"
    detector = RoadSceneDetector(model_path=args.model, confidence=args.confidence)
    try:
        result = render_video_with_detector(
            video_path=video_path,
            output_path=output_path,
            detector=detector,
            evaluate_risks=evaluate_risks,
            calculate_brightness=calculate_brightness,
            frame_stride=args.render_stride,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print("Annotated video render complete")
    print(f"Processed frames: {result['processed_frames']}")
    print(f"Risk overlay frames: {result['risk_frames']}")
    print(f"Annotated video saved to: {result['output_path']}")
    return 0


def run_platform_pipeline(args: argparse.Namespace) -> int:
    """Run the synthetic-detection ADAS scenario mining platform pipeline."""
    try:
        from src.defect_triage import DefectTriage
        from src.report_generator import write_platform_reports
        from src.scenario_miner import ScenarioMiner
        from src.synthetic_data import generate_synthetic_assets
        from src.test_case_generator import TestCaseGenerator
        from src.utils import ensure_dir
    except ImportError as exc:
        print(f"Error: missing dependency or module: {exc}")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    output_dir = ensure_dir(args.output_dir)
    detections_path: Path

    if args.generate_synthetic:
        assets = generate_synthetic_assets(Path("data/synthetic"), fps=args.fps, ego_speed_kmh=args.ego_speed_kmh)
        detections_path = assets["detections"]
        print(f"Synthetic video saved to: {assets['video']}")
        print(f"Synthetic detections saved to: {detections_path}")
    elif args.use_synthetic_detections:
        detections_path = Path("data/synthetic/synthetic_detections.json")
        if not detections_path.exists():
            print("Synthetic detections not found. Run `python src/main.py --generate-synthetic --output outputs` first.")
            return 1
    else:
        if not args.detections:
            print("Error: --detections is required when --skip-detection is used.")
            return 1
        detections_path = Path(args.detections)
        if not detections_path.exists():
            print(f"Error: detections file not found: {detections_path}")
            return 1

    data = json.loads(detections_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    fps = int(metadata.get("fps", args.fps))
    ego_speed = float(metadata.get("ego_speed_kmh", args.ego_speed_kmh))

    miner = ScenarioMiner(fps=fps, ego_speed_kmh=ego_speed)
    events = miner.mine(data.get("frames", []))
    if args.save_evidence:
        _write_evidence_frames(events, output_dir / "evidence_frames")

    defects = DefectTriage().triage(events)
    test_cases = TestCaseGenerator().generate(events, defects)
    report_paths = write_platform_reports(
        events,
        defects,
        test_cases,
        output_dir,
        metadata={
            "source": metadata.get("source", "detections_json"),
            "detections_path": str(detections_path).replace("\\", "/"),
            "fps": fps,
            "ego_speed_kmh": ego_speed,
        },
    )

    print("ADAS platform analysis complete")
    print(f"Total risk events: {len(events)}")
    print(f"Generated defects: {len(defects)}")
    print(f"Generated test cases: {len(test_cases)}")
    print(f"HTML report saved to: {report_paths['html']}")
    print(f"Defect report saved to: {report_paths['defect_report']}")
    print(f"Test cases saved to: {report_paths['test_cases_xlsx']}")
    return 0


def _write_evidence_frames(events: list[dict[str, Any]], evidence_dir: Path) -> None:
    """Create simple deterministic evidence images for mined synthetic events."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    evidence_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        path = evidence_dir / f"{event['event_id']}.jpg"
        image = np.full((360, 640, 3), (36, 42, 48), dtype=np.uint8)
        color = (40, 40, 230) if event.get("risk_level") in {"CRITICAL", "HIGH"} else (30, 160, 220)
        cv2.rectangle(image, (180, 120), (460, 300), color, 3)
        cv2.putText(image, str(event["event_id"]), (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
        cv2.putText(image, str(event["scenario_type"]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Risk: {event.get('risk_level')}", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
        cv2.imwrite(str(path), image)
        event["evidence_frame"] = str(path).replace("\\", "/")


if __name__ == "__main__":
    sys.exit(main())
