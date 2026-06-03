# Demo Walkthrough

This guide shows how to demonstrate the project locally.

## 1. Install Dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Run The Synthetic Regression Demo

```powershell
python src/main.py --generate-synthetic --output outputs
python src/main.py --use-synthetic-detections --output outputs
```

Open these files after the run:

- `outputs/summary.json`
- `outputs/defect_report.md`
- `outputs/test_cases.csv`
- `outputs/test_report.html`

This is the best demo for showing the full ADAS workflow because it produces risk events, defects, regression test cases, charts, and evidence frames.

## 3. Analyze A Real Video

Put a local MP4 file under `data/raw/`, then run:

```powershell
python src/main.py --video data/raw/test111.mp4 --output outputs/test111
```

Open:

- `outputs/test111/event_log.csv`
- `outputs/test111/risk_events.xlsx`
- `outputs/test111/summary.json`
- `outputs/test111/test_report.html`
- `outputs/test111/annotated_frames/`
- `outputs/test111/adas_closed_loop/defect_report.md`
- `outputs/test111/adas_closed_loop/test_cases.csv`
- `outputs/test111/adas_closed_loop/test_report.html`

## 4. Render An Annotated Video

```powershell
python src/main.py --video data/raw/test111.mp4 --render-video --render-stride 1 --output outputs/test111
```

For a faster but less dense preview:

```powershell
python src/main.py --video data/raw/test111.mp4 --render-video --render-stride 3 --output outputs/test111_fast
```

Open:

- `outputs/test111/annotated_video.mp4`

## 5. Open The Dashboard

```powershell
streamlit run dashboard.py
```

In the browser, select a run under `outputs`. The dashboard shows metrics, charts, event tables, evidence frames, and an annotated video preview when available. Real-video closed-loop reports appear as nested runs such as `test111/adas_closed_loop`.

## 6. Run Verification

```powershell
python -m pytest tests
python -m compileall src tests
python -m py_compile dashboard.py
```

## Demo Narrative

Use this short explanation:

> The project starts from road-scene observations, turns them into structured ADAS risk events, and then produces artifacts that a test engineer can review: evidence frames, reports, defect records, and regression test cases. The real-video path now demonstrates a first-version closed loop from YOLO detections to ADAS defects and regression tests. The synthetic path remains the deterministic regression demo.
