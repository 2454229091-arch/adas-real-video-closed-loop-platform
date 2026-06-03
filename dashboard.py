"""Local Streamlit dashboard for FSD Road Scene Test Logger outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
EVENT_COLUMNS = [
    "video_name",
    "frame_number",
    "timestamp_sec",
    "event_type",
    "risk_level",
    "detected_objects",
    "object_count",
    "brightness",
    "evidence_image",
    "tester_note",
]


def discover_runs(outputs_dir: Path = OUTPUTS_DIR) -> dict[str, Path]:
    """Find output folders that contain both summary and event log files."""
    runs: dict[str, Path] = {}
    if not outputs_dir.exists():
        return runs

    for summary_path in outputs_dir.rglob("summary.json"):
        run_dir = summary_path.parent
        if (run_dir / "event_log.csv").exists():
            runs[str(run_dir.relative_to(outputs_dir)).replace("\\", "/")] = run_dir

    if (outputs_dir / "summary.json").exists() and (outputs_dir / "event_log.csv").exists():
        runs["outputs"] = outputs_dir

    return dict(sorted(runs.items()))


@st.cache_data(show_spinner=False)
def load_summary(run_dir_text: str) -> dict[str, Any]:
    return json.loads((Path(run_dir_text) / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_events(run_dir_text: str) -> pd.DataFrame:
    csv_path = Path(run_dir_text) / "event_log.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.read_csv(csv_path)


def normalize_evidence_path(run_dir: Path, evidence_image: str) -> Path | None:
    if not evidence_image or evidence_image == "nan":
        return None

    evidence_path = Path(str(evidence_image))
    if evidence_path.is_absolute():
        return evidence_path

    if evidence_path.parts and evidence_path.parts[0] == "outputs":
        return ROOT / evidence_path

    return run_dir / evidence_path


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --road-ink: #101820;
          --road-muted: #6b7280;
          --road-panel: #f4f6f8;
          --road-line: #d8dee6;
          --road-high: #b42318;
          --road-medium: #b54708;
          --road-low: #175cd3;
        }
        .block-container {
          max-width: 1280px;
          padding-top: 2rem;
        }
        div[data-testid="stMetric"] {
          background: linear-gradient(180deg, #ffffff 0%, var(--road-panel) 100%);
          border: 1px solid var(--road-line);
          border-radius: 6px;
          padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] p {
          color: var(--road-muted);
          font-size: 0.86rem;
          letter-spacing: 0;
        }
        .risk-strip {
          border-left: 5px solid var(--road-high);
          background: #fff8f6;
          padding: 12px 14px;
          margin: 8px 0 18px;
        }
        .small-caption {
          color: var(--road-muted);
          font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(summary: dict[str, Any]) -> None:
    st.title("FSD Road Scene Test Dashboard")
    st.caption("Local ADAS/FSD road-scene test results. No upload, no database, offline report review.")
    video_label = summary.get("video_name") or summary.get("project_name", "ADAS Scenario Platform")
    st.markdown(
        f"""
        <div class="risk-strip">
          <strong>Run:</strong> {video_label}
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Model:</strong> {summary.get("model", "Unknown")}
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Generated:</strong> {summary.get("generated_at", "Unknown")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(summary: dict[str, Any]) -> None:
    risk_counts = summary.get("risk_level_counts", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Analyzed Frames", summary.get("analyzed_frames", summary.get("total_frames_analyzed", 0)))
    col2.metric("Risk Events", summary.get("total_events", summary.get("total_risk_events", 0)))
    col3.metric("Critical/High", risk_counts.get("CRITICAL", 0) + risk_counts.get("HIGH", risk_counts.get("High", 0)))
    col4.metric("Medium", risk_counts.get("MEDIUM", risk_counts.get("Medium", 0)))
    col5.metric("Low", risk_counts.get("LOW", risk_counts.get("Low", 0)))


def render_charts(summary: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Risk Level Distribution")
        risk_counts = pd.Series(summary.get("risk_level_counts", {}), dtype="int64")
        st.bar_chart(risk_counts)

    with right:
        st.subheader("Scenario / Event Counts")
        event_counts = pd.Series(
            summary.get("scenario_type_counts", summary.get("event_type_counts", {})),
            dtype="int64",
        ).sort_values(ascending=False)
        st.bar_chart(event_counts)

    object_counts = pd.Series(summary.get("detected_object_counts", {}), dtype="int64").sort_values(ascending=False)
    if not object_counts.empty:
        st.subheader("Detected Object Counts")
        st.bar_chart(object_counts)


def render_filters(events: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Risk Event Explorer")
    if events.empty:
        st.info("No risk events detected for this run.")
        return events

    risk_options = sorted(events["risk_level"].dropna().unique().tolist())
    event_column = "event_type" if "event_type" in events.columns else "scenario_type"
    event_options = sorted(events[event_column].dropna().unique().tolist())

    col1, col2, col3 = st.columns([1, 1.4, 1])
    selected_risks = col1.multiselect("Risk level", risk_options, default=risk_options)
    selected_events = col2.multiselect("Event type", event_options, default=event_options)
    max_rows = col3.number_input("Rows to show", min_value=10, max_value=500, value=100, step=10)

    filtered = events[
        events["risk_level"].isin(selected_risks)
        & events[event_column].isin(selected_events)
    ].copy()
    time_column = "timestamp_sec" if "timestamp_sec" in filtered.columns else "timestamp"
    filtered = filtered.sort_values([time_column, "risk_level", event_column]).head(int(max_rows))

    st.dataframe(
        filtered.drop(columns=["evidence_image"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )
    return filtered


def render_evidence(run_dir: Path, events: pd.DataFrame) -> None:
    st.subheader("Evidence Frames")
    if events.empty:
        st.caption("No evidence frames to show.")
        return

    evidence_column = "evidence_image" if "evidence_image" in events.columns else "evidence_frame"
    if evidence_column not in events.columns:
        st.caption("This run has event rows, but no evidence image column was recorded.")
        return

    sample = events.dropna(subset=[evidence_column]).head(12)
    if sample.empty:
        st.caption("This run has event rows, but no evidence images were recorded.")
        return

    columns = st.columns(3)
    for index, (_, row) in enumerate(sample.iterrows()):
        image_path = normalize_evidence_path(run_dir, str(row.get(evidence_column, "")))
        if not image_path or not image_path.exists():
            continue
        with columns[index % 3]:
            st.image(str(image_path), use_container_width=True)
            st.caption(
                f"{row.get('timestamp_sec', row.get('timestamp', '?'))}s | {row.get('risk_level', '?')} | {row.get('event_type', row.get('scenario_type', '?'))}"
            )


def render_annotated_video(run_dir: Path) -> None:
    video_path = run_dir / "annotated_video.mp4"
    if not video_path.exists():
        return
    st.subheader("Annotated Video")
    st.video(str(video_path))


def main() -> None:
    st.set_page_config(
        page_title="FSD Road Scene Test Dashboard",
        page_icon="",
        layout="wide",
    )
    inject_styles()

    runs = discover_runs()
    if not runs:
        st.title("FSD Road Scene Test Dashboard")
        st.warning("No analysis outputs found. Run `python src/main.py --video data/raw/sample_drive.mp4` first.")
        return

    with st.sidebar:
        st.header("Run Selector")
        selected_name = st.selectbox("Analysis run", list(runs.keys()))
        run_dir = runs[selected_name]
        st.markdown(f"<span class='small-caption'>{run_dir}</span>", unsafe_allow_html=True)
        st.divider()
        st.caption("This dashboard reads local files from the project outputs folder.")

    summary = load_summary(str(run_dir))
    events = load_events(str(run_dir))

    render_header(summary)
    render_metrics(summary)
    render_charts(summary)
    filtered = render_filters(events)
    render_annotated_video(run_dir)
    render_evidence(run_dir, filtered)


if __name__ == "__main__":
    main()
