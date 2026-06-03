import json

from dashboard import discover_runs


def test_discover_runs_finds_nested_closed_loop_outputs(tmp_path):
    run_dir = tmp_path / "test111" / "adas_closed_loop"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"total_risk_events": 1}), encoding="utf-8")
    (run_dir / "event_log.csv").write_text("event_id,scenario_type\nEVT-000001,pedestrian_crossing\n", encoding="utf-8")

    runs = discover_runs(tmp_path)

    assert runs["test111/adas_closed_loop"] == run_dir
