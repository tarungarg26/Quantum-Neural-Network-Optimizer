"""Data loading and tidying engine (M15).

Loads runs from the `runs/` filesystem into three canonical DataFrames:
- `runs`: One row per run (key metadata, final metrics, status, failure reasons)
- `trace`: Optimization trajectory (per-update and checkpoint metrics)
- `events`: Granular ledger accounting events
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TidyBundle:
    """Bundle of three tidy DataFrames."""

    runs: pd.DataFrame
    trace: pd.DataFrame
    events: pd.DataFrame

    def filter_main_track(self) -> TidyBundle:
        """Filter out historical or noise tracks."""
        if "track" in self.runs.columns:
            main_runs = self.runs[self.runs["track"] == "main"].copy()
            valid_ids = set(main_runs["run_id"])
            main_trace = self.trace[self.trace["run_id"].isin(valid_ids)].copy()
            main_events = self.events[self.events["run_id"].isin(valid_ids)].copy()
            return TidyBundle(runs=main_runs, trace=main_trace, events=main_events)
        return self


def load_runs_dir(runs_dir: str | Path) -> TidyBundle:
    """Scan and parse all completed and failed runs from runs_dir."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        empty_df = pd.DataFrame()
        return TidyBundle(runs=empty_df, trace=empty_df, events=empty_df)

    run_records: list[dict] = []
    trace_frames: list[pd.DataFrame] = []
    event_records: list[dict] = []

    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        final_file = run_dir / "final.json"
        trace_file = run_dir / "trace.csv"
        events_file = run_dir / "events.jsonl"

        # 1. Parse final.json if present
        if final_file.exists():
            try:
                with open(final_file) as f:
                    final_data = json.load(f)
                final_data["run_id"] = run_id
                run_records.append(final_data)
            except Exception:
                run_records.append({"run_id": run_id, "status": "corrupt_final"})

        # 2. Parse trace.csv if present
        if trace_file.exists():
            try:
                tdf = pd.read_csv(trace_file)
                tdf["run_id"] = run_id
                trace_frames.append(tdf)
            except Exception:
                pass

        # 3. Parse events.jsonl if present
        if events_file.exists():
            try:
                with open(events_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            event_records.append(json.loads(line))
            except Exception:
                pass

    runs_df = pd.DataFrame(run_records) if run_records else pd.DataFrame(columns=["run_id", "status"])
    trace_df = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame(columns=["run_id", "update"])
    events_df = pd.DataFrame(event_records) if event_records else pd.DataFrame(columns=["run_id", "category"])

    return TidyBundle(runs=runs_df, trace=trace_df, events=events_df)
