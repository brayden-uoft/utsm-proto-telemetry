"""Auditable multi-run training data for transferable strategy models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .core import build_laps, derive_motion_energy, merge_by_time, read_gpx, read_telemetry
from .simulation import (
    build_strategy_segments_by_distance,
    evaluate_baseline_prediction,
    fit_empirical_energy_model,
)


@dataclass
class TrainingDataset:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    sources: list[dict[str, Any]]

    @property
    def run_ids(self) -> list[str]:
        return [str(source["run_id"]) for source in self.sources]

    @property
    def lap_count(self) -> int:
        return int(sum(int(source["loaded_laps"]) for source in self.sources))


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Training manifest must be a JSON object.")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Training manifest must contain a non-empty 'runs' list.")
    run_ids = [str(run.get("id", "")) for run in runs if isinstance(run, dict)]
    if len(run_ids) != len(runs) or any(not run_id for run_id in run_ids):
        raise ValueError("Every training run must have a non-empty 'id'.")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Training run IDs must be unique.")
    manifest["_path"] = str(manifest_path)
    return manifest


def _single_file(run_dir: Path, suffix: str) -> Path:
    files = sorted(path for path in run_dir.iterdir() if path.is_file() and path.name.lower().endswith(suffix))
    if len(files) != 1:
        raise ValueError(
            f"Training run '{run_dir.name}' must contain exactly one {suffix} file; found {len(files)}."
        )
    return files[0]


def _resolve_runs_dir(manifest: dict[str, Any]) -> Path:
    manifest_path = Path(str(manifest["_path"]))
    runs_dir = Path(str(manifest.get("runs_dir", "../runs")))
    if not runs_dir.is_absolute():
        runs_dir = manifest_path.parent / runs_dir
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Training runs directory not found: {runs_dir}")
    return runs_dir


def _calibration(manifest: dict[str, Any]) -> tuple[float, float]:
    calibration = manifest.get("current_calibration", {})
    slope = float(calibration.get("slope", 1.0))
    intercept_mA = float(calibration.get("intercept_mA", 0.0))
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept_mA):
        raise ValueError("Current calibration requires a positive finite slope and finite intercept_mA.")
    return slope, intercept_mA


def _load_run(
    manifest: dict[str, Any],
    run_spec: dict[str, Any],
    runs_dir: Path,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    run_id = str(run_spec["id"])
    run_dir = (runs_dir / run_id).resolve()
    if run_dir.parent != runs_dir or not run_dir.is_dir():
        raise FileNotFoundError(f"Training run folder not found: {run_dir}")
    gpx_path = _single_file(run_dir, ".gpx")
    telemetry_path = _single_file(run_dir, ".csv")
    run_manifest_path = run_dir / "run.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Training run manifest not found: {run_manifest_path}")
    with run_manifest_path.open(encoding="utf-8") as handle:
        run_manifest = json.load(handle)

    expected_laps = int(run_spec.get("laps", run_manifest.get("laps", 0)))
    split_method = str(run_spec.get("split_method", run_manifest.get("split_method", "start")))
    tolerance_sec = float(manifest.get("merge_tolerance_sec", 1.5))
    gps = read_gpx(str(gpx_path))
    telemetry = read_telemetry(str(telemetry_path))
    telemetry["uncalibrated_current_mA"] = telemetry["current_mA"]
    slope, intercept_mA = _calibration(manifest)
    telemetry["current_mA"] = (
        telemetry["uncalibrated_current_mA"] * slope + intercept_mA
    ).clip(lower=0.0)

    gps_laps, telemetry_laps, _ = build_laps(
        gps,
        telemetry,
        laps=expected_laps,
        split_method=split_method,
        start_time=None,
        time_offset_ms=0.0,
        tolerance_sec=tolerance_sec,
        lap_times=None,
    )
    rows: list[pd.DataFrame] = []
    for lap_index, (lap_gps, lap_telemetry) in enumerate(zip(gps_laps, telemetry_laps), start=1):
        if lap_gps.empty or lap_telemetry.empty:
            continue
        merged = merge_by_time(lap_telemetry, lap_gps, tolerance_sec)
        lap_start = lap_gps["time"].iloc[0]
        lap_end = lap_gps["time"].iloc[-1]
        merged = merged[(merged["time"] >= lap_start) & (merged["time"] <= lap_end)].copy()
        if merged.empty:
            continue
        derived = derive_motion_energy(merged)
        if derived.empty:
            continue
        derived["run_id"] = run_id
        derived["lap_id"] = lap_index
        lap_distance = max(float(derived["cumdist_m"].iloc[-1]), 1.0)
        derived["run_cumdist_m"] = derived["cumdist_m"]
        derived["model_position_frac"] = derived["cumdist_m"] / lap_distance
        rows.append(derived)

    if len(rows) != expected_laps:
        raise ValueError(
            f"Training run '{run_id}' loaded {len(rows)} complete laps; expected {expected_laps}."
        )

    source_current = pd.to_numeric(telemetry["source_current_mA"], errors="coerce")
    source = {
        "run_id": run_id,
        "label": str(run_manifest.get("label", run_id)),
        "gpx": str(gpx_path),
        "telemetry": str(telemetry_path),
        "split_method": split_method,
        "expected_laps": expected_laps,
        "loaded_laps": len(rows),
        "raw_telemetry_samples": int(len(telemetry)),
        "merged_training_samples": int(sum(len(frame) for frame in rows)),
        "source_negative_current_samples": int((source_current < 0).sum()),
        "source_over_20a_samples": int((source_current.abs() > 20_000).sum()),
    }
    return rows, source


def _apply_equal_lap_weights(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    group_sizes = frame.groupby(["run_id", "lap_id"])["lap_id"].transform("size")
    lap_count = int(frame[["run_id", "lap_id"]].drop_duplicates().shape[0])
    frame["sample_weight"] = len(frame) / (lap_count * group_sizes)
    return frame


def load_manifest_training_data(
    path: str | Path,
    *,
    include_run_ids: Iterable[str] | None = None,
) -> TrainingDataset:
    manifest = load_training_manifest(path)
    selected = set(include_run_ids) if include_run_ids is not None else None
    runs_dir = _resolve_runs_dir(manifest)
    lap_frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for run_spec in manifest["runs"]:
        run_id = str(run_spec["id"])
        if selected is not None and run_id not in selected:
            continue
        rows, source = _load_run(manifest, run_spec, runs_dir)
        lap_frames.extend(rows)
        sources.append(source)
    if not lap_frames:
        raise ValueError("No training laps were selected from the manifest.")
    frame = pd.concat(lap_frames, ignore_index=True)
    frame = _apply_equal_lap_weights(frame)
    return TrainingDataset(frame=frame, manifest=manifest, sources=sources)


def training_provenance(dataset: TrainingDataset) -> dict[str, Any]:
    slope, intercept_mA = _calibration(dataset.manifest)
    def portable(path_value: str) -> str:
        path = Path(path_value)
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    sources = []
    for source in dataset.sources:
        item = dict(source)
        item["gpx"] = portable(str(item["gpx"]))
        item["telemetry"] = portable(str(item["telemetry"]))
        sources.append(item)
    return {
        "name": str(dataset.manifest.get("name", "multi-run-training")),
        "manifest": portable(str(dataset.manifest["_path"])),
        "run_count": len(dataset.sources),
        "lap_count": dataset.lap_count,
        "training_sample_count": int(len(dataset.frame)),
        "current_calibration": {
            "slope": slope,
            "intercept_mA": intercept_mA,
            "note": str(dataset.manifest.get("current_calibration", {}).get("note", "")),
        },
        "sources": sources,
    }


def evaluate_model_by_run(
    frame: pd.DataFrame,
    model: dict[str, object],
    *,
    strategy_step_m: float = 50.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, run_frame in frame.groupby("run_id", sort=True):
        actual_energy_j = 0.0
        predicted_energy_j = 0.0
        current_errors: list[float] = []
        power_errors: list[float] = []
        lap_count = 0
        for _, lap_frame in run_frame.groupby("lap_id", sort=True):
            lap = lap_frame.copy().sort_values("time").reset_index(drop=True)
            lap["run_cumdist_m"] = lap["cumdist_m"]
            segments = build_strategy_segments_by_distance(lap, strategy_step_m)
            metrics = evaluate_baseline_prediction(segments, model)
            actual_energy_j += float(metrics["actual_energy_j"])
            predicted_energy_j += float(metrics["pred_energy_j"])
            current_errors.append(float(metrics["current_mae_mA"]))
            power_errors.append(float(metrics["power_mae_w"]))
            lap_count += 1
        energy_error_pct = (
            (predicted_energy_j - actual_energy_j) / actual_energy_j * 100.0
            if actual_energy_j > 0
            else 0.0
        )
        rows.append({
            "run_id": str(run_id),
            "lap_count": lap_count,
            "actual_energy_j": actual_energy_j,
            "predicted_energy_j": predicted_energy_j,
            "energy_error_pct": energy_error_pct,
            "absolute_energy_error_pct": abs(energy_error_pct),
            "current_mae_mA": sum(current_errors) / max(len(current_errors), 1),
            "power_mae_w": sum(power_errors) / max(len(power_errors), 1),
        })
    return pd.DataFrame(rows)


def leave_one_run_out_validation(
    dataset: TrainingDataset,
    *,
    baseline_model: dict[str, object] | None = None,
    strategy_step_m: float = 50.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for held_run_id in dataset.run_ids:
        train = dataset.frame[dataset.frame["run_id"] != held_run_id].copy()
        train = _apply_equal_lap_weights(train)
        held = dataset.frame[dataset.frame["run_id"] == held_run_id].copy()
        model = fit_empirical_energy_model(
            train,
            sample_weight_column="sample_weight",
            transferable=True,
        )
        metrics = evaluate_model_by_run(held, model, strategy_step_m=strategy_step_m).iloc[0].to_dict()
        row = {"held_out_run_id": held_run_id, **{f"front_campus_{key}": value for key, value in metrics.items() if key != "run_id"}}
        if baseline_model is not None:
            baseline = evaluate_model_by_run(
                held, baseline_model, strategy_step_m=strategy_step_m
            ).iloc[0].to_dict()
            row.update({f"indy_{key}": value for key, value in baseline.items() if key != "run_id"})
            row["absolute_energy_error_improvement_pct_points"] = (
                float(baseline["absolute_energy_error_pct"])
                - float(metrics["absolute_energy_error_pct"])
            )
        rows.append(row)
    return pd.DataFrame(rows)
