"""Build a self-contained multi-run telemetry strategy dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install pandas numpy"
    ) from exc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_strategy import build_laps
from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    add_xy,
    build_full_run_distance,
    build_motor_config,
    build_strategy_report,
    build_strategy_samples,
    build_strategy_segments,
    build_strategy_segments_by_distance,
    compute_accel_candidate_scores,
    derive_motion_energy,
    evaluate_baseline_prediction,
    fit_empirical_energy_model,
    merge_by_time,
    optimize_speed_profile,
    read_gpx,
    read_telemetry,
)

DEFAULT_RUNS = [
    {
        "id": "morning",
        "label": "Morning run",
        "gps": "Utsm.gpx",
        "telemetry": os.path.join("telemetry_dumps", "telemetry_20260411_112302.csv"),
    },
    {
        "id": "afternoon",
        "label": "Afternoon run",
        "gps": "Utsm-2.gpx",
        "telemetry": os.path.join("telemetry_dumps", "telemetry_20260411_122713.csv"),
    },
]
DEFAULT_OUTPUT = os.path.join("outputs", "telemetry_strategy_dashboard.html")

METRICS = {
    "speed": {"label": "Speed", "unit": "km/h", "field": "speed", "color": "#1f77b4"},
    "targetSpeed": {
        "label": "Strategy target speed",
        "unit": "km/h",
        "field": "targetSpeed",
        "color": "#f97316",
    },
    "current": {"label": "Current", "unit": "mA", "field": "current", "color": "#d62728"},
    "gpsAccel": {
        "label": "GPS accel magnitude",
        "unit": "m/s^2",
        "field": "gpsAccel",
        "color": "#2ca02c",
    },
    "imuAccel": {
        "label": "MPU dynamic acceleration",
        "unit": "m/s^2",
        "field": "imuAccel",
        "color": "#16a34a",
    },
    "power": {"label": "Power", "unit": "W", "field": "power", "color": "#9467bd"},
    "runEnergyJ": {
        "label": "Total energy",
        "unit": "J",
        "field": "runEnergyJ",
        "color": "#b45309",
        "map_selectable": False,
    },
}

STRATEGY_LEGEND = {
    "accelerate": {"label": "Accelerate", "color": "#f97316"},
    "hold": {"label": "Hold", "color": "#2563eb"},
    "coast": {"label": "Coast", "color": "#16a34a"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-run interactive telemetry strategy dashboard."
    )
    parser.add_argument("--gps")
    parser.add_argument("--telemetry")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument(
        "--split-method",
        choices=["points", "time", "line", "start"],
        default="start",
    )
    parser.add_argument("--lap-times", nargs="+", metavar="ELAPSED")
    parser.add_argument("--start-time")
    parser.add_argument("--time-offset-ms", type=float, default=0.0)
    parser.add_argument("--tolerance-sec", type=float, default=1.5)
    parser.add_argument("--forward-axis", choices=FORWARD_AXIS_CHOICES, default="ax")
    parser.add_argument("--accel-window", type=int, default=5)
    parser.add_argument("--accel-scale", type=float, default=1000.0)
    parser.add_argument("--imu-axis", choices=["ax", "ay", "az"], default="ax")
    parser.add_argument("--imu-axis-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--accel-bias-window-sec", type=float, default=30.0)
    parser.add_argument("--accel-smooth-window-sec", type=float, default=8.0)
    parser.add_argument("--strategy-segments", type=int, default=None, help="Legacy fixed segment count. Overrides --strategy-step-m when set.")
    parser.add_argument("--strategy-step-m", type=float, default=50.0)
    parser.add_argument("--display-region-min-m", type=float, default=150.0)
    parser.add_argument("--strategy-speed-min-kph", type=float, default=8.0)
    parser.add_argument("--strategy-speed-max-kph", type=float, default=40.0)
    parser.add_argument("--strategy-max-delta-kph-per-segment", type=float, default=6.0)
    parser.add_argument("--strategy-speed-step-kph", type=float, default=1.0)
    parser.add_argument("--strategy-hold-delta-kph", type=float, default=1.0)
    parser.add_argument("--strategy-time-budget-sec", type=float, default=2100.0)
    parser.add_argument("--strategy-time-tolerance-pct", type=float, default=3.0)
    parser.add_argument("--fuse-current-ma", type=float, default=20000.0)
    parser.add_argument("--fuse-max-duration-sec", type=float, default=1.0)
    parser.add_argument("--current-penalty-weight", type=float, default=5.0)
    parser.add_argument("--wheel-diameter-m", type=float, default=0.50)
    parser.add_argument("--vehicle-mass-kg", type=float, default=100.0)
    parser.add_argument("--rolling-resistance-coeff", type=float, default=0.008)
    parser.add_argument("--drivetrain-efficiency", type=float, default=0.82)
    parser.add_argument("--strategy-start-speed-kph", type=float, default=0.0)
    return parser.parse_args()


def resolve_run_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.gps and args.telemetry:
        return [{
            "id": "custom",
            "label": "Custom run",
            "gps": args.gps,
            "telemetry": args.telemetry,
        }]
    if args.gps or args.telemetry:
        raise ValueError("--gps and --telemetry must be provided together.")
    return list(DEFAULT_RUNS)


def load_single_run(spec: dict[str, str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    gps_df = read_gpx(spec["gps"])
    telem_df = read_telemetry(spec["telemetry"])
    gps_laps, telem_laps, _ = build_laps(gps_df, telem_df, args)

    rows = []
    for lap_num, (lap_gps, lap_telem) in enumerate(zip(gps_laps, telem_laps), start=1):
        if lap_gps.empty or lap_telem.empty:
            fallback = lap_gps.copy()
            fallback["current_mA"] = 0.0
            fallback["voltage_mV"] = 0.0
            fallback["ax_x100"] = np.nan
            fallback["ay_x100"] = np.nan
            fallback["az_x100"] = np.nan
            fallback["gps_time"] = pd.to_datetime(fallback["time"])
            derived = derive_motion_energy(
                fallback,
                forward_axis=args.forward_axis,
                accel_window=args.accel_window,
                accel_scale=args.accel_scale,
                imu_axis=args.imu_axis,
                imu_axis_sign=args.imu_axis_sign,
                accel_bias_window_sec=args.accel_bias_window_sec,
                accel_smooth_window_sec=args.accel_smooth_window_sec,
            )
            derived["lap"] = lap_num
            derived["telemetry_available"] = False
            rows.append(derived)
            continue
        try:
            merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        except ValueError as exc:
            print(f"{spec['label']} lap {lap_num}: telemetry merge failed, using GPS-only fallback - {exc}")
            fallback = lap_gps.copy()
            fallback["current_mA"] = 0.0
            fallback["voltage_mV"] = 0.0
            fallback["ax_x100"] = np.nan
            fallback["ay_x100"] = np.nan
            fallback["az_x100"] = np.nan
            fallback["gps_time"] = pd.to_datetime(fallback["time"])
            derived = derive_motion_energy(
                fallback,
                forward_axis=args.forward_axis,
                accel_window=args.accel_window,
                accel_scale=args.accel_scale,
                imu_axis=args.imu_axis,
                imu_axis_sign=args.imu_axis_sign,
                accel_bias_window_sec=args.accel_bias_window_sec,
                accel_smooth_window_sec=args.accel_smooth_window_sec,
            )
            derived["lap"] = lap_num
            derived["telemetry_available"] = False
            rows.append(derived)
            continue
        lap_start = lap_gps["time"].iloc[0]
        lap_end = lap_gps["time"].iloc[-1]
        merged = merged[(merged["time"] >= lap_start) & (merged["time"] <= lap_end)].copy()
        if merged.empty:
            fallback = lap_gps.copy()
            fallback["current_mA"] = 0.0
            fallback["voltage_mV"] = 0.0
            fallback["ax_x100"] = np.nan
            fallback["ay_x100"] = np.nan
            fallback["az_x100"] = np.nan
            fallback["gps_time"] = pd.to_datetime(fallback["time"])
            derived = derive_motion_energy(
                fallback,
                forward_axis=args.forward_axis,
                accel_window=args.accel_window,
                accel_scale=args.accel_scale,
                imu_axis=args.imu_axis,
                imu_axis_sign=args.imu_axis_sign,
                accel_bias_window_sec=args.accel_bias_window_sec,
                accel_smooth_window_sec=args.accel_smooth_window_sec,
            )
            derived["lap"] = lap_num
            derived["telemetry_available"] = False
            rows.append(derived)
            continue
        derived = derive_motion_energy(
            merged,
            forward_axis=args.forward_axis,
            accel_window=args.accel_window,
            accel_scale=args.accel_scale,
            imu_axis=args.imu_axis,
            imu_axis_sign=args.imu_axis_sign,
            accel_bias_window_sec=args.accel_bias_window_sec,
            accel_smooth_window_sec=args.accel_smooth_window_sec,
        )
        derived["lap"] = lap_num
        derived["telemetry_available"] = True
        rows.append(derived)

    if not rows:
        raise ValueError(f"No lap data could be merged for {spec['label']}.")

    df = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    df = add_xy(df)
    start_time = pd.to_datetime(df["time"].iloc[0])
    df["elapsed_s"] = (pd.to_datetime(df["time"]) - start_time).dt.total_seconds()
    df = build_full_run_distance(df)

    train_df = df[df["telemetry_available"]].copy()
    strategy_model = fit_empirical_energy_model(train_df if not train_df.empty else df)
    if args.strategy_segments is not None:
        strategy_segments = build_strategy_segments(df, args.strategy_segments)
    else:
        strategy_segments = build_strategy_segments_by_distance(df, args.strategy_step_m)
    motor_config = build_motor_config(
        wheel_diameter_m=args.wheel_diameter_m,
        vehicle_mass_kg=args.vehicle_mass_kg,
        rolling_resistance_coeff=args.rolling_resistance_coeff,
        drivetrain_efficiency=args.drivetrain_efficiency,
    )
    calibration = evaluate_baseline_prediction(
        strategy_segments,
        strategy_model,
        motor_config=motor_config,
        hold_delta_kph=args.strategy_hold_delta_kph,
        start_speed_kph=args.strategy_start_speed_kph,
    )
    baseline_time_s = float(pd.to_numeric(df["dt_s"], errors="coerce").fillna(0.0).sum())
    time_budget_s = min(
        args.strategy_time_budget_sec,
        baseline_time_s * (1.0 + args.strategy_time_tolerance_pct / 100.0),
    )
    min_time_s = baseline_time_s * (1.0 - args.strategy_time_tolerance_pct / 100.0)
    strategy_profile = optimize_speed_profile(
        strategy_segments,
        strategy_model,
        time_budget_sec=time_budget_s,
        speed_min_kph=args.strategy_speed_min_kph,
        speed_max_kph=args.strategy_speed_max_kph,
        max_delta_kph_per_segment=args.strategy_max_delta_kph_per_segment,
        speed_step_kph=args.strategy_speed_step_kph,
        hold_delta_kph=args.strategy_hold_delta_kph,
        fuse_current_ma=args.fuse_current_ma,
        fuse_max_duration_sec=args.fuse_max_duration_sec,
        current_penalty_weight=args.current_penalty_weight,
        motor_config=motor_config,
        start_speed_kph=args.strategy_start_speed_kph,
        min_time_sec=min_time_s,
    )
    strategy_report = build_strategy_report(
        df,
        strategy_profile,
        time_budget_s,
        calibration=calibration,
    )
    aligned = build_strategy_samples(df, strategy_profile).reset_index(drop=True)
    df = df.reset_index(drop=True)
    df["segment"] = aligned["segment"]
    df["target_speed_kph"] = aligned["target_speed_kph"]
    df["strategy_action"] = aligned["strategy_action"]
    df["pred_current_mA"] = aligned["pred_current_mA"]
    df["pred_avg_current_mA"] = aligned["pred_avg_current_mA"]
    df["pred_peak_current_mA"] = aligned["pred_peak_current_mA"]
    df["pred_on_current_mA"] = aligned["pred_on_current_mA"]
    df["throttle_duty"] = aligned["throttle_duty"]
    df["throttle_level"] = aligned["throttle_level"]
    df["pred_power_w"] = aligned["pred_power_w"]
    df["pulse_duration_s"] = aligned["pulse_duration_s"]
    df["coast_duration_s"] = aligned["coast_duration_s"]
    df["coast_speed_drop_kph"] = aligned["coast_speed_drop_kph"]
    df["pred_energy_j"] = aligned["pred_energy_j"]
    df["pred_cum_energy_j"] = aligned["pred_cum_energy_j"]
    df["pred_over_fuse_limit"] = aligned["pred_over_fuse_limit"]
    return df, strategy_profile, strategy_report


def finite_float(value: Any, digits: int = 3) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return round(result, digits)


def domain(
    values: pd.Series,
    pad_fraction: float = 0.06,
    robust: bool = False,
    min_zero: bool = False,
) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return [0.0, 1.0]
    if robust and len(finite) >= 20:
        lo = float(finite.quantile(0.02))
        hi = float(finite.quantile(0.98))
    else:
        lo = float(finite.min())
        hi = float(finite.max())
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        lower = 0.0 if min_zero else lo - pad
        return [round(lower, 3), round(hi + pad, 3)]
    pad = (hi - lo) * pad_fraction
    lower = 0.0 if min_zero else lo - pad
    return [round(lower, 3), round(hi + pad, 3)]


def build_display_regions(
    df: pd.DataFrame,
    strategy_profile: pd.DataFrame,
    min_region_m: float,
) -> list[dict[str, Any]]:
    df = df.copy()
    if "run_cumdist_m" not in df.columns:
        if "dist_m" in df.columns:
            df = build_full_run_distance(df)
        elif {"x", "y"}.issubset(df.columns):
            dx = pd.to_numeric(df["x"], errors="coerce").diff().fillna(0.0)
            dy = pd.to_numeric(df["y"], errors="coerce").diff().fillna(0.0)
            df["run_cumdist_m"] = np.sqrt(dx * dx + dy * dy).cumsum()
        else:
            df["run_cumdist_m"] = np.arange(len(df), dtype=float)
    min_region_m = max(float(min_region_m), 1.0)
    regions = []
    current_rows = []
    current_action = None
    current_length = 0.0

    for row in strategy_profile.itertuples(index=False):
        action = str(row.action)
        length = float(getattr(row, "length_m", float(row.dist_end_m) - float(row.dist_start_m)))
        if current_rows and action != current_action and current_length >= min_region_m:
            regions.append(_make_display_region(df, current_rows, len(regions) + 1))
            current_rows = []
            current_length = 0.0
        current_rows.append(row)
        current_action = action
        current_length += length
    if current_rows:
        regions.append(_make_display_region(df, current_rows, len(regions) + 1))
    return regions


def _make_display_region(df: pd.DataFrame, rows: list[Any], display_segment: int) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    dist_start = float(first.dist_start_m)
    dist_end = float(last.dist_end_m)
    seg_rows = df[(df["run_cumdist_m"] >= dist_start) & (df["run_cumdist_m"] <= dist_end)]
    mid = seg_rows.iloc[len(seg_rows) // 2] if not seg_rows.empty else None
    label_dx_m = None
    label_dy_m = None
    if len(seg_rows) >= 3:
        mid_idx = len(seg_rows) // 2
        before = seg_rows.iloc[max(mid_idx - 2, 0)]
        after = seg_rows.iloc[min(mid_idx + 2, len(seg_rows) - 1)]
        tx = float(after["x"] - before["x"])
        ty = float(after["y"] - before["y"])
        norm = math.hypot(tx, ty)
        if norm > 0:
            side = -1.0 if display_segment % 2 else 1.0
            label_dx_m = side * (-ty / norm) * 28.0
            label_dy_m = side * (tx / norm) * 28.0
    length_m = sum(float(getattr(row, "length_m", float(row.dist_end_m) - float(row.dist_start_m))) for row in rows)
    action_counts = pd.Series([str(row.action) for row in rows]).value_counts()
    action = str(action_counts.index[0])
    return {
        "segment": int(display_segment),
        "internal_start_segment": int(first.segment),
        "internal_end_segment": int(last.segment),
        "lap": int(mid["lap"]) if mid is not None else None,
        "label_x": finite_float(mid["x"], 2) if mid is not None else None,
        "label_y": finite_float(mid["y"], 2) if mid is not None else None,
        "label_dx_m": finite_float(label_dx_m, 2),
        "label_dy_m": finite_float(label_dy_m, 2),
        "dist_start_m": finite_float(dist_start, 2),
        "dist_end_m": finite_float(dist_end, 2),
        "length_m": finite_float(length_m, 2),
        "target_speed_kph": finite_float(float(last.target_speed_kph), 2),
        "entry_speed_kph": finite_float(float(first.entry_speed_kph), 2),
        "speed_delta_kph": finite_float(float(last.target_speed_kph) - float(first.entry_speed_kph), 2),
        "pred_current_mA": finite_float(np.mean([float(row.pred_current_mA) for row in rows]), 0),
        "pred_peak_current_mA": finite_float(max(float(row.pred_peak_current_mA) for row in rows), 0),
        "pred_on_current_mA": finite_float(max(float(row.pred_on_current_mA) for row in rows), 0),
        "throttle_duty": finite_float(np.mean([float(row.throttle_duty) for row in rows]), 3),
        "throttle_level": finite_float(np.mean([float(row.throttle_level) for row in rows]), 3),
        "pulse_duration_s": finite_float(sum(float(getattr(row, "pulse_duration_s", 0.0)) for row in rows), 2),
        "coast_duration_s": finite_float(sum(float(getattr(row, "coast_duration_s", 0.0)) for row in rows), 2),
        "pred_power_w": finite_float(np.mean([float(row.pred_power_w) for row in rows]), 2),
        "pred_energy_j": finite_float(sum(float(row.pred_energy_j) for row in rows), 2),
        "action": action,
        "over_fuse_limit": bool(any(bool(row.over_fuse_limit) for row in rows)),
    }


def make_run_payload(
    spec: dict[str, str],
    df: pd.DataFrame,
    strategy_profile: pd.DataFrame,
    strategy_report: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    df = df.copy()
    if "pred_avg_current_mA" not in df.columns:
        df["pred_avg_current_mA"] = df["pred_current_mA"]
    if "pred_peak_current_mA" not in df.columns:
        df["pred_peak_current_mA"] = df["pred_current_mA"]
    if "pred_on_current_mA" not in df.columns:
        df["pred_on_current_mA"] = df["pred_current_mA"]
    if "throttle_duty" not in df.columns:
        df["throttle_duty"] = 0.0
    if "throttle_level" not in df.columns:
        df["throttle_level"] = df["throttle_duty"]
    if "pulse_duration_s" not in df.columns:
        df["pulse_duration_s"] = 0.0
    if "coast_duration_s" not in df.columns:
        df["coast_duration_s"] = 0.0
    if "coast_speed_drop_kph" not in df.columns:
        df["coast_speed_drop_kph"] = 0.0
    strategy_profile = strategy_profile.copy()
    for col, fallback in (
        ("pred_avg_current_mA", "pred_current_mA"),
        ("pred_peak_current_mA", "pred_current_mA"),
        ("pred_on_current_mA", "pred_current_mA"),
    ):
        if col not in strategy_profile.columns:
            strategy_profile[col] = strategy_profile[fallback]
    if "throttle_duty" not in strategy_profile.columns:
        strategy_profile["throttle_duty"] = 0.0
    if "throttle_level" not in strategy_profile.columns:
        strategy_profile["throttle_level"] = strategy_profile["throttle_duty"]
    if "pulse_duration_s" not in strategy_profile.columns:
        strategy_profile["pulse_duration_s"] = 0.0
    if "coast_duration_s" not in strategy_profile.columns:
        strategy_profile["coast_duration_s"] = 0.0
    if "coast_speed_drop_kph" not in strategy_profile.columns:
        strategy_profile["coast_speed_drop_kph"] = 0.0
    if "length_m" not in strategy_profile.columns:
        strategy_profile["length_m"] = (
            pd.to_numeric(strategy_profile["dist_end_m"], errors="coerce")
            - pd.to_numeric(strategy_profile["dist_start_m"], errors="coerce")
        ).fillna(0.0)
    samples = []
    for row in df.itertuples(index=False):
        samples.append({
            "t": finite_float(row.elapsed_s, 2),
            "lap": int(row.lap),
            "x": finite_float(row.x, 2),
            "y": finite_float(row.y, 2),
            "segment": int(row.segment),
            "current": finite_float(row.current_mA, 0),
            "predCurrent": finite_float(row.pred_current_mA, 0),
            "predPeakCurrent": finite_float(row.pred_peak_current_mA, 0),
            "predOnCurrent": finite_float(row.pred_on_current_mA, 0),
            "throttleDuty": finite_float(row.throttle_duty, 3),
            "throttleLevel": finite_float(row.throttle_level, 3),
            "speed": finite_float(row.speed_kph, 2),
            "targetSpeed": finite_float(row.target_speed_kph, 2),
            "gpsAccel": finite_float(row.gps_longitudinal_accel_abs_m_s2, 3),
            "imuAccel": finite_float(row.imu_forward_dynamic_m_s2, 3),
            "power": finite_float(row.power_w, 2),
            "predPower": finite_float(row.pred_power_w, 2),
            "pulseDuration": finite_float(row.pulse_duration_s, 2),
            "coastDuration": finite_float(row.coast_duration_s, 2),
            "coastSpeedDrop": finite_float(row.coast_speed_drop_kph, 2),
            "runEnergyJ": finite_float(row.cum_energy_j, 2),
            "predRunEnergyJ": finite_float(row.pred_cum_energy_j, 2),
            "strategyAction": str(row.strategy_action),
            "predOverFuse": bool(row.pred_over_fuse_limit),
        })

    metric_domains = {}
    metric_series = {
        "speed": pd.concat(
            [pd.to_numeric(df["speed_kph"], errors="coerce"), pd.to_numeric(df["target_speed_kph"], errors="coerce")],
            ignore_index=True,
        ),
        "targetSpeed": pd.to_numeric(df["target_speed_kph"], errors="coerce"),
        "current": pd.concat(
            [
                pd.to_numeric(df["current_mA"], errors="coerce"),
                pd.to_numeric(df["pred_current_mA"], errors="coerce"),
                pd.to_numeric(df["pred_peak_current_mA"], errors="coerce"),
            ],
            ignore_index=True,
        ),
        "gpsAccel": pd.to_numeric(df["gps_longitudinal_accel_abs_m_s2"], errors="coerce"),
        "imuAccel": pd.to_numeric(df["imu_forward_dynamic_m_s2"], errors="coerce"),
        "power": pd.concat(
            [pd.to_numeric(df["power_w"], errors="coerce"), pd.to_numeric(df["pred_power_w"], errors="coerce")],
            ignore_index=True,
        ),
        "runEnergyJ": pd.concat(
            [pd.to_numeric(df["cum_energy_j"], errors="coerce"), pd.to_numeric(df["pred_cum_energy_j"], errors="coerce")],
            ignore_index=True,
        ),
    }
    for key, values in metric_series.items():
        metric_domains[key] = domain(
            values,
            robust=key in ("gpsAccel", "imuAccel"),
            min_zero=key in ("gpsAccel", "runEnergyJ", "targetSpeed"),
        )

    laps = []
    for lap_num, group in df.groupby("lap"):
        laps.append({
            "lap": int(lap_num),
            "start": int(group.index.min()),
            "end": int(group.index.max()),
            "start_t": finite_float(group["elapsed_s"].iloc[0], 2),
            "end_t": finite_float(group["elapsed_s"].iloc[-1], 2),
        })

    segments = build_display_regions(
        df,
        strategy_profile,
        getattr(args, "display_region_min_m", 150.0),
    )

    total_over = float(pd.to_numeric(strategy_profile["fuse_over_duration_s"], errors="coerce").fillna(0.0).sum())
    longest_over = longest_true_duration(
        strategy_profile["over_fuse_limit"].tolist(),
        strategy_profile["segment_time_s"].tolist(),
    )
    baseline_energy = float(pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0).sum())
    predicted_energy = float(pd.to_numeric(strategy_profile["pred_energy_j"], errors="coerce").fillna(0.0).sum())
    baseline_time = float(pd.to_numeric(df["dt_s"], errors="coerce").fillna(0.0).sum())
    predicted_time = float(pd.to_numeric(strategy_profile["segment_time_s"], errors="coerce").fillna(0.0).sum())
    total_coast_time = float(pd.to_numeric(strategy_profile["coast_duration_s"], errors="coerce").fillna(0.0).sum())
    total_coast_dist = float(strategy_profile.loc[strategy_profile["coast_duration_s"] > 0, "length_m"].sum())
    pulse_count = int((pd.to_numeric(strategy_profile["pulse_duration_s"], errors="coerce").fillna(0.0) > 0).sum())
    avg_pulse_duration = float(
        pd.to_numeric(strategy_profile.loc[strategy_profile["pulse_duration_s"] > 0, "pulse_duration_s"], errors="coerce")
        .dropna()
        .mean()
    )
    if not math.isfinite(avg_pulse_duration):
        avg_pulse_duration = 0.0
    motor_config = strategy_profile.attrs.get("motor_config", {}) or build_motor_config(
        getattr(args, "wheel_diameter_m", 0.50)
    )
    accel_diagnostics = []
    for row in compute_accel_candidate_scores(df):
        accel_diagnostics.append({
            "candidate": row["candidate"],
            "gps_corr": finite_float(row["gps_corr"], 4),
            "current_corr": finite_float(row["current_corr"], 4),
            "power_corr": finite_float(row["power_corr"], 4),
        })

    return {
        "id": spec["id"],
        "label": spec["label"],
        "meta": {
            "gps": spec["gps"],
            "telemetry": spec["telemetry"],
            "sample_count": len(samples),
            "duration_s": finite_float(df["elapsed_s"].iloc[-1], 2),
            "forward_axis": args.forward_axis,
            "accel": {
                "scale": args.accel_scale,
                "imu_axis": args.imu_axis,
                "imu_axis_sign": args.imu_axis_sign,
                "bias_window_s": args.accel_bias_window_sec,
                "smooth_window_s": args.accel_smooth_window_sec,
            },
        },
        "samples": samples,
        "laps": laps,
        "segments": segments,
        "accel_diagnostics": accel_diagnostics,
        "domains": {
            "x": domain(df["x"]),
            "y": domain(df["y"]),
            "metrics": metric_domains,
        },
        "strategy": {
            "time_budget_s": finite_float(strategy_profile.attrs.get("time_budget_sec", args.strategy_time_budget_sec), 2),
            "min_time_s": finite_float(strategy_profile.attrs.get("min_time_sec"), 2),
            "baseline_energy_j": finite_float(baseline_energy, 2),
            "predicted_energy_j": finite_float(predicted_energy, 2),
            "delta_energy_j": finite_float(predicted_energy - baseline_energy, 2),
            "delta_energy_pct": finite_float(
                ((predicted_energy - baseline_energy) / baseline_energy * 100.0) if baseline_energy > 0 else 0.0,
                2,
            ),
            "baseline_time_s": finite_float(baseline_time, 2),
            "predicted_time_s": finite_float(predicted_time, 2),
            "time_error_pct": finite_float(
                ((predicted_time - baseline_time) / baseline_time * 100.0) if baseline_time > 0 else 0.0,
                2,
            ),
            "total_coast_time_s": finite_float(total_coast_time, 2),
            "total_coast_distance_m": finite_float(total_coast_dist, 2),
            "pulse_count": pulse_count,
            "avg_pulse_duration_s": finite_float(avg_pulse_duration, 2),
            "strategy_step_m": finite_float(getattr(args, "strategy_step_m", None), 2),
            "display_region_min_m": finite_float(getattr(args, "display_region_min_m", None), 2),
            "total_over_fuse_s": finite_float(total_over, 2),
            "longest_over_fuse_s": finite_float(longest_over, 2),
            "fuse_current_ma": finite_float(args.fuse_current_ma, 0),
            "peak_current_mA": finite_float(
                pd.to_numeric(strategy_profile["pred_peak_current_mA"], errors="coerce").fillna(0.0).max(),
                0,
            ),
            "motor": {
                "name": motor_config.get("name"),
                "nominal_voltage_v": finite_float(motor_config.get("nominal_voltage_v"), 2),
                "nominal_output_w": finite_float(motor_config.get("nominal_output_w"), 1),
                "nominal_speed_rpm": finite_float(motor_config.get("nominal_speed_rpm"), 0),
                "top_speed_kph": finite_float(motor_config.get("top_speed_kph"), 1),
                "wheel_diameter_m": finite_float(motor_config.get("wheel_diameter_m"), 3),
                "vehicle_mass_kg": finite_float(motor_config.get("vehicle_mass_kg"), 1),
                "rolling_resistance_coeff": finite_float(motor_config.get("rolling_resistance_coeff"), 4),
                "drivetrain_efficiency": finite_float(motor_config.get("drivetrain_efficiency"), 3),
                "inferred_gear_ratio": finite_float(motor_config.get("inferred_gear_ratio"), 2),
            },
            "start_speed_kph": finite_float(strategy_profile.attrs.get("start_speed_kph", 0.0), 1),
            "report": strategy_report,
        },
    }


def make_payload(run_payloads: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "title": "UTSM Strategy Dashboard",
        "defaultRun": run_payloads[0]["id"],
        "runOrder": [run["id"] for run in run_payloads],
        "runs": {run["id"]: run for run in run_payloads},
        "metrics": {
            key: {
                "label": spec["label"],
                "unit": spec["unit"],
                "field": spec["field"],
                "color": spec["color"],
                "map_selectable": spec.get("map_selectable", True),
            }
            for key, spec in METRICS.items()
        },
        "controls": {
            "fuse_current_ma": finite_float(args.fuse_current_ma, 0),
        },
        "strategyLegend": STRATEGY_LEGEND,
        "mapLegend": {
            "car": {"label": "Car", "color": "#111827"},
            "lapStart": {"label": "Lap start", "color": "#3ecf59"},
            "reference": {"label": "Reference track", "color": "#d0d5dc"},
        },
    }


def longest_true_duration(flags: list[bool], durations: list[float]) -> float:
    best = 0.0
    run = 0.0
    for flag, duration in zip(flags, durations):
        if flag:
            run += float(duration)
            best = max(best, run)
        else:
            run = 0.0
    return best


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UTSM Strategy Dashboard</title>
  <style>
    :root {{
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #0f172a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      padding: 12px 18px;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid #d9dee6;
    }}
    h1 {{ margin: 0; font-size: 18px; }}
    .meta {{ font-size: 12px; color: #64748b; line-height: 1.35; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 18px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(560px, 1.05fr) minmax(520px, 0.95fr);
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #d9dee6;
      border-radius: 8px;
      padding: 14px;
    }}
    .map-wrap {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: stretch;
    }}
    .map-stage {{ position: relative; min-width: 0; }}
    svg {{ display: block; width: 100%; height: auto; background: #fff; }}
    #trackSvg {{ aspect-ratio: 1 / 1.08; border: 1px solid #d9dee6; }}
    .metric-legend {{
      border: 1px solid #d9dee6;
      display: grid;
      grid-template-rows: auto 1fr auto auto;
      width: 56px;
      min-height: 100%;
    }}
    .ramp {{ background: linear-gradient(to top, #2d0a73, #7e03a8, #cc4778, #f89540, #f0f921); }}
    .legend-label {{ font-size: 11px; text-align: center; padding: 4px 2px; color: #475569; }}
    .legend-title {{
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      font-size: 10px;
      color: #334155;
      text-align: center;
      padding: 6px 2px;
      border-bottom: 1px solid #e5e9ef;
      min-height: 92px;
    }}
    .legend-stack {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 28px;
      padding: 5px 8px;
      border: 1px solid #e5e9ef;
      border-radius: 6px;
      color: #334155;
      background: #fafbfc;
      font-size: 12px;
      white-space: nowrap;
    }}
    .legend-swatch {{
      width: 16px;
      height: 4px;
      border-radius: 999px;
      background: var(--swatch);
      border: 1px solid rgba(15, 23, 42, 0.18);
      flex: 0 0 auto;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--swatch);
      border: 2px solid #fff;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.25);
      flex: 0 0 auto;
    }}
    .map-readout {{
      position: absolute;
      left: 12px;
      right: 12px;
      bottom: 12px;
      max-width: calc(100% - 24px);
      padding: 6px 8px;
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.9);
      color: #334155;
      font-size: 11px;
      line-height: 1.25;
      pointer-events: none;
      white-space: normal;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid #e5e9ef;
    }}
    .slider-control {{
      flex: 1 1 280px;
      min-width: 220px;
    }}
    .controls label {{ flex: 0 0 auto; }}
    button, select {{
      height: 34px;
      border: 1px solid #b9c1cb;
      border-radius: 6px;
      background: #fff;
      color: #0f172a;
      font: inherit;
      padding: 0 10px;
    }}
    button {{ cursor: pointer; }}
    input[type="range"] {{ width: 100%; }}
    label {{
      font-size: 13px;
      color: #334155;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      margin-top: 12px;
    }}
    .stat {{
      border: 1px solid #e5e9ef;
      border-radius: 6px;
      padding: 8px;
      background: #fafbfc;
    }}
    .stat span {{ display: block; font-size: 11px; color: #64748b; margin-bottom: 3px; }}
    .stat strong {{ font-size: 16px; }}
    .charts {{
      display: grid;
      grid-template-rows: repeat(6, 118px);
      gap: 10px;
    }}
    .chart {{ border: 1px solid #d9dee6; }}
    .strategy-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .summary-card {{
      border: 1px solid #e5e9ef;
      border-radius: 6px;
      padding: 8px;
      background: #fafbfc;
    }}
    .summary-card span {{ display: block; font-size: 11px; color: #64748b; margin-bottom: 3px; }}
    .summary-card strong {{ font-size: 16px; }}
    pre {{
      margin: 0;
      padding: 10px;
      border: 1px solid #e5e9ef;
      border-radius: 6px;
      background: #fafbfc;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    details {{
      margin-top: 12px;
    }}
    summary {{
      cursor: pointer;
      color: #334155;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .axis-label {{ fill: #334155; font-size: 12px; }}
    .tick-label {{ fill: #64748b; font-size: 10px; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .map-wrap {{ grid-template-columns: 1fr; }}
      .metric-legend {{ width: 100%; min-height: 64px; grid-template-columns: auto 1fr auto auto; grid-template-rows: 1fr; }}
      .legend-title {{ writing-mode: horizontal-tb; transform: none; min-height: 0; border-bottom: 0; border-right: 1px solid #e5e9ef; }}
      .stats, .strategy-summary {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>UTSM Strategy Dashboard</h1>
    <div class="meta" id="metaText"></div>
  </header>
  <main>
    <section class="layout">
      <div class="panel">
        <div class="map-wrap">
          <div class="map-stage">
            <svg id="trackSvg" viewBox="0 0 720 780" role="img" aria-label="Track map">
              <g id="mapGrid"></g>
              <path id="fullTrack" fill="none" stroke="#d0d5dc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
              <g id="strategyLayer"></g>
              <g id="trailLayer"></g>
              <g id="labelLayer"></g>
              <g id="lapBoundaryLayer"></g>
              <circle id="lapStartMarker" r="7" fill="#3ecf59" stroke="#ffffff" stroke-width="2"></circle>
              <circle id="carMarker" r="8" fill="#ffffff" stroke="#111827" stroke-width="3"></circle>
            </svg>
            <div id="mapReadout" class="map-readout"></div>
          </div>
          <div class="metric-legend" id="metricLegend">
            <div class="legend-title" id="legendTitle"></div>
            <div class="ramp"></div>
            <div class="legend-label" id="legendMax"></div>
            <div class="legend-label" id="legendMin"></div>
          </div>
        </div>
        <div class="legend-stack" id="strategyLegend"></div>
        <div class="controls">
          <label>Run
            <select id="runSelect"></select>
          </label>
          <input id="timeSlider" class="slider-control" type="range" min="0" max="0" value="0" step="0.1">
          <button id="playButton" type="button">Play</button>
          <label>Metric
            <select id="metricSelect"></select>
          </label>
          <label>Map mode
            <select id="colorMode">
              <option value="action" selected>Action regions + metric trail</option>
              <option value="metric">Metric only</option>
            </select>
          </label>
          <label><input id="showStrategy" type="checkbox" checked> Strategy</label>
          <label><input id="showLabels" type="checkbox" checked> Labels</label>
          <span id="timeText" class="meta"></span>
        </div>
        <div class="stats">
          <div class="stat"><span>Lap</span><strong id="lapValue"></strong></div>
          <div class="stat"><span>Strategy</span><strong id="strategyValue"></strong></div>
          <div class="stat"><span>Target speed</span><strong id="targetSpeedValue"></strong></div>
          <div class="stat"><span>Pred current</span><strong id="predCurrentValue"></strong></div>
          <div class="stat"><span>Pred power</span><strong id="predPowerValue"></strong></div>
        </div>
      </div>
      <div class="panel">
        <div class="strategy-summary">
          <div class="summary-card"><span>Energy delta</span><strong id="energyDeltaValue"></strong></div>
          <div class="summary-card"><span>Pred time</span><strong id="predTimeValue"></strong></div>
          <div class="summary-card"><span>Time > 20A</span><strong id="overFuseValue"></strong></div>
          <div class="summary-card"><span>Longest burst</span><strong id="longestFuseValue"></strong></div>
          <div class="summary-card"><span>Pred peak current</span><strong id="peakCurrentValue"></strong></div>
          <div class="summary-card"><span>Coast time</span><strong id="coastTimeValue"></strong></div>
          <div class="summary-card"><span>Coast distance</span><strong id="coastDistanceValue"></strong></div>
          <div class="summary-card"><span>Pulse segments</span><strong id="pulseCountValue"></strong></div>
          <div class="summary-card"><span>Time error</span><strong id="timeErrorValue"></strong></div>
          <div class="summary-card"><span>Gear estimate</span><strong id="gearValue"></strong></div>
        </div>
        <div class="charts" id="charts"></div>
        <div class="legend-stack" id="chartLegend"></div>
        <details>
          <summary>Strategy report</summary>
          <pre id="reportText"></pre>
        </details>
      </div>
    </section>
  </main>
  <script>
    const DATA = {data_json};
    const ACTION_COLORS = Object.fromEntries(
      Object.entries(DATA.strategyLegend).map(([key, value]) => [key, value.color])
    );
    const W = 720;
    const H = 780;
    const PAD = 54;
    const CHART_W = 620;
    const CHART_H = 132;
    const CHART_PAD = {{ left: 58, right: 14, top: 14, bottom: 28 }};
    const chartKeys = ["current", "speed", "gpsAccel", "imuAccel", "power", "runEnergyJ"];
    const overlayFields = {{
      current: {{ field: "predCurrent", color: "#fb7185", label: "Pred avg current" }},
      speed: {{ field: "targetSpeed", color: "#f97316", label: "Target speed" }},
      power: {{ field: "predPower", color: "#c084fc", label: "Pred power" }},
      runEnergyJ: {{ field: "predRunEnergyJ", color: "#d97706", label: "Pred energy" }}
    }};
    const peakOverlayFields = {{
      current: {{ field: "predPeakCurrent", color: "#ef4444", label: "Pred peak current" }}
    }};
    const metricSpecs = DATA.metrics;
    const metricKeys = Object.keys(metricSpecs).filter(k => metricSpecs[k].map_selectable !== false);
    const state = {{
      runId: DATA.defaultRun,
      index: 0,
      metric: "speed",
      playing: false,
      lastTick: null
    }};

    const el = {{
      metaText: document.getElementById("metaText"),
      runSelect: document.getElementById("runSelect"),
      timeSlider: document.getElementById("timeSlider"),
      playButton: document.getElementById("playButton"),
      metricSelect: document.getElementById("metricSelect"),
      colorMode: document.getElementById("colorMode"),
      showStrategy: document.getElementById("showStrategy"),
      showLabels: document.getElementById("showLabels"),
      timeText: document.getElementById("timeText"),
      lapValue: document.getElementById("lapValue"),
      strategyValue: document.getElementById("strategyValue"),
      targetSpeedValue: document.getElementById("targetSpeedValue"),
      predCurrentValue: document.getElementById("predCurrentValue"),
      predPowerValue: document.getElementById("predPowerValue"),
      energyDeltaValue: document.getElementById("energyDeltaValue"),
      predTimeValue: document.getElementById("predTimeValue"),
      overFuseValue: document.getElementById("overFuseValue"),
      longestFuseValue: document.getElementById("longestFuseValue"),
      peakCurrentValue: document.getElementById("peakCurrentValue"),
      coastTimeValue: document.getElementById("coastTimeValue"),
      coastDistanceValue: document.getElementById("coastDistanceValue"),
      pulseCountValue: document.getElementById("pulseCountValue"),
      timeErrorValue: document.getElementById("timeErrorValue"),
      gearValue: document.getElementById("gearValue"),
      reportText: document.getElementById("reportText"),
      chartLegend: document.getElementById("chartLegend"),
      strategyLegend: document.getElementById("strategyLegend"),
      metricLegend: document.getElementById("metricLegend"),
      legendTitle: document.getElementById("legendTitle"),
      legendMin: document.getElementById("legendMin"),
      legendMax: document.getElementById("legendMax"),
      trackSvg: document.getElementById("trackSvg"),
      fullTrack: document.getElementById("fullTrack"),
      strategyLayer: document.getElementById("strategyLayer"),
      labelLayer: document.getElementById("labelLayer"),
      trailLayer: document.getElementById("trailLayer"),
      lapBoundaryLayer: document.getElementById("lapBoundaryLayer"),
      lapStartMarker: document.getElementById("lapStartMarker"),
      carMarker: document.getElementById("carMarker"),
      mapReadout: document.getElementById("mapReadout"),
      charts: document.getElementById("charts"),
    }};

    function getRun() {{
      return DATA.runs[state.runId];
    }}

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function scaleLinear(value, domain, range) {{
      const [d0, d1] = domain;
      const [r0, r1] = range;
      if (d0 === d1) return (r0 + r1) / 2;
      return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
    }}

    function xMap(x) {{
      return scaleLinear(x, getRun().domains.x, [PAD, W - PAD]);
    }}

    function yMap(y) {{
      return scaleLinear(y, getRun().domains.y, [H - PAD, PAD]);
    }}

    function chartX(t) {{
      return scaleLinear(t, [0, getRun().meta.duration_s], [CHART_PAD.left, CHART_W - CHART_PAD.right]);
    }}

    function chartY(value, key) {{
      return scaleLinear(value, getRun().domains.metrics[key], [CHART_H - CHART_PAD.bottom, CHART_PAD.top]);
    }}

    function linePath(rows, xFn, yFn) {{
      let d = "";
      for (const row of rows) {{
        const x = xFn(row);
        const y = yFn(row);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        d += d ? `L${{x.toFixed(1)}} ${{y.toFixed(1)}}` : `M${{x.toFixed(1)}} ${{y.toFixed(1)}}`;
      }}
      return d;
    }}

    function runSamples() {{
      return getRun().samples;
    }}

    function nearestIndexByTime(t) {{
      const samples = runSamples();
      let lo = 0;
      let hi = samples.length - 1;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (samples[mid].t < t) lo = mid + 1;
        else hi = mid;
      }}
      if (lo > 0 && Math.abs(samples[lo - 1].t - t) < Math.abs(samples[lo].t - t)) return lo - 1;
      return lo;
    }}

    function currentLapRange(index) {{
      const samples = runSamples();
      const lap = samples[index].lap;
      return getRun().laps.find(item => item.lap === lap);
    }}

    function colorForMetric(value, key) {{
      const [lo, hi] = getRun().domains.metrics[key];
      const ratio = clamp((value - lo) / (hi - lo || 1), 0, 1);
      const stops = [[45,10,115],[126,3,168],[204,71,120],[248,149,64],[240,249,33]];
      const scaled = ratio * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const local = scaled - idx;
      const a = stops[idx];
      const b = stops[idx + 1];
      const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * local));
      return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
    }}

    function format(value, digits, suffix) {{
      if (!Number.isFinite(value)) return "-";
      return `${{value.toFixed(digits)}}${{suffix}}`;
    }}

    function drawMapGrid() {{
      const grid = document.getElementById("mapGrid");
      const lines = [];
      for (let i = 0; i <= 6; i++) {{
        const x = PAD + ((W - 2 * PAD) * i) / 6;
        const y = PAD + ((H - 2 * PAD) * i) / 6;
        lines.push(`<line x1="${{x}}" x2="${{x}}" y1="${{PAD}}" y2="${{H - PAD}}" stroke="#eef1f4" stroke-width="1"/>`);
        lines.push(`<line x1="${{PAD}}" x2="${{W - PAD}}" y1="${{y}}" y2="${{y}}" stroke="#eef1f4" stroke-width="1"/>`);
      }}
      grid.innerHTML = lines.join("");
    }}

    function drawFullTrack() {{
      const samples = runSamples();
      el.fullTrack.setAttribute("d", linePath(samples, s => xMap(s.x), s => yMap(s.y)));
    }}

    function drawBoundaries() {{
      const samples = runSamples();
      el.lapBoundaryLayer.innerHTML = getRun().laps.map(lap => {{
        const s = samples[lap.start];
        return `<g><circle cx="${{xMap(s.x)}}" cy="${{yMap(s.y)}}" r="5" fill="#f59e0b" stroke="#fff" stroke-width="2"></circle><text x="${{xMap(s.x)+8}}" y="${{yMap(s.y)-8}}" class="tick-label">L${{lap.lap}}</text></g>`;
      }}).join("");
    }}

    function drawStrategy(index) {{
      const samples = runSamples();
      const range = currentLapRange(index);
      if (!el.showStrategy.checked) {{
        el.strategyLayer.innerHTML = "";
        el.labelLayer.innerHTML = "";
        return;
      }}
      const lapRows = samples.slice(range.start, range.end + 1);
      let strategyMarkup = "";
      const primary = el.colorMode.value === "action";
      const strokeWidth = primary ? 8 : 3;
      const opacity = primary ? 0.32 : 0.12;
      for (let i = 1; i < lapRows.length; i++) {{
        const a = lapRows[i - 1];
        const b = lapRows[i];
        const color = ACTION_COLORS[b.strategyAction] || "#64748b";
        strategyMarkup += `<line x1="${{xMap(a.x).toFixed(1)}}" y1="${{yMap(a.y).toFixed(1)}}" x2="${{xMap(b.x).toFixed(1)}}" y2="${{yMap(b.y).toFixed(1)}}" stroke="${{color}}" stroke-width="${{strokeWidth}}" stroke-linecap="round" opacity="${{opacity}}"/>`;
      }}
      el.strategyLayer.innerHTML = strategyMarkup;

      if (!el.showLabels.checked) {{
        el.labelLayer.innerHTML = "";
        return;
      }}
      const labels = getRun().segments
        .filter(seg => seg.lap === lapRows[0].lap && seg.label_x !== null && seg.label_y !== null)
        .map(seg => {{
          const lx = xMap(seg.label_x + (seg.label_dx_m || 0));
          const ly = yMap(seg.label_y + (seg.label_dy_m || 0));
          const color = ACTION_COLORS[seg.action] || "#64748b";
          return `<g><rect x="${{lx - 15}}" y="${{ly - 9}}" width="30" height="18" rx="9" fill="rgba(255,255,255,0.94)" stroke="${{color}}" stroke-width="1.5"/><text x="${{lx}}" y="${{ly + 3}}" text-anchor="middle" class="tick-label">S${{seg.segment}}</text></g>`;
        }});
      el.labelLayer.innerHTML = labels.join("");
    }}

    function drawTrail(index) {{
      const samples = runSamples();
      const range = currentLapRange(index);
      const rows = samples.slice(range.start, index + 1);
      if (rows.length < 2) {{
        el.trailLayer.innerHTML = "";
        return;
      }}
      let markup = "";
      for (let i = 1; i < rows.length; i++) {{
        const a = rows[i - 1];
        const b = rows[i];
        const color = colorForMetric(b[metricSpecs[state.metric].field], state.metric);
        const metricMode = el.colorMode.value === "metric";
        const width = metricMode ? 6 : 4;
        const opacity = metricMode ? 0.96 : 0.9;
        markup += `<line x1="${{xMap(a.x).toFixed(1)}}" y1="${{yMap(a.y).toFixed(1)}}" x2="${{xMap(b.x).toFixed(1)}}" y2="${{yMap(b.y).toFixed(1)}}" stroke="${{color}}" stroke-width="${{width}}" stroke-linecap="round" opacity="${{opacity}}"/>`;
      }}
      el.trailLayer.innerHTML = markup;
    }}

    function makeChart(key) {{
      const spec = metricSpecs[key];
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "chart");
      svg.setAttribute("viewBox", `0 0 ${{CHART_W}} ${{CHART_H}}`);
      svg.dataset.key = key;
      svg.innerHTML = `
        <text x="8" y="18" class="axis-label">${{spec.label}} (${{spec.unit}})</text>
        <path class="full-line" fill="none" stroke="#d2d7de" stroke-width="2"></path>
        <path class="progress-line" fill="none" stroke="${{spec.color}}" stroke-width="2.4"></path>
        <path class="overlay-full" fill="none" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="6 5"></path>
        <path class="overlay-progress" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="6 5"></path>
        <path class="peak-overlay-full" fill="none" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="2 4"></path>
        <path class="peak-overlay-progress" fill="none" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="2 4"></path>
        <line class="cursor-line" y1="${{CHART_PAD.top}}" y2="${{CHART_H - CHART_PAD.bottom}}" stroke="#111827" stroke-width="1"></line>
        <line class="threshold-line" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="5 4"></line>
        <text class="threshold-label tick-label" text-anchor="end"></text>
        <text class="tick-label min-label" x="${{CHART_PAD.left}}" y="${{CHART_H - 8}}"></text>
        <text class="tick-label max-label" x="${{CHART_PAD.left}}" y="11"></text>
      `;
      el.charts.appendChild(svg);
    }}

    function refreshCharts() {{
      const run = getRun();
      const samples = runSamples();
      document.querySelectorAll(".chart").forEach(svg => {{
        const key = svg.dataset.key;
        const spec = metricSpecs[key];
        svg.querySelector(".full-line").setAttribute("d", linePath(samples, s => chartX(s.t), s => chartY(s[spec.field], key)));
        const [lo, hi] = run.domains.metrics[key];
        svg.querySelector(".min-label").textContent = lo.toFixed(1);
        svg.querySelector(".max-label").textContent = hi.toFixed(1);
        const overlay = overlayFields[key];
        if (overlay) {{
          svg.querySelector(".overlay-full").setAttribute("d", linePath(samples, s => chartX(s.t), s => chartY(s[overlay.field], key)));
        }} else {{
          svg.querySelector(".overlay-full").setAttribute("d", "");
          svg.querySelector(".overlay-progress").setAttribute("d", "");
        }}
        const peakOverlay = peakOverlayFields[key];
        if (peakOverlay) {{
          svg.querySelector(".peak-overlay-full").setAttribute("d", linePath(samples, s => chartX(s.t), s => chartY(s[peakOverlay.field], key)));
        }} else {{
          svg.querySelector(".peak-overlay-full").setAttribute("d", "");
          svg.querySelector(".peak-overlay-progress").setAttribute("d", "");
        }}
        const threshold = svg.querySelector(".threshold-line");
        const thresholdLabel = svg.querySelector(".threshold-label");
        if (key === "current") {{
          const y = chartY(DATA.controls.fuse_current_ma, "current");
          threshold.setAttribute("x1", CHART_PAD.left);
          threshold.setAttribute("x2", CHART_W - CHART_PAD.right);
          threshold.setAttribute("y1", y);
          threshold.setAttribute("y2", y);
          thresholdLabel.setAttribute("x", CHART_W - CHART_PAD.right - 4);
          thresholdLabel.setAttribute("y", y - 4);
          thresholdLabel.textContent = "20 A fuse";
          threshold.style.display = "";
          thresholdLabel.style.display = "";
        }} else {{
          threshold.style.display = "none";
          thresholdLabel.style.display = "none";
        }}
      }});
    }}

    function updateCharts(index) {{
      const run = getRun();
      const samples = runSamples();
      const row = samples[index];
      const lapRange = currentLapRange(index);
      const lapRows = samples.slice(lapRange.start, index + 1);
      document.querySelectorAll(".chart").forEach(svg => {{
        const key = svg.dataset.key;
        const spec = metricSpecs[key];
        const rows = key === "runEnergyJ" ? samples.slice(0, index + 1) : lapRows;
        svg.querySelector(".progress-line").setAttribute("d", linePath(rows, s => chartX(s.t), s => chartY(s[spec.field], key)));
        const overlay = overlayFields[key];
        if (overlay) {{
          svg.querySelector(".overlay-progress").setAttribute("d", linePath(rows, s => chartX(s.t), s => chartY(s[overlay.field], key)));
        }}
        const peakOverlay = peakOverlayFields[key];
        if (peakOverlay) {{
          svg.querySelector(".peak-overlay-progress").setAttribute("d", linePath(rows, s => chartX(s.t), s => chartY(s[peakOverlay.field], key)));
        }}
        const x = chartX(row.t);
        const cursor = svg.querySelector(".cursor-line");
        cursor.setAttribute("x1", x);
        cursor.setAttribute("x2", x);
      }});
    }}

    function updateSummary() {{
      const strategy = getRun().strategy;
      el.energyDeltaValue.textContent = `${{strategy.delta_energy_pct.toFixed(2)}}%`;
      el.predTimeValue.textContent = `${{strategy.predicted_time_s.toFixed(1)}} s`;
      el.overFuseValue.textContent = `${{strategy.total_over_fuse_s.toFixed(2)}} s`;
      el.longestFuseValue.textContent = `${{strategy.longest_over_fuse_s.toFixed(2)}} s`;
      el.peakCurrentValue.textContent = `${{(strategy.peak_current_mA / 1000).toFixed(1)}} A`;
      el.coastTimeValue.textContent = `${{strategy.total_coast_time_s.toFixed(1)}} s`;
      el.coastDistanceValue.textContent = `${{strategy.total_coast_distance_m.toFixed(0)}} m`;
      el.pulseCountValue.textContent = `${{strategy.pulse_count}}`;
      el.timeErrorValue.textContent = `${{strategy.time_error_pct.toFixed(2)}}%`;
      el.gearValue.textContent = `${{strategy.motor.inferred_gear_ratio.toFixed(2)}}:1`;
      el.reportText.textContent = strategy.report;
    }}

    function update(index) {{
      const run = getRun();
      const samples = runSamples();
      state.index = clamp(index, 0, samples.length - 1);
      const row = samples[state.index];
      const lapRange = currentLapRange(state.index);
      const lapStart = samples[lapRange.start];
      el.timeSlider.value = row.t;
      el.carMarker.setAttribute("cx", xMap(row.x));
      el.carMarker.setAttribute("cy", yMap(row.y));
      el.lapStartMarker.setAttribute("cx", xMap(lapStart.x));
      el.lapStartMarker.setAttribute("cy", yMap(lapStart.y));
      drawStrategy(state.index);
      drawTrail(state.index);
      drawBoundaries();
      updateCharts(state.index);
      el.timeText.textContent = `t=${{row.t.toFixed(1)}}s / ${{run.meta.duration_s.toFixed(1)}}s`;
      const metric = metricSpecs[state.metric];
      const metricValue = row[metric.field];
      el.mapReadout.textContent = `t=${{row.t.toFixed(1)}}s  lap=${{row.lap}}  seg=${{row.segment}}  speed=${{row.speed.toFixed(1)}} km/h  action=${{row.strategyAction}}  pred avg=${{row.predCurrent.toFixed(0)}} mA  pred peak=${{row.predPeakCurrent.toFixed(0)}} mA  pulse=${{row.pulseDuration.toFixed(1)}}s coast=${{row.coastDuration.toFixed(1)}}s  ${{metric.label}}=${{format(metricValue, 2, " " + metric.unit)}}`;
      el.lapValue.textContent = String(row.lap);
      el.strategyValue.textContent = row.strategyAction;
      el.targetSpeedValue.textContent = format(row.targetSpeed, 1, " km/h");
      el.predCurrentValue.textContent = format(row.predCurrent, 0, " mA");
      el.predPowerValue.textContent = format(row.predPower, 1, " W");
      const domain = run.domains.metrics[state.metric];
      el.legendMin.textContent = domain[0].toFixed(1);
      el.legendMax.textContent = domain[1].toFixed(1);
      el.legendTitle.textContent = `Trail color: ${{metric.label}} (${{metric.unit}})`;
      el.metricLegend.style.display = "grid";
      el.metaText.textContent = `${{run.label}} | ${{run.meta.sample_count}} samples | ${{run.meta.duration_s.toFixed(1)}}s | energy delta ${{run.strategy.delta_energy_pct.toFixed(2)}}%`;
    }}

    function buildLegends() {{
      const actionItems = Object.entries(DATA.strategyLegend).map(([key, item]) =>
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:${{item.color}}"></span>${{item.label}}</span>`
      );
      const map = DATA.mapLegend;
      const mapItems = [
        `<span class="legend-item"><span class="legend-dot" style="--swatch:#fff; border-color:${{map.car.color}}"></span>${{map.car.label}}</span>`,
        `<span class="legend-item"><span class="legend-dot" style="--swatch:${{map.lapStart.color}}"></span>${{map.lapStart.label}}</span>`,
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:${{map.reference.color}}"></span>${{map.reference.label}}</span>`
      ];
      el.strategyLegend.innerHTML = [...actionItems, ...mapItems].join("");
      el.chartLegend.innerHTML = [
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:#d2d7de"></span>Actual full run</span>`,
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:${{overlayFields.current.color}}"></span>Predicted average current / predicted overlay</span>`,
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:${{peakOverlayFields.current.color}}"></span>Predicted peak current spikes</span>`,
        `<span class="legend-item"><span class="legend-swatch" style="--swatch:#dc2626"></span>20 A fuse threshold</span>`
      ].join("");
    }}

    function animationTick(now) {{
      if (!state.playing) return;
      if (state.lastTick === null) state.lastTick = now;
      const delta = (now - state.lastTick) / 1000;
      state.lastTick = now;
      const nextT = Math.min(getRun().meta.duration_s, runSamples()[state.index].t + delta * 10);
      update(nearestIndexByTime(nextT));
      if (nextT >= getRun().meta.duration_s) {{
        state.playing = false;
        el.playButton.textContent = "Play";
        state.lastTick = null;
        return;
      }}
      requestAnimationFrame(animationTick);
    }}

    function switchRun(runId) {{
      state.runId = runId;
      state.index = 0;
      const run = getRun();
      el.timeSlider.max = run.meta.duration_s;
      drawFullTrack();
      refreshCharts();
      updateSummary();
      update(0);
    }}

    function init() {{
      DATA.runOrder.forEach(runId => {{
        const run = DATA.runs[runId];
        const option = document.createElement("option");
        option.value = runId;
        option.textContent = run.label;
        el.runSelect.appendChild(option);
      }});
      metricKeys.forEach(key => {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = metricSpecs[key].label;
        el.metricSelect.appendChild(option);
      }});
      el.metricSelect.value = state.metric;
      chartKeys.forEach(makeChart);
      buildLegends();
      drawMapGrid();
      el.runSelect.addEventListener("change", e => switchRun(e.target.value));
      el.timeSlider.addEventListener("input", e => update(nearestIndexByTime(Number(e.target.value))));
      el.metricSelect.addEventListener("change", e => {{
        state.metric = e.target.value;
        update(state.index);
      }});
      el.colorMode.addEventListener("change", () => update(state.index));
      el.showStrategy.addEventListener("change", () => update(state.index));
      el.showLabels.addEventListener("change", () => update(state.index));
      el.playButton.addEventListener("click", () => {{
        state.playing = !state.playing;
        el.playButton.textContent = state.playing ? "Pause" : "Play";
        state.lastTick = null;
        if (state.playing) requestAnimationFrame(animationTick);
      }});
      switchRun(DATA.defaultRun);
    }}

    init();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    run_specs = resolve_run_specs(args)
    run_payloads = []
    for spec in run_specs:
        df, strategy_profile, strategy_report = load_single_run(spec, args)
        run_payloads.append(make_run_payload(spec, df, strategy_profile, strategy_report, args))
    payload = make_payload(run_payloads, args)
    html = build_html(payload)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote interactive dashboard: {args.output}")
    for run in run_payloads:
        print(f"{run['label']}: {run['meta']['sample_count']} samples, {run['meta']['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
