"""Empirical speed-profile optimizer for UTSM telemetry runs."""

from __future__ import annotations

import argparse
import os
import sys

try:
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
    build_full_run_distance,
    build_motor_config,
    build_strategy_report,
    build_strategy_samples,
    build_strategy_segments,
    build_strategy_segments_by_distance,
    derive_motion_energy,
    evaluate_baseline_prediction,
    fit_empirical_energy_model,
    merge_by_time,
    optimize_speed_profile,
    read_gpx,
    read_telemetry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize an explicit accelerate/hold/coast strategy from historical telemetry data."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
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
    parser.add_argument(
        "--forward-axis",
        choices=FORWARD_AXIS_CHOICES,
        default="ax",
    )
    parser.add_argument("--accel-window", type=int, default=5)
    parser.add_argument("--accel-scale", type=float, default=1000.0)
    parser.add_argument("--imu-axis", choices=["ax", "ay", "az"], default="ax")
    parser.add_argument("--imu-axis-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--accel-bias-window-sec", type=float, default=30.0)
    parser.add_argument("--accel-smooth-window-sec", type=float, default=8.0)
    parser.add_argument("--segments", type=int, default=None, help="Legacy fixed segment count. Overrides --strategy-step-m when set.")
    parser.add_argument("--strategy-step-m", type=float, default=50.0)
    parser.add_argument("--time-tolerance-pct", type=float, default=3.0)
    parser.add_argument("--time-budget-sec", type=float, default=2100.0)
    parser.add_argument("--lap-time-target-sec", type=float)
    parser.add_argument("--speed-min-kph", type=float, default=8.0)
    parser.add_argument("--speed-max-kph", type=float, default=40.0)
    parser.add_argument("--max-delta-kph-per-segment", type=float, default=6.0)
    parser.add_argument("--speed-step-kph", type=float, default=1.0)
    parser.add_argument("--hold-delta-kph", type=float, default=1.0)
    parser.add_argument("--fuse-current-ma", type=float, default=20000.0)
    parser.add_argument("--fuse-max-duration-sec", type=float, default=1.0)
    parser.add_argument("--current-penalty-weight", type=float, default=5.0)
    parser.add_argument("--wheel-diameter-m", type=float, default=0.50)
    parser.add_argument("--vehicle-mass-kg", type=float, default=100.0)
    parser.add_argument("--rolling-resistance-coeff", type=float, default=0.008)
    parser.add_argument("--drivetrain-efficiency", type=float, default=0.82)
    parser.add_argument("--start-speed-kph", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="outputs/speed_strategy")
    return parser.parse_args()


def load_full_run(args: argparse.Namespace) -> pd.DataFrame:
    gps_df = read_gpx(args.gps)
    telem_df = read_telemetry(args.telemetry)
    gps_laps, telem_laps, _ = build_laps(gps_df, telem_df, args)

    rows = []
    distance_offset = 0.0
    energy_offset = 0.0
    for lap_num, (lap_gps, lap_telem) in enumerate(zip(gps_laps, telem_laps), start=1):
        if lap_gps.empty or lap_telem.empty:
            fallback = lap_gps.copy()
            fallback["current_mA"] = 0.0
            fallback["voltage_mV"] = 0.0
            fallback["ax_x100"] = 0.0
            fallback["ay_x100"] = 0.0
            fallback["az_x100"] = 0.0
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
            derived["run_cumdist_m"] = derived["cumdist_m"] + distance_offset
            derived["cum_energy_j"] = derived["cum_energy_j"] + energy_offset
            distance_offset = float(derived["run_cumdist_m"].iloc[-1])
            energy_offset = float(derived["cum_energy_j"].iloc[-1])
            rows.append(derived)
            continue
        try:
            merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        except ValueError as exc:
            print(f"Lap {lap_num}: telemetry merge failed, using GPS-only fallback - {exc}")
            fallback = lap_gps.copy()
            fallback["current_mA"] = 0.0
            fallback["voltage_mV"] = 0.0
            fallback["ax_x100"] = 0.0
            fallback["ay_x100"] = 0.0
            fallback["az_x100"] = 0.0
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
            derived["run_cumdist_m"] = derived["cumdist_m"] + distance_offset
            derived["cum_energy_j"] = derived["cum_energy_j"] + energy_offset
            distance_offset = float(derived["run_cumdist_m"].iloc[-1])
            energy_offset = float(derived["cum_energy_j"].iloc[-1])
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
        derived["run_cumdist_m"] = derived["cumdist_m"] + distance_offset
        derived["cum_energy_j"] = derived["cum_energy_j"] + energy_offset
        distance_offset = float(derived["run_cumdist_m"].iloc[-1])
        energy_offset = float(derived["cum_energy_j"].iloc[-1])
        rows.append(derived)

    if not rows:
        raise ValueError("No lap data could be merged for simulation.")
    return build_full_run_distance(
        pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Reading GPX: {args.gps}")
    print(f"Reading telemetry: {args.telemetry}")
    full_run = load_full_run(args)
    train_run = full_run[full_run["telemetry_available"]].copy()
    model = fit_empirical_energy_model(train_run if not train_run.empty else full_run)
    if args.segments is not None:
        segments_df = build_strategy_segments(full_run, args.segments)
    else:
        segments_df = build_strategy_segments_by_distance(full_run, args.strategy_step_m)
    motor_config = build_motor_config(
        wheel_diameter_m=args.wheel_diameter_m,
        vehicle_mass_kg=args.vehicle_mass_kg,
        rolling_resistance_coeff=args.rolling_resistance_coeff,
        drivetrain_efficiency=args.drivetrain_efficiency,
    )
    calibration = evaluate_baseline_prediction(
        segments_df,
        model,
        motor_config=motor_config,
        hold_delta_kph=args.hold_delta_kph,
        start_speed_kph=args.start_speed_kph,
    )

    baseline_time_s = float(pd.to_numeric(full_run["dt_s"], errors="coerce").fillna(0.0).sum())
    if args.lap_time_target_sec is not None and args.laps > 0:
        time_budget_sec = args.lap_time_target_sec * args.laps
        time_target_sec = time_budget_sec
    else:
        time_target_sec = baseline_time_s
        configured_budget = args.time_budget_sec if args.time_budget_sec is not None else baseline_time_s
        time_budget_sec = min(configured_budget, baseline_time_s * (1.0 + args.time_tolerance_pct / 100.0))
    min_time_sec = max(time_target_sec * (1.0 - args.time_tolerance_pct / 100.0), 1.0)

    profile_df = optimize_speed_profile(
        segments_df,
        model,
        time_budget_sec=time_budget_sec,
        speed_min_kph=args.speed_min_kph,
        speed_max_kph=args.speed_max_kph,
        max_delta_kph_per_segment=args.max_delta_kph_per_segment,
        speed_step_kph=args.speed_step_kph,
        hold_delta_kph=args.hold_delta_kph,
        fuse_current_ma=args.fuse_current_ma,
        fuse_max_duration_sec=args.fuse_max_duration_sec,
        current_penalty_weight=args.current_penalty_weight,
        motor_config=motor_config,
        start_speed_kph=args.start_speed_kph,
        min_time_sec=min_time_sec,
    )
    samples_df = build_strategy_samples(full_run, profile_df)
    report = build_strategy_report(full_run, profile_df, time_budget_sec, calibration=calibration)

    profile_csv = args.output_prefix + "_strategy_profile.csv"
    samples_csv = args.output_prefix + "_strategy_samples.csv"
    report_txt = args.output_prefix + "_strategy_report.txt"

    profile_df.to_csv(profile_csv, index=False)
    samples_df.to_csv(samples_csv, index=False)
    with open(report_txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(f"Wrote: {profile_csv}, {samples_csv}, {report_txt}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
