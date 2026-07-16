"""Export optimized dashboard strategy into firmware lookup tables."""

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

from simulate_speed_strategy import load_full_run
from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    build_motor_config,
    build_strategy_samples,
    build_strategy_segments,
    build_strategy_segments_by_distance,
    evaluate_baseline_prediction,
    fit_empirical_energy_model,
    optimize_speed_profile,
)
from utsm_telemetry.strategy_export import (
    build_firmware_strategy_table,
    write_strategy_csv,
    write_strategy_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the same optimized speed strategy used by the dashboard and "
            "export it as a firmware-friendly GPS lookup table."
        )
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
    parser.add_argument("--name", default="strategy_indy", help="Map/header name used in generated identifiers.")
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
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--strategy-step-m", type=float, default=50.0)
    parser.add_argument("--export-spacing-m", type=float, default=10.0)
    parser.add_argument("--offtrack-radius-m", type=float, default=40.0)
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
    parser.add_argument("--vehicle-mass-kg", type=float, default=50.0)
    parser.add_argument("--driver-mass-kg", type=float, default=50.0)
    parser.add_argument("--rolling-resistance-coeff", type=float, default=0.008)
    parser.add_argument("--drivetrain-efficiency", type=float, default=0.82)
    parser.add_argument("--corner-drag-factor", type=float, default=0.1)
    parser.add_argument("--cd-area-m2", type=float, default=0.07235)
    parser.add_argument("--air-density-kg-m3", type=float, default=1.225)
    parser.add_argument("--start-speed-kph", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="outputs/firmware_strategy/indy")
    return parser.parse_args()


def build_strategy(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        driver_mass_kg=args.driver_mass_kg,
        rolling_resistance_coeff=args.rolling_resistance_coeff,
        drivetrain_efficiency=args.drivetrain_efficiency,
        corner_drag_factor=args.corner_drag_factor,
        cd_area_m2=args.cd_area_m2,
        air_density_kg_m3=args.air_density_kg_m3,
    )
    evaluate_baseline_prediction(
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
    return samples_df, profile_df


def main() -> int:
    args = parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    samples_df, profile_df = build_strategy(args)
    table, meta = build_firmware_strategy_table(
        samples_df,
        profile_df,
        name=args.name,
        spacing_m=args.export_spacing_m,
        offtrack_radius_m=args.offtrack_radius_m,
    )

    csv_path = args.output_prefix + "_strategy_map.csv"
    header_path = args.output_prefix + "_strategy_map.h"
    write_strategy_csv(table, csv_path)
    write_strategy_header(table, meta, header_path)
    print(f"Wrote firmware strategy CSV: {csv_path}")
    print(f"Wrote firmware strategy header: {header_path}")
    print(f"Points: {len(table)}, spacing: {meta.spacing_m:.1f} m, off-track radius: {meta.offtrack_radius_m:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
