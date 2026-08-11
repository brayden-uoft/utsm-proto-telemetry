"""Generate a transferable maximum-efficiency strategy for a reference track."""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utsm_telemetry import (
    build_laps,
    build_motor_config,
    derive_motion_energy,
    fit_empirical_energy_model,
    merge_by_time,
    optimize_speed_profile,
    read_gpx,
    read_telemetry,
)


GPX_NS = "http://www.topografix.com/GPX/1/1"
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the existing telemetry energy model onto a geometry-only "
            "reference track and minimize predicted electrical energy subject "
            "to a lap-time constraint."
        )
    )
    parser.add_argument("reference_gpx", help="Untimed reference-track centerline GPX")
    parser.add_argument("model_gpx", help="Recorded GPX used to fit the vehicle model")
    parser.add_argument("model_telemetry", help="Recorded telemetry CSV used to fit the vehicle model")
    parser.add_argument("--output-prefix", default=os.path.join("outputs", "reference-efficiency"))
    parser.add_argument("--preview", help="Optional PNG with the map and speed-vs-distance curve")
    parser.add_argument("--model-laps", type=int, default=3)
    parser.add_argument("--target-lap-time-sec", type=float, default=60.0)
    parser.add_argument("--strategy-step-m", type=float, default=20.0)
    parser.add_argument("--speed-min-kph", type=float, default=8.0)
    parser.add_argument("--speed-max-kph", type=float, default=35.0)
    parser.add_argument("--speed-step-kph", type=float, default=1.0)
    parser.add_argument("--max-delta-kph-per-segment", type=float, default=5.0)
    parser.add_argument("--start-speed-kph", type=float)
    parser.add_argument("--closed-loop-tolerance-kph", type=float, default=0.5)
    parser.add_argument("--fuse-current-ma", type=float, default=20000.0)
    parser.add_argument("--fuse-max-duration-sec", type=float, default=1.0)
    parser.add_argument("--wheel-diameter-m", type=float, default=0.50)
    parser.add_argument("--vehicle-mass-kg", type=float, default=50.0)
    parser.add_argument("--driver-mass-kg", type=float, default=50.0)
    parser.add_argument("--rolling-resistance-coeff", type=float, default=0.008)
    parser.add_argument("--drivetrain-efficiency", type=float, default=0.82)
    parser.add_argument("--corner-drag-factor", type=float, default=0.1)
    parser.add_argument("--cd-area-m2", type=float, default=0.07235)
    return parser.parse_args()


def read_reference_geometry(path: str | os.PathLike[str]) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    ns = {"gpx": GPX_NS}
    points = [
        (
            float(node.attrib["lat"]),
            float(node.attrib["lon"]),
            float(node.find("gpx:ele", ns).text)
            if node.find("gpx:ele", ns) is not None
            else 0.0,
        )
        for node in root.findall("gpx:trk/gpx:trkseg/gpx:trkpt", ns)
    ]
    if len(points) < 8:
        raise ValueError("Reference GPX needs at least eight points.")

    frame = pd.DataFrame(points, columns=["lat", "lon", "elev"])
    lat0 = float(frame["lat"].mean())
    lon0 = float(frame["lon"].mean())
    frame["y"] = np.radians(frame["lat"] - lat0) * EARTH_RADIUS_M
    frame["x"] = (
        np.radians(frame["lon"] - lon0)
        * EARTH_RADIUS_M
        * math.cos(math.radians(lat0))
    )
    if float(np.hypot(frame["x"].iloc[-1] - frame["x"].iloc[0], frame["y"].iloc[-1] - frame["y"].iloc[0])) < 0.5:
        frame = frame.iloc[:-1].copy()
    frame = frame.reset_index(drop=True)

    xy = frame[["x", "y"]].to_numpy(dtype=float)
    closed_delta = np.roll(xy, -1, axis=0) - xy
    edge_lengths = np.linalg.norm(closed_delta, axis=1)
    total_length = float(edge_lengths.sum())
    if total_length <= 0:
        raise ValueError("Reference GPX has zero path length.")
    frame["run_cumdist_m"] = np.concatenate(([0.0], np.cumsum(edge_lengths[:-1])))
    frame["curvature_1_m"] = periodic_curvature(xy, step=2)
    frame.attrs["track_length_m"] = total_length
    return frame


def periodic_curvature(xy: np.ndarray, step: int = 2) -> np.ndarray:
    """Estimate closed-loop curvature from circumcircles through spaced points."""
    count = len(xy)
    if count < 2 * step + 1:
        raise ValueError("Not enough points for the requested curvature window.")
    curvature = np.zeros(count, dtype=float)
    for index in range(count):
        a = xy[(index - step) % count]
        b = xy[index]
        c = xy[(index + step) % count]
        ab = float(np.linalg.norm(a - b))
        bc = float(np.linalg.norm(b - c))
        ca = float(np.linalg.norm(c - a))
        ba = b - a
        ca_vector = c - a
        twice_area = abs(float(ba[0] * ca_vector[1] - ba[1] * ca_vector[0]))
        denominator = ab * bc * ca
        curvature[index] = (2.0 * twice_area / denominator) if denominator > 1e-9 else 0.0
    return circular_median(curvature, radius=1)


def circular_median(values: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray(
        [
            np.median([values[(index + offset) % len(values)] for offset in range(-radius, radius + 1)])
            for index in range(len(values))
        ],
        dtype=float,
    )


def build_reference_segments(
    geometry: pd.DataFrame,
    strategy_step_m: float,
    nominal_speed_kph: float,
) -> pd.DataFrame:
    if strategy_step_m <= 0:
        raise ValueError("strategy_step_m must be positive.")
    total_length = float(geometry.attrs["track_length_m"])
    segment_count = max(4, int(math.ceil(total_length / strategy_step_m)))
    edges = np.linspace(0.0, total_length, segment_count + 1)
    rows = []
    distance = geometry["run_cumdist_m"].to_numpy(dtype=float)
    curvature = geometry["curvature_1_m"].to_numpy(dtype=float)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (distance >= lo) & (distance < hi)
        values = curvature[mask]
        if len(values) == 0:
            center = (lo + hi) * 0.5
            sample_index = int(np.argmin(np.abs(distance - center)))
            values = np.asarray([curvature[sample_index]])
        length_m = float(hi - lo)
        rows.append(
            {
                "segment": index,
                "dist_start_m": float(lo),
                "dist_end_m": float(hi),
                "length_m": length_m,
                "center_frac": float(((lo + hi) * 0.5) / total_length),
                "baseline_speed_kph": float(nominal_speed_kph),
                "baseline_grade_pct": 0.0,
                "baseline_curvature_1_m": float(np.quantile(values, 0.75)),
                "baseline_current_mA": 0.0,
                "baseline_power_w": 0.0,
                "baseline_energy_j": 0.0,
                "baseline_time_s": float(length_m / max(nominal_speed_kph / 3.6, 0.1)),
            }
        )
    return pd.DataFrame(rows)


def load_model_training_data(
    gps_path: str,
    telemetry_path: str,
    *,
    laps: int,
) -> pd.DataFrame:
    gps = read_gpx(gps_path)
    telemetry = read_telemetry(telemetry_path)
    gps_laps, telemetry_laps, _ = build_laps(
        gps,
        telemetry,
        laps=laps,
        split_method="start",
        start_time=None,
        time_offset_ms=0.0,
        tolerance_sec=1.5,
        lap_times=None,
    )
    rows = []
    for lap_gps, lap_telemetry in zip(gps_laps, telemetry_laps):
        if lap_gps.empty or lap_telemetry.empty:
            continue
        merged = merge_by_time(lap_telemetry, lap_gps, 1.5)
        derived = derive_motion_energy(merged)
        rows.append(derived)
    if not rows:
        raise ValueError("No recorded laps could be merged for model training.")
    training = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    training["run_cumdist_m"] = pd.to_numeric(training["dist_m"], errors="coerce").fillna(0.0).cumsum()
    return training


def make_transferable_model(model: dict[str, object]) -> dict[str, object]:
    """Neutralize the recorded circuit's absolute position feature."""
    transferable = dict(model)
    for key in ("current_coeffs", "power_coeffs"):
        coefficients = np.asarray(model[key], dtype=float).copy()
        coefficients[0] += coefficients[7] * 0.5
        coefficients[7] = 0.0
        transferable[key] = coefficients
    transferable["transfer_position_neutralized"] = True
    return transferable


def map_profile_to_geometry(geometry: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    total_length = float(geometry.attrs["track_length_m"])
    closed = pd.concat([geometry, geometry.iloc[[0]].copy()], ignore_index=True)
    closed.loc[closed.index[-1], "run_cumdist_m"] = total_length
    edges = profile["dist_end_m"].to_numpy(dtype=float)
    indices = np.searchsorted(edges, closed["run_cumdist_m"].to_numpy(dtype=float), side="left")
    indices = np.clip(indices, 0, len(profile) - 1)
    mapped = profile.iloc[indices].reset_index(drop=True)
    output = closed[["run_cumdist_m", "lat", "lon", "x", "y", "curvature_1_m"]].copy()
    output = output.rename(columns={"run_cumdist_m": "distance_m"})
    for source, target in (
        ("segment", "segment"),
        ("target_speed_kph", "target_speed_kph"),
        ("action", "action"),
        ("pred_current_mA", "pred_current_mA"),
        ("pred_peak_current_mA", "pred_peak_current_mA"),
        ("pred_power_w", "pred_power_w"),
        ("throttle_duty", "throttle_duty"),
    ):
        output[target] = mapped[source].to_numpy()
    return output


def build_report(
    geometry: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    target_lap_time_sec: float,
    model_gpx: str,
    model_telemetry: str,
) -> str:
    length_m = float(geometry.attrs["track_length_m"])
    predicted_time = float(profile["segment_time_s"].sum())
    predicted_energy = float(profile["pred_energy_j"].sum())
    speeds = pd.to_numeric(profile["target_speed_kph"], errors="coerce")
    return "\n".join(
        [
            "=== Autodrome Chaudiere Transfer Strategy ===",
            "",
            "Objective: minimum predicted electrical energy subject to the lap-time,",
            "speed-change, motor, and fuse constraints.",
            "",
            f"Reference track length: {length_m:.1f} m",
            f"Lap-time constraint: {target_lap_time_sec:.1f} s",
            f"Predicted lap time: {predicted_time:.1f} s",
            f"Predicted lap energy: {predicted_energy:.1f} J",
            f"Predicted energy intensity: {predicted_energy / max(length_m / 1000.0, 1e-9):.1f} J/km",
            f"Target speed range: {speeds.min():.1f}-{speeds.max():.1f} km/h",
            f"Mean segment target: {speeds.mean():.1f} km/h",
            f"Accelerate/hold/coast segments: "
            f"{int((profile['action'] == 'accelerate').sum())}/"
            f"{int((profile['action'] == 'hold').sum())}/"
            f"{int((profile['action'] == 'coast').sum())}",
            "",
            f"Transferred model GPX: {model_gpx}",
            f"Transferred model telemetry: {model_telemetry}",
            "The recorded-circuit position coefficient was neutralized before transfer.",
            "",
            "Important: this is a model-derived initial strategy, not measured",
            "Autodrome telemetry. Refit and validate it after a real driven lap.",
        ]
    )


def write_preview(
    path: str | os.PathLike[str],
    samples: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, (map_ax, curve_ax) = plt.subplots(1, 2, figsize=(13, 6))
    points = samples[["x", "y"]].to_numpy(dtype=float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        linewidth=7,
        array=samples["target_speed_kph"].iloc[1:].to_numpy(dtype=float),
    )
    map_ax.add_collection(collection)
    map_ax.autoscale()
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_title("Autodrome target speed by position")
    map_ax.set_xlabel("East (m)")
    map_ax.set_ylabel("North (m)")
    map_ax.grid(True, linestyle="--", alpha=0.25)
    fig.colorbar(collection, ax=map_ax, label="Target speed (km/h)")

    curve_ax.step(
        profile["dist_start_m"],
        profile["target_speed_kph"],
        where="post",
        linewidth=2.5,
        color="#2563eb",
    )
    action_colors = {"accelerate": "#f97316", "hold": "#2563eb", "coast": "#16a34a"}
    for row in profile.itertuples(index=False):
        curve_ax.axvspan(
            row.dist_start_m,
            row.dist_end_m,
            color=action_colors.get(row.action, "#94a3b8"),
            alpha=0.12,
        )
    curve_ax.set_xlim(0.0, float(profile["dist_end_m"].iloc[-1]))
    curve_ax.set_title("Maximum-efficiency closed-loop speed curve")
    curve_ax.set_xlabel("Lap distance (m)")
    curve_ax.set_ylabel("Target speed (km/h)")
    curve_ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.target_lap_time_sec <= 0:
        raise ValueError("--target-lap-time-sec must be positive.")
    geometry = read_reference_geometry(args.reference_gpx)
    nominal_speed = float(geometry.attrs["track_length_m"] / args.target_lap_time_sec * 3.6)
    start_speed = args.start_speed_kph if args.start_speed_kph is not None else round(nominal_speed)
    segments = build_reference_segments(geometry, args.strategy_step_m, nominal_speed)
    training = load_model_training_data(args.model_gpx, args.model_telemetry, laps=args.model_laps)
    model = make_transferable_model(fit_empirical_energy_model(training))
    motor = build_motor_config(
        wheel_diameter_m=args.wheel_diameter_m,
        vehicle_mass_kg=args.vehicle_mass_kg,
        driver_mass_kg=args.driver_mass_kg,
        rolling_resistance_coeff=args.rolling_resistance_coeff,
        drivetrain_efficiency=args.drivetrain_efficiency,
        corner_drag_factor=args.corner_drag_factor,
        cd_area_m2=args.cd_area_m2,
    )
    profile = optimize_speed_profile(
        segments,
        model,
        time_budget_sec=args.target_lap_time_sec,
        speed_min_kph=args.speed_min_kph,
        speed_max_kph=args.speed_max_kph,
        max_delta_kph_per_segment=args.max_delta_kph_per_segment,
        speed_step_kph=args.speed_step_kph,
        fuse_current_ma=args.fuse_current_ma,
        fuse_max_duration_sec=args.fuse_max_duration_sec,
        motor_config=motor,
        start_speed_kph=start_speed,
        end_speed_kph=start_speed,
        end_speed_tolerance_kph=args.closed_loop_tolerance_kph,
    )
    profile.attrs["time_budget_sec"] = float(args.target_lap_time_sec)
    profile["time_budget_sec"] = float(args.target_lap_time_sec)
    samples = map_profile_to_geometry(geometry, profile)
    report = build_report(
        geometry,
        profile,
        target_lap_time_sec=args.target_lap_time_sec,
        model_gpx=args.model_gpx,
        model_telemetry=args.model_telemetry,
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    segments_path = Path(f"{prefix}-strategy-segments.csv")
    samples_path = Path(f"{prefix}-efficiency-strategy.csv")
    report_path = Path(f"{prefix}-strategy-report.txt")
    profile.to_csv(segments_path, index=False)
    samples.to_csv(samples_path, index=False)
    report_path.write_text(report + "\n", encoding="utf-8")
    if args.preview:
        write_preview(args.preview, samples, profile)

    print(report)
    print("")
    print(f"Wrote segments: {segments_path}")
    print(f"Wrote mapped curve: {samples_path}")
    print(f"Wrote report: {report_path}")
    if args.preview:
        print(f"Wrote preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
