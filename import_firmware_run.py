"""Split firmware CSV exports into analysis-ready telemetry and GPX runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from utsm_telemetry import add_xy, compute_distance, find_lap_boundaries_by_start_gate


GPS_COLUMNS = {
    "gps_location_valid",
    "gps_lat",
    "gps_long",
    "gps_alt_m",
    "gps_speed_kmph",
    "gps_hdop",
    "gps_sats",
    "gps_time_valid",
    "gps_time_utc",
    "gps_centisecond",
}
REQUIRED_COLUMNS = {
    "timestamp_ms",
    "current_mA",
    "voltage_mV",
    "temperature_C",
    "temperature_valid",
    "wheel_speed_valid",
    "wheel_speed_kmph",
    "ax_x100",
    "ay_x100",
    "az_x100",
    "gps_location_valid",
    "gps_lat",
    "gps_long",
    "gps_time_valid",
    "gps_time_utc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import one firmware CSV, a folder of split CSVs, or a ZIP. GPS "
            "columns become GPX and the remaining temperature/speed telemetry "
            "becomes the dashboard CSV."
        )
    )
    parser.add_argument("input", help="Raw .csv, directory, or .zip")
    parser.add_argument("--output-dir", default=os.path.join("data", "runs"))
    parser.add_argument("--name", default="front-campus")
    parser.add_argument("--date", help="UTC date (YYYY-MM-DD); inferred from input name")
    parser.add_argument("--session-gap-sec", type=float, default=60.0)
    parser.add_argument("--min-session-distance-m", type=float, default=250.0)
    return parser.parse_args()


def infer_date(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not match:
        raise ValueError("Could not infer the UTC date; pass --date YYYY-MM-DD.")
    return match.group(1)


def csv_paths(input_path: Path, extracted_dir: Path | None = None) -> list[Path]:
    if input_path.suffix.lower() == ".zip":
        if extracted_dir is None:
            raise ValueError("A ZIP extraction directory is required.")
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(extracted_dir)
        return sorted(extracted_dir.rglob("*.csv"))
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def read_raw_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_ms"] = pd.to_numeric(combined["timestamp_ms"], errors="raise")
    return (
        combined.sort_values("timestamp_ms")
        .drop_duplicates(subset=["timestamp_ms"], keep="last")
        .reset_index(drop=True)
    )


def add_gps_datetimes(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    frame = frame.copy()
    centiseconds = pd.to_numeric(frame.get("gps_centisecond", 0), errors="coerce").fillna(0)
    base = pd.to_datetime(date + " " + frame["gps_time_utc"].astype(str), utc=True, errors="coerce")
    frame["gps_datetime"] = base + pd.to_timedelta(centiseconds * 10, unit="ms")
    if frame["gps_datetime"].isna().any():
        raise ValueError("One or more gps_time_utc values could not be parsed.")
    # Accommodate a recording that crosses UTC midnight.
    day_rollover = frame["gps_datetime"].diff() < pd.Timedelta(hours=-12)
    frame["gps_datetime"] += pd.to_timedelta(day_rollover.cumsum(), unit="D")
    return frame


def session_frames(frame: pd.DataFrame, gap_sec: float) -> list[pd.DataFrame]:
    session_id = (frame["timestamp_ms"].diff() > gap_sec * 1000).cumsum()
    return [group.reset_index(drop=True) for _, group in frame.groupby(session_id, sort=True)]


def gps_frame(session: pd.DataFrame) -> pd.DataFrame:
    valid = (
        pd.to_numeric(session["gps_location_valid"], errors="coerce").fillna(0).astype(bool)
        & pd.to_numeric(session["gps_time_valid"], errors="coerce").fillna(0).astype(bool)
    )
    gps = session.loc[valid, ["gps_lat", "gps_long", "gps_alt_m", "gps_datetime"]].copy()
    gps.columns = ["lat", "lon", "elev", "time"]
    for column in ("lat", "lon", "elev"):
        gps[column] = pd.to_numeric(gps[column], errors="coerce")
    return gps.dropna(subset=["lat", "lon", "time"]).drop_duplicates("time").reset_index(drop=True)


def lap_boundaries(gps: pd.DataFrame) -> list[int]:
    if len(gps) < 3:
        return [0]
    gps_xy = add_xy(gps)
    x_span = float(gps_xy["x"].max() - gps_xy["x"].min())
    y_span = float(gps_xy["y"].max() - gps_xy["y"].min())
    estimated_lap_m = max(200.0, x_span + y_span)
    return find_lap_boundaries_by_start_gate(
        gps,
        0,
        laps=50,
        y_band_width=max(10.0, min(20.0, y_span * 0.10)),
        x_window_width=max(40.0, min(80.0, x_span * 0.40)),
        min_gap_points=5,
        min_lap_distance_m=estimated_lap_m,
        pre_race_max_distance_m=estimated_lap_m * 0.6,
    )


def write_gpx(gps: pd.DataFrame, path: Path, name: str) -> None:
    ET.register_namespace("", "http://www.topografix.com/GPX/1/1")
    namespace = "http://www.topografix.com/GPX/1/1"
    root = ET.Element(f"{{{namespace}}}gpx", {"version": "1.1", "creator": "UTSM proto telemetry"})
    track = ET.SubElement(root, f"{{{namespace}}}trk")
    ET.SubElement(track, f"{{{namespace}}}name").text = name
    segment = ET.SubElement(track, f"{{{namespace}}}trkseg")
    for row in gps.itertuples(index=False):
        point = ET.SubElement(
            segment,
            f"{{{namespace}}}trkpt",
            {"lat": f"{row.lat:.7f}", "lon": f"{row.lon:.7f}"},
        )
        if math.isfinite(float(row.elev)):
            ET.SubElement(point, f"{{{namespace}}}ele").text = f"{row.elev:.2f}"
        timestamp = pd.Timestamp(row.time).isoformat().replace("+00:00", "Z")
        ET.SubElement(point, f"{{{namespace}}}time").text = timestamp
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def telemetry_frame(session: pd.DataFrame) -> pd.DataFrame:
    split_columns = GPS_COLUMNS | {"gps_datetime"}
    telemetry = session.drop(columns=[column for column in split_columns if column in session.columns]).copy()
    telemetry = telemetry.rename(columns={
        "timestamp_ms": "source_timestamp_ms",
        "temperature_C": "motor_temperature_C",
        "temperature_valid": "motor_temperature_valid",
        "wheel_speed_kmph": "wheel_speed_kph",
    })
    telemetry.insert(
        0,
        "timestamp_ms",
        telemetry["source_timestamp_ms"] - telemetry["source_timestamp_ms"].iloc[0],
    )
    return telemetry


def import_runs(args: argparse.Namespace) -> list[Path]:
    source = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    date = infer_date(source, args.date)
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="utsm-firmware-import-") as temp:
        raw = read_raw_files(csv_paths(source, Path(temp)))
        raw = add_gps_datetimes(raw, date)
        candidates = session_frames(raw, args.session_gap_sec)
        accepted: list[tuple[pd.DataFrame, pd.DataFrame, list[int]]] = []
        for session in candidates:
            gps = gps_frame(session)
            if len(gps) < 2 or compute_distance(gps) < args.min_session_distance_m:
                continue
            boundaries = lap_boundaries(gps)
            if len(boundaries) < 2:
                continue
            accepted.append((session, gps, boundaries))

        for index, (session, gps, boundaries) in enumerate(accepted, start=1):
            run_name = f"{args.name}-{date}-run-{index:02d}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            write_gpx(gps, run_dir / f"{run_name}.gpx", run_name)
            telemetry_frame(session).to_csv(run_dir / f"{run_name}-telemetry.csv", index=False)
            manifest = {
                "label": f"{args.name.replace('-', ' ').title()} {date} Run {index}",
                "laps": len(boundaries) - 1,
                "split_method": "gate",
                "source": source.name,
                "source_files": sorted(session["source_file"].unique().tolist()),
                "source_timestamp_ms": [
                    int(session["timestamp_ms"].iloc[0]),
                    int(session["timestamp_ms"].iloc[-1]),
                ],
                "distance_m": round(compute_distance(gps), 1),
                "lap_boundary_gpx_indices": boundaries,
            }
            (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            written.append(run_dir)
    return written


def main() -> None:
    args = parse_args()
    runs = import_runs(args)
    if not runs:
        raise SystemExit("No complete sessions met the distance and lap criteria.")
    for run in runs:
        print(f"Wrote {run}")


if __name__ == "__main__":
    main()
