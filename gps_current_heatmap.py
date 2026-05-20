import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install matplotlib pandas numpy"
    )

NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align telemetry current data with a GPX track and render a current heatmap."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
    parser.add_argument(
        "--output",
        "-o",
        default="current_heatmap.png",
        help="Output image file for the heatmap",
    )
    parser.add_argument(
        "--metric",
        choices=["current", "accel", "magnitude"],
        default="current",
        help="Metric to color the track by",
    )
    parser.add_argument(
        "--start-time",
        help=(
            "Force the telemetry start time to align with this GPS time. "
            "Use ISO 8601 (e.g. 2026-04-11T14:29:07Z). "
            "If omitted, telemetry row 0 is aligned to the first GPS point time."
        ),
    )
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Add a millisecond offset to the telemetry timestamps after aligning start. "
            "Useful when the telemetry stream begins slightly before or after the GPS track."
        ),
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=1.5,
        help="Maximum seconds to tolerate when matching telemetry rows to GPS track points.",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Number of laps to split the GPS track into and render separately.",
    )
    parser.add_argument(
        "--split-method",
        choices=["points", "time", "line", "start"],
        default="start",
        help=(
            "How to split the GPS track into laps: by equal point count, equal time segments, "
            "by crossing a line, or starting at the first current spike and returning to that point."
        ),
    )
    parser.add_argument(
        "--lap-times",
        nargs="+",
        metavar="ELAPSED",
        help=(
            "Elapsed time from the start of the Strava/GPX activity for each lap start. "
            "Use MM:SS or H:MM:SS (e.g. 11:20 or 1:02:30). Provide one per lap. "
            "The first time is also used to anchor telemetry: "
            "the first 10 A current spike is aligned to this point."
        ),
    )
    return parser.parse_args()


def parse_iso8601(timestamp: str) -> pd.Timestamp:
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    return pd.to_datetime(timestamp)


def parse_lap_time(value: str, track_start: pd.Timestamp) -> pd.Timestamp:
    """Accept MM:SS or H:MM:SS elapsed time and return an absolute timestamp."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(
            f"Unrecognised lap time format '{value}'. Use MM:SS or H:MM:SS."
        )
    return track_start + pd.Timedelta(hours=h, minutes=m, seconds=s)


def read_gpx(gpx_path: str) -> pd.DataFrame:
    if not os.path.exists(gpx_path):
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")

    tree = ET.parse(gpx_path)
    root = tree.getroot()
    points = []
    for trkseg in root.findall("gpx:trk/gpx:trkseg", NAMESPACE):
        for trkpt in trkseg.findall("gpx:trkpt", NAMESPACE):
            lat = float(trkpt.attrib["lat"])
            lon = float(trkpt.attrib["lon"])
            ele_node = trkpt.find("gpx:ele", NAMESPACE)
            time_node = trkpt.find("gpx:time", NAMESPACE)
            elev = float(ele_node.text) if ele_node is not None else math.nan
            if time_node is None or not time_node.text:
                raise ValueError("GPX points must contain <time> values")
            time = parse_iso8601(time_node.text)
            points.append({"lat": lat, "lon": lon, "elev": elev, "time": time})

    if not points:
        raise ValueError("No track points found in GPX file")

    df = pd.DataFrame(points)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def read_telemetry(telemetry_path: str) -> pd.DataFrame:
    if not os.path.exists(telemetry_path):
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    df = pd.read_csv(telemetry_path)
    expected = {"timestamp_ms", "current_mA", "voltage_mV", "ax_x100", "ay_x100", "az_x100"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Telemetry CSV must contain columns: {', '.join(sorted(expected))}. "
            f"Found: {', '.join(df.columns)}"
        )

    df = df.copy()
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    if df["timestamp_ms"].isna().any():
        bad = df["timestamp_ms"].isna().sum()
        print(f"WARNING: Dropping {bad} telemetry rows with invalid timestamp_ms values.")
        df = df.dropna(subset=["timestamp_ms"]).reset_index(drop=True)

    df["current_mA"] = pd.to_numeric(df["current_mA"], errors="coerce").abs()
    df["voltage_mV"] = pd.to_numeric(df["voltage_mV"], errors="coerce")
    df["ax_x100"] = pd.to_numeric(df["ax_x100"], errors="coerce")
    df["ay_x100"] = pd.to_numeric(df["ay_x100"], errors="coerce")
    df["az_x100"] = pd.to_numeric(df["az_x100"], errors="coerce")
    df["accel_m_s2"] = np.sqrt(
        (df["ax_x100"] / 100.0) ** 2
        + (df["ay_x100"] / 100.0) ** 2
        + (df["az_x100"] / 100.0) ** 2
    )
    df["accel_mag"] = np.sqrt(
        (df["ax_x100"].abs() / 100.0) ** 2
        + (df["ay_x100"].abs() / 100.0) ** 2
        + (df["az_x100"].abs() / 100.0) ** 2
    )
    return df


def add_xy(df: pd.DataFrame) -> pd.DataFrame:
    lat0 = df["lat"].iloc[0]
    lon0 = df["lon"].iloc[0]
    avg_lat_rad = np.deg2rad(df["lat"].mean())
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * np.cos(avg_lat_rad)
    df = df.copy()
    df["x"] = (df["lon"] - lon0) * meters_per_deg_lon
    df["y"] = (df["lat"] - lat0) * meters_per_deg_lat
    return df


def find_start_spike(telemetry: pd.DataFrame, threshold_mA: float = 10000.0) -> int:
    spikes = telemetry[telemetry["current_mA"] >= threshold_mA]
    if spikes.empty:
        raise ValueError(f"No current spike over {threshold_mA} mA found in telemetry.")
    return int(spikes.index[0])


def find_nearest_gps_index(gps: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    times = pd.to_datetime(gps["time"])
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp)
    diffs = (times - timestamp).abs()
    return int(diffs.idxmin())


def find_lap_boundaries_by_y_crossing(
    gps: pd.DataFrame,
    start_index: int,
    laps: int,
    y_band_width: float = 5.0,
    min_gap_points: int = 50,
    min_lap_distance_m: float = 1000.0,
) -> list[int]:
    """Detect lap boundaries by counting Y-line crossings.

    The Y-line is the Y coordinate of the start GPS point.  Every two times
    the track re-enters the band around that Y value, one full lap has been
    completed and a new boundary is recorded.

    If a detected segment is shorter than min_lap_distance_m it is treated as
    pre-race/paddock movement: the start pointer advances to that crossing and
    counting restarts from there so the first real lap is found correctly.
    """
    gps_xy = add_xy(gps)
    y_start = float(gps_xy.loc[start_index, "y"])
    y = gps_xy["y"].to_numpy()

    # Precompute cumulative distance so we can measure each segment cheaply.
    xy = gps_xy[["x", "y"]].to_numpy()
    seg_dists = np.zeros(len(xy))
    seg_dists[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    cum_dist = np.cumsum(seg_dists)

    boundaries = [start_index]
    current_lap_start_idx = start_index

    # Escape the starting band first so the initial position doesn't count.
    escaped = False
    in_band = False
    crossing_count = 0

    for idx in range(start_index + 1, len(y)):
        near = abs(y[idx] - y_start) <= y_band_width

        if not escaped:
            if not near:
                escaped = True
                in_band = False
            continue

        if near and not in_band:
            in_band = True
            if idx - current_lap_start_idx >= min_gap_points:
                crossing_count += 1
                if crossing_count % 2 == 0:
                    lap_dist = cum_dist[idx] - cum_dist[current_lap_start_idx]
                    if lap_dist < min_lap_distance_m:
                        # Too short — paddock/warmup movement, not a real lap.
                        # Advance the start pointer here and restart counting.
                        print(
                            f"  Skipping short segment ({lap_dist:.0f}m) at GPS index {idx}"
                            " — treated as pre-race movement."
                        )
                        boundaries[-1] = idx
                        current_lap_start_idx = idx
                        crossing_count = 0
                    else:
                        boundaries.append(idx)
                        current_lap_start_idx = idx
                        if len(boundaries) >= laps + 1:
                            break
        elif not near:
            in_band = False

    print(
        f"Y-line crossing detection: y_start={y_start:.1f}m, "
        f"found {len(boundaries) - 1} lap boundaries (wanted {laps})."
    )
    return boundaries


def count_line_crossings(y: np.ndarray, y_line: float, width: float) -> list[tuple[int, str]]:
    crossings = []
    in_band = np.abs(y - y_line) <= width
    outside = not bool(in_band[0])
    for idx in range(1, len(y)):
        if outside and in_band[idx]:
            prev = y[idx - 1]
            cur = y[idx]
            direction = "up" if cur > prev else "down"
            crossings.append((idx, direction))
            outside = False
        elif not in_band[idx]:
            outside = True
    return crossings


def detect_lap_line(df: pd.DataFrame, laps: int, width: float = 2.0) -> tuple[float, list[int]]:
    df = add_xy(df)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    y_min = float(df["y"].min())
    y_max = float(df["y"].max())

    best_match = None
    candidates = []
    for y_line in np.linspace(y_min, y_max, 301):
        all_crossings = count_line_crossings(y, y_line, width)
        if not all_crossings:
            continue
        band = np.abs(y - y_line) <= width
        x_range = float(x[band].max() - x[band].min()) if np.any(band) else 0.0
        for direction in ["down", "up"]:
            filtered = [idx for idx, dir in all_crossings if dir == direction]
            candidates.append((y_line, direction, filtered, x_range))

    exact_candidates = [c for c in candidates if len(c[2]) == laps]
    if exact_candidates:
        best_match = max(exact_candidates, key=lambda item: (item[3], -abs(item[0] - y[0])))
        return best_match[0], best_match[2]

    best_match = min(
        candidates,
        key=lambda item: (abs(len(item[2]) - laps), abs(item[0] - y[0]), -item[3]),
    )
    return best_match[0], best_match[2]


def compute_distance(df: pd.DataFrame) -> float:
    coords = add_xy(df)[["x", "y"]].to_numpy()
    if len(coords) < 2:
        return 0.0
    dist = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    return float(dist.sum())


def compute_lap_stats(df: pd.DataFrame) -> dict[str, float]:
    stats = {
        "duration_s": (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds(),
        "points": len(df),
        "distance_m": compute_distance(df),
        "avg_current_mA": float(df["current_mA"].mean(skipna=True)),
        "max_current_mA": float(df["current_mA"].max(skipna=True)),
        "min_current_mA": float(df["current_mA"].min(skipna=True)),
        "avg_accel_m_s2": float(df["accel_m_s2"].mean(skipna=True)),
        "max_accel_m_s2": float(df["accel_m_s2"].max(skipna=True)),
    }
    if stats["duration_s"] > 0:
        stats["avg_speed_m_s"] = stats["distance_m"] / stats["duration_s"]
    else:
        stats["avg_speed_m_s"] = 0.0
    return stats


def split_gps_into_laps(df: pd.DataFrame, laps: int, method: str = "points") -> list[pd.DataFrame]:
    if laps <= 1:
        return [df]

    if method == "line":
        y_line, crossings = detect_lap_line(df, laps)
        if len(crossings) < min(laps, 2):
            print(
                "Warning: Failed to detect a clear lap line. Falling back to point-based lap splitting."
            )
            method = "points"
        else:
            print(f"Detected lap line at y={y_line:.1f} and {len(crossings)} crossings")
            segments = []
            start_idx = 0
            for end_idx in crossings[:laps]:
                segments.append(df.iloc[start_idx:end_idx].reset_index(drop=True))
                start_idx = end_idx
            if len(segments) < laps and start_idx < len(df):
                segments.append(df.iloc[start_idx:].reset_index(drop=True))
            return segments

    if method == "start":
        raise ValueError("The 'start' split method must be handled after GPS/telemetry start alignment.")

    if method == "time":
        start = df["time"].iloc[0]
        end = df["time"].iloc[-1]
        total = (end - start).total_seconds()
        segments = []
        for i in range(laps):
            lap_start = start + pd.Timedelta(seconds=(total * i) / laps)
            lap_end = start + pd.Timedelta(seconds=(total * (i + 1)) / laps)
            lap_df = df[(df["time"] >= lap_start) & (df["time"] <= lap_end)].copy()
            if not lap_df.empty:
                segments.append(lap_df.reset_index(drop=True))
        return segments

    # Default: split by equal point count
    n = len(df)
    segments = []
    base = n // laps
    remainder = n % laps
    start = 0
    for i in range(laps):
        size = base + (1 if i < remainder else 0)
        end = start + size
        segments.append(df.iloc[start:end].reset_index(drop=True))
        start = end
    return segments


def align_telemetry(telemetry: pd.DataFrame, gps: pd.DataFrame, start_time: str | pd.Timestamp | None, offset_ms: float) -> pd.DataFrame:
    if start_time is not None:
        telemetry_start = start_time if isinstance(start_time, pd.Timestamp) else parse_iso8601(start_time)
    else:
        telemetry_start = gps["time"].iloc[0]

    telemetry = telemetry.copy()
    telemetry["time"] = telemetry_start + pd.to_timedelta(telemetry["timestamp_ms"] + offset_ms, unit="ms")
    return telemetry


def merge_by_time(telemetry: pd.DataFrame, gps: pd.DataFrame, tolerance_sec: float) -> pd.DataFrame:
    telemetry = telemetry.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    gps = gps.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if telemetry.empty:
        raise ValueError("Telemetry contains no valid time values after alignment.")

    merged = pd.merge_asof(
        telemetry,
        gps,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
    )
    merged = merged.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError(
            "No telemetry rows could be matched to GPS track points within the tolerance. "
            "Try increasing --tolerance-sec or verifying the time alignment."
        )
    return merged


def format_output_path(output: str, lap_index: int) -> str:
    base, ext = os.path.splitext(output)
    if not ext:
        ext = ".png"
    return f"{base}_lap{lap_index}{ext}"


def plot_heatmap(df: pd.DataFrame, color_column: str, output_path: str, lap_index: int | None = None, stats: dict | None = None) -> None:
    df = add_xy(df)
    coords = df[["x", "y"]].to_numpy()
    if len(coords) < 2:
        raise ValueError("Not enough merged points to draw a track")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    values = df[color_column].to_numpy()
    norm = plt.Normalize(np.nanmin(values), np.nanmax(values))
    cmap = "plasma" if color_column == "current_mA" else "viridis"

    fig, ax = plt.subplots(figsize=(10, 8))
    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=3,
        capstyle="round",
    )
    line_collection.set_array(values[:-1])
    ax.add_collection(line_collection)
    ax.autoscale()
    cbar = fig.colorbar(line_collection, ax=ax, pad=0.02)
    cbar.set_label(color_column)

    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color="green", marker="o", s=80, label="Start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", marker="X", s=80, label="End")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    title = f"Lap {lap_index} track heatmap" if lap_index is not None else "Track heatmap"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if stats is not None:
        stat_text = (
            f"Duration: {stats['duration_s']:.1f}s\n"
            f"Distance: {stats['distance_m']:.1f}m\n"
            f"Avg current: {stats['avg_current_mA']:.1f}mA, max: {stats['max_current_mA']:.1f}mA\n"
            f"Avg accel: {stats['avg_accel_m_s2']:.2f}m/s², max: {stats['max_accel_m_s2']:.2f}m/s²"
        )
        ax.text(
            0.98,
            0.02,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray", "boxstyle": "round,pad=0.3"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved heatmap to: {output_path}")


def main() -> int:
    args = parse_args()

    gps_df = read_gpx(args.gps)
    telem_df = read_telemetry(args.telemetry)

    if args.metric == "current":
        color_column = "current_mA"
    elif args.metric == "accel":
        color_column = "accel_m_s2"
    else:
        color_column = "accel_mag"

    # --- Manual lap times mode ---
    if args.lap_times:
        track_start = gps_df["time"].iloc[0]
        lap_timestamps = [parse_lap_time(t, track_start) for t in args.lap_times]
        if len(lap_timestamps) < 2:
            raise ValueError("--lap-times requires at least 2 times: one lap start and a finish time.")
        print(f"GPX track starts at {track_start}. Lap boundaries resolved to:")
        for i, ts in enumerate(lap_timestamps):
            label = f"Lap {i + 1} start" if i < len(lap_timestamps) - 1 else "Finish"
            print(f"  {label}: {args.lap_times[i]} \u2192 {ts}")

        # Align telemetry: first 10A spike maps to lap 1 start time.
        spike_idx = find_start_spike(telem_df)
        spike_ms = float(telem_df.loc[spike_idx, "timestamp_ms"])
        telemetry_start = lap_timestamps[0] - pd.Timedelta(milliseconds=spike_ms)
        print(
            f"Telemetry spike at row {spike_idx} (timestamp_ms={spike_ms:.0f}). "
            f"Aligning to lap 1 start: {lap_timestamps[0]}. "
            f"Computed telemetry epoch: {telemetry_start}."
        )
        telem_df = align_telemetry(telem_df, gps_df, telemetry_start, args.time_offset_ms)

        # N timestamps define N-1 laps; last timestamp is the finish cap.
        laps = []
        for i in range(len(lap_timestamps) - 1):
            lap_start = lap_timestamps[i]
            lap_end = lap_timestamps[i + 1]
            lap_gps = gps_df[(gps_df["time"] >= lap_start) & (gps_df["time"] < lap_end)].copy().reset_index(drop=True)
            laps.append(lap_gps)
            print(f"Lap {i + 1}: GPS points={len(lap_gps)} ({lap_start} \u2192 {lap_end})")

    elif args.split_method == "start":
        telem_df = align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)
        start_idx = find_start_spike(telem_df)
        start_time = telem_df.loc[start_idx, "time"]
        gps_start_idx = find_nearest_gps_index(gps_df, start_time)
        print(
            f"Start spike at telemetry index {start_idx}, time {start_time}, "
            f"matching GPS index {gps_start_idx}."
        )
        gps_df = gps_df.loc[gps_start_idx:].reset_index(drop=True)
        lap_boundaries = find_lap_boundaries_by_y_crossing(gps_df, 0, args.laps)
        if len(lap_boundaries) < args.laps + 1:
            print(
                f"Warning: only found {len(lap_boundaries) - 1} complete lap(s) via Y-line crossings "
                f"(wanted {args.laps}). Remaining laps will use the final segment."
            )
        laps = []
        for i in range(min(len(lap_boundaries) - 1, args.laps)):
            laps.append(gps_df.iloc[lap_boundaries[i] : lap_boundaries[i + 1]].reset_index(drop=True))
        if len(laps) < args.laps and lap_boundaries:
            laps.append(gps_df.iloc[lap_boundaries[-1] :].reset_index(drop=True))
        if not laps:
            raise ValueError("Could not create any laps from the start point detection.")
    else:
        telem_df = align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)
        laps = split_gps_into_laps(gps_df, args.laps, args.split_method)

    for idx, lap_gps in enumerate(laps, start=1):
        if lap_gps.empty:
            print(f"Skipping empty lap {idx}")
            continue

        lap_start = lap_gps["time"].iloc[0]
        lap_end = lap_gps["time"].iloc[-1]
        lap_telem = telem_df[(telem_df["time"] >= lap_start - pd.Timedelta(seconds=args.tolerance_sec))
                             & (telem_df["time"] <= lap_end + pd.Timedelta(seconds=args.tolerance_sec))].copy()

        if lap_telem.empty:
            print(f"Skipping lap {idx}: no telemetry rows fall within lap {idx} time range.")
            continue

        merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        stats = compute_lap_stats(merged)
        print(
            f"Lap {idx}: duration={stats['duration_s']:.1f}s, distance={stats['distance_m']:.1f}m, "
            f"avg_current={stats['avg_current_mA']:.1f}mA, max_current={stats['max_current_mA']:.1f}mA, "
            f"avg_accel={stats['avg_accel_m_s2']:.2f}m/s², max_accel={stats['max_accel_m_s2']:.2f}m/s²"
        )

        output_path = format_output_path(args.output, idx) if len(laps) > 1 else args.output
        plot_heatmap(merged, color_column, output_path, lap_index=idx, stats=stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
