"""Core helpers shared across the UTSM telemetry toolchain.

Extracted from gps_current_heatmap.py so that analyze_strategy.py (and
any future scripts) can import them without duplicating logic. 

Does this actually work? God only knows.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: "
        "pip install matplotlib pandas numpy"
    ) from exc

NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_gpx(gpx_path: str) -> pd.DataFrame:
    """Parse a GPX file into a DataFrame with lat, lon, elev, time columns."""
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
    """Read a telemetry CSV and coerce / validate expected columns."""
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


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def add_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Add local flat-earth X/Y columns (metres) to a GPS DataFrame."""
    lat0 = df["lat"].iloc[0]
    lon0 = df["lon"].iloc[0]
    avg_lat_rad = np.deg2rad(df["lat"].mean())
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * np.cos(avg_lat_rad)
    df = df.copy()
    df["x"] = (df["lon"] - lon0) * meters_per_deg_lon
    df["y"] = (df["lat"] - lat0) * meters_per_deg_lat
    return df


def compute_distance(df: pd.DataFrame) -> float:
    """Total track distance in metres using flat-earth XY."""
    coords = add_xy(df)[["x", "y"]].to_numpy()
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(coords[1:] - coords[:-1], axis=1).sum())


# ---------------------------------------------------------------------------
# Lap detection
# ---------------------------------------------------------------------------

def find_start_spike(telemetry: pd.DataFrame, threshold_mA: float = 10000.0) -> int:
    """Return the index of the first telemetry row whose current exceeds threshold."""
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
    """Detect lap boundaries by counting Y-line crossings around start point."""
    gps_xy = add_xy(gps)
    y_start = float(gps_xy.loc[start_index, "y"])
    y = gps_xy["y"].to_numpy()

    xy = gps_xy[["x", "y"]].to_numpy()
    seg_dists = np.zeros(len(xy))
    seg_dists[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    cum_dist = np.cumsum(seg_dists)

    boundaries = [start_index]
    current_lap_start_idx = start_index

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

    candidates = []
    for y_line in np.linspace(y_min, y_max, 301):
        all_crossings = count_line_crossings(y, y_line, width)
        if not all_crossings:
            continue
        band = np.abs(y - y_line) <= width
        x_range = float(x[band].max() - x[band].min()) if np.any(band) else 0.0
        for direction in ["down", "up"]:
            filtered = [idx for idx, d in all_crossings if d == direction]
            candidates.append((y_line, direction, filtered, x_range))

    exact_candidates = [c for c in candidates if len(c[2]) == laps]
    if exact_candidates:
        best = max(exact_candidates, key=lambda item: (item[3], -abs(item[0] - y[0])))
        return best[0], best[2]

    best = min(
        candidates,
        key=lambda item: (abs(len(item[2]) - laps), abs(item[0] - y[0]), -item[3]),
    )
    return best[0], best[2]


def split_gps_into_laps(df: pd.DataFrame, laps: int, method: str = "points") -> list[pd.DataFrame]:
    if laps <= 1:
        return [df]

    if method == "line":
        y_line, crossings = detect_lap_line(df, laps)
        if len(crossings) < min(laps, 2):
            print("Warning: Failed to detect a clear lap line. Falling back to point-based lap splitting.")
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


# ---------------------------------------------------------------------------
# Alignment and merging
# ---------------------------------------------------------------------------

def align_telemetry(
    telemetry: pd.DataFrame,
    gps: pd.DataFrame,
    start_time: "str | pd.Timestamp | None",
    offset_ms: float,
) -> pd.DataFrame:
    if start_time is not None:
        telemetry_start = start_time if isinstance(start_time, pd.Timestamp) else parse_iso8601(start_time)
    else:
        telemetry_start = gps["time"].iloc[0]

    telemetry = telemetry.copy()
    telemetry["time"] = telemetry_start + pd.to_timedelta(
        telemetry["timestamp_ms"] + offset_ms, unit="ms"
    )
    return telemetry


def merge_by_time(
    telemetry: pd.DataFrame, gps: pd.DataFrame, tolerance_sec: float
) -> pd.DataFrame:
    """Nearest-time join between telemetry rows and GPS track points."""
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


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

def derive_motion_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Add dt_s, dist_m, elev_diff_m, speed_m_s, speed_kph, grade_pct,
    power_w, energy_wh, and cumdist_m columns to a merged lap DataFrame.

    Expects columns: time, lat, lon, elev, current_mA, voltage_mV.
    """
    df = add_xy(df.copy())
    df = df.sort_values("time").reset_index(drop=True)

    # Time delta
    times = pd.to_datetime(df["time"])
    df["dt_s"] = times.diff().dt.total_seconds().fillna(0.0).clip(lower=0)

    # Point-to-point distance
    xy = df[["x", "y"]].to_numpy()
    seg = np.zeros(len(xy))
    seg[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    df["dist_m"] = seg

    # Elevation change
    elev = pd.to_numeric(df["elev"], errors="coerce").fillna(0.0).to_numpy()
    elev_diff = np.zeros(len(elev))
    elev_diff[1:] = np.diff(elev)
    df["elev_diff_m"] = elev_diff

    # Speed
    df["speed_m_s"] = np.where(df["dt_s"] > 0, df["dist_m"] / df["dt_s"], 0.0)
    df["speed_kph"] = df["speed_m_s"] * 3.6

    # Grade (%)
    df["grade_pct"] = np.where(
        df["dist_m"] > 0.01,
        (df["elev_diff_m"] / df["dist_m"]) * 100.0,
        0.0,
    )

    # Power and energy
    df["power_w"] = (df["current_mA"].abs() / 1000.0) * (df["voltage_mV"] / 1000.0)
    df["energy_wh"] = df["power_w"] * df["dt_s"] / 3600.0

    # Cumulative distance through the lap
    df["cumdist_m"] = df["dist_m"].cumsum()

    return df


# ---------------------------------------------------------------------------
# Lap stats (basic, used by heatmap script)
# ---------------------------------------------------------------------------

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
