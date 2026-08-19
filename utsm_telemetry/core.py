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
GRAVITY_M_S2 = 9.80665
FORWARD_AXIS_CHOICES = ("ax", "ay", "az", "neg_ax", "neg_ay", "neg_az")


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
    # Firmware CSVs historically used ``kmph`` and a generic temperature
    # label, while the live API uses ``kph`` and ``motor_temperature``.  Make
    # both generations available through one canonical analysis schema.
    aliases = {
        "wheel_speed_kmph": "wheel_speed_kph",
        "temperature_C": "motor_temperature_C",
        "temperature_valid": "motor_temperature_valid",
    }
    for source, destination in aliases.items():
        if destination not in df.columns and source in df.columns:
            df[destination] = df[source]

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    if df["timestamp_ms"].isna().any():
        bad = df["timestamp_ms"].isna().sum()
        print(f"WARNING: Dropping {bad} telemetry rows with invalid timestamp_ms values.")
        df = df.dropna(subset=["timestamp_ms"]).reset_index(drop=True)

    source_current = pd.to_numeric(df["current_mA"], errors="coerce")
    # Keep the signed source measurement available for audits. Historical
    # analysis treats current as a magnitude, but model-specific calibration
    # must not destroy the value that was actually written by firmware.
    df["source_current_mA"] = source_current
    df["current_mA"] = source_current.abs()
    df["voltage_mV"] = pd.to_numeric(df["voltage_mV"], errors="coerce")
    df["ax_x100"] = pd.to_numeric(df["ax_x100"], errors="coerce")
    df["ay_x100"] = pd.to_numeric(df["ay_x100"], errors="coerce")
    df["az_x100"] = pd.to_numeric(df["az_x100"], errors="coerce")
    if "amag_x100" in df.columns:
        df["amag_x100"] = pd.to_numeric(df["amag_x100"], errors="coerce")
    for column in ("motor_temperature_C", "wheel_speed_kph", "wheel_rpm"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for valid_column, value_column in (
        ("motor_temperature_valid", "motor_temperature_C"),
        ("wheel_speed_valid", "wheel_speed_kph"),
    ):
        if valid_column in df.columns and value_column in df.columns:
            valid = pd.to_numeric(df[valid_column], errors="coerce").fillna(0).astype(bool)
            df[valid_column] = valid
            df.loc[~valid, value_column] = np.nan
    return derive_acceleration_features(df)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pre-processing  (V2)
# ---------------------------------------------------------------------------

def clean_telemetry(
    df: pd.DataFrame,
    *,
    current_max_mA: float = 60_000.0,
    voltage_min_mV: float = 18_000.0,
    voltage_max_mV: float = 30_000.0,
    imu_spike_zscore: float = 5.0,
    gap_fill_method: str = "linear",
) -> pd.DataFrame:
    """Clean raw telemetry: remove spikes and fill gaps.
    """
    df = df.copy().sort_values("timestamp_ms").reset_index(drop=True)
    log: dict[str, int] = {}

    # 1. Duplicate timestamps
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp_ms"], keep="last").reset_index(drop=True)
    log["duplicate_rows_removed"] = n_before - len(df)

    sensor_cols = [c for c in ("current_mA", "voltage_mV", "ax_x100", "ay_x100", "az_x100", "amag_x100") if c in df.columns]

    # 2. Current spikes
    cur = pd.to_numeric(df["current_mA"], errors="coerce")
    spike_cur = cur.abs() > current_max_mA
    df.loc[spike_cur, "current_mA"] = np.nan
    log["current_spikes_nulled"] = int(spike_cur.sum())

    # 3. Voltage outliers
    volt = pd.to_numeric(df["voltage_mV"], errors="coerce")
    bad_volt = (volt < voltage_min_mV) | (volt > voltage_max_mV)
    df.loc[bad_volt, "voltage_mV"] = np.nan
    log["voltage_outliers_nulled"] = int(bad_volt.sum())

    # 4. IMU spike rejection (z-score per axis)
    imu_axes = [c for c in ("ax_x100", "ay_x100", "az_x100") if c in df.columns]
    imu_total = 0
    for col in imu_axes:
        s = pd.to_numeric(df[col], errors="coerce")
        std = s.std(skipna=True)
        if std and std > 0:
            z = (s - s.mean(skipna=True)) / std
            spikes = z.abs() > imu_spike_zscore
            df.loc[spikes, col] = np.nan
            imu_total += int(spikes.sum())
    log["imu_spikes_nulled"] = imu_total

    # 5. Fill all NaN sensor values (never touch timestamp_ms)
    for col in sensor_cols:
        if df[col].isna().any():
            if gap_fill_method == "ffill":
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
    log["total_rows_after_clean"] = len(df)

    n_removed = n_before - len(df)
    parts = []
    if log["duplicate_rows_removed"]:
        parts.append(f"{log['duplicate_rows_removed']} duplicate rows")
    if log["current_spikes_nulled"]:
        parts.append(f"{log['current_spikes_nulled']} current spikes")
    if log["voltage_outliers_nulled"]:
        parts.append(f"{log['voltage_outliers_nulled']} voltage outliers")
    if log["imu_spikes_nulled"]:
        parts.append(f"{log['imu_spikes_nulled']} IMU spikes")
    if parts:
        print(f"  [clean_telemetry] Fixed: {', '.join(parts)}.")
    else:
        print("  [clean_telemetry] No issues found.")

    df.attrs["clean_log"] = log
    return df


def reindex_to_distance(
    df: pd.DataFrame,
    *,
    dist_col: str = "cumdist_m",
    step_m: float = 1.0,
    interp_cols: "list[str] | None" = None,
) -> pd.DataFrame:
    """Re-index a lap DataFrame from the time domain to a uniform distance grid.

    Method
    ------
    * Build a uniform grid ``[0, step_m, 2*step_m, …, lap_total_dist_m]``.
    * Linearly interpolate every numeric sensor/derived column onto that grid.
    * Non-numeric columns (e.g. string labels) are forward-filled.

    Parameters
    ----------
    df:
        Merged, derived lap DataFrame (output of ``derive_motion_energy``).
        Must contain ``dist_col``.
    dist_col:
        The cumulative-distance column to use as the new x-axis.  Defaults
        to ``"cumdist_m"`` (Haversine GPS distance).
    step_m:
        Uniform grid spacing in metres.  1 m gives ~lap_length rows per lap
        (a few thousand rows) which is fine for plotting and comparison.
    interp_cols:
        Explicit list of columns to interpolate.  ``None`` (default) means
        all numeric columns except ``timestamp_ms`` and index-like columns.

    Returns
    -------
    pd.DataFrame on a uniform distance grid with ``dist_m_grid`` as the
    index column.
    """
    if dist_col not in df.columns:
        dist_col = "cumdist_m" if "cumdist_m" in df.columns else None
    if dist_col is None:
        raise ValueError("No cumulative distance column found. Run derive_motion_energy first.")

    df = df.copy().sort_values(dist_col).reset_index(drop=True)

    x_raw = pd.to_numeric(df[dist_col], errors="coerce").ffill().fillna(0.0).to_numpy()
    # Ensure monotone (small GPS jitter can produce tiny back-steps)
    for i in range(1, len(x_raw)):
        if x_raw[i] < x_raw[i - 1]:
            x_raw[i] = x_raw[i - 1]

    total_dist = float(x_raw[-1])
    if total_dist <= 0:
        raise ValueError("Cumulative distance is zero — cannot reindex.")

    x_grid = np.arange(0.0, total_dist + step_m * 0.5, step_m)

    # Decide which columns to interpolate
    skip = {"timestamp_ms", dist_col, "lap", "sector"}
    if interp_cols is None:
        interp_cols = [
            c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]

    out = pd.DataFrame({"dist_m_grid": x_grid})
    for col in interp_cols:
        y_raw = pd.to_numeric(df[col], errors="coerce").to_numpy()
        # Fill internal NaN before interpolating
        valid = ~np.isnan(y_raw)
        if valid.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(x_grid, x_raw[valid], y_raw[valid])

    # Forward-fill string/categorical columns
    for col in df.columns:
        if col not in out.columns and col != dist_col:
            try:
                out[col] = np.interp(x_grid, x_raw, df[col].ffill().bfill().to_numpy(dtype=float))
            except (ValueError, TypeError):
                pass  # skip non-numeric non-interp-able columns

    return out


def compute_energy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute instantaneous and cumulative energy using Haversine GPS distance.

    Adds or updates
    ---------------
    ``power_w``             — V * I  (unchanged from derive_motion_energy)
    ``energy_j``            — power_w * dt_s  (per-row Joules)
    ``energy_wh``           — energy_j / 3600  (per-row Wh)
    ``cum_energy_j``        — cumulative Joules from lap start
    ``cum_energy_wh``       — cumulative Wh from lap start
    ``efficiency_wh_per_km``— instantaneous efficiency in Wh/km, computed
                              from ``dist_m`` (Haversine GPS distance).
                              A rolling 5-sample median is applied to reduce
                              per-row noise.
    """
    df = df.copy()

    current = pd.to_numeric(df["current_mA"], errors="coerce").abs()
    voltage = pd.to_numeric(df["voltage_mV"], errors="coerce")
    dt      = pd.to_numeric(df["dt_s"],       errors="coerce").fillna(0.0).clip(lower=0)

    df["power_w"]    = (current / 1000.0) * (voltage / 1000.0)
    df["energy_j"]   = df["power_w"] * dt
    df["energy_wh"]  = df["energy_j"] / 3600.0
    df["cum_energy_j"]  = df["energy_j"].cumsum()
    df["cum_energy_wh"] = df["energy_wh"].cumsum()

    dist_km = pd.to_numeric(df["dist_m"], errors="coerce").fillna(0.0) / 1000.0
    raw_eff = df["energy_wh"] / dist_km.replace(0, np.nan)
    window = max(3, _window_samples(df, 5.0))
    df["efficiency_wh_per_km"] = (
        raw_eff.rolling(window=window, min_periods=1, center=True)
        .median()
        .replace([np.inf, -np.inf], np.nan)
    )
    return df



# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0  # Mean Earth radius in metres


def haversine_distance(
    lat1: "np.ndarray | pd.Series | float",
    lon1: "np.ndarray | pd.Series | float",
    lat2: "np.ndarray | pd.Series | float",
    lon2: "np.ndarray | pd.Series | float",
) -> "np.ndarray | float":
    """Return the great-circle distance(s) in metres between two GPS points.

    Uses the Haversine formula, which is accurate to within ~0.5% for the
    short inter-sample distances typical of a race track.

    Parameters
    ----------
    lat1, lon1 : latitude / longitude of point 1 in decimal degrees.
    lat2, lon2 : latitude / longitude of point 2 in decimal degrees.

    Returns
    -------
    Distance in metres (same shape as the inputs).
    """
    lat1_r = np.asarray(lat1, dtype=float) / 180.0 * np.pi
    lat2_r = np.asarray(lat2, dtype=float) / 180.0 * np.pi
    lon1_r = np.asarray(lon1, dtype=float) / 180.0 * np.pi
    lon2_r = np.asarray(lon2, dtype=float) / 180.0 * np.pi

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


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
    """Total track distance in metres using the Haversine formula."""
    if len(df) < 2:
        return 0.0
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    seg_m = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(seg_m.sum())


def add_gps_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add GPS-native distance and speed before telemetry rows are merged in.

    Speed must be derived on the GPX sampling clock. If it is computed after
    telemetry rows are nearest-joined to GPS points, a one-second GPS movement
    can be divided by a sub-second telemetry interval and the speed scale is
    overstated.
    """
    df = add_xy(df.copy()).sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    df["gps_time"] = times
    df["gps_dt_s"] = times.diff().dt.total_seconds().fillna(0.0).clip(lower=0)

    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    seg = np.zeros(len(df))
    if len(df) > 1:
        seg[1:] = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    df["gps_dist_m"] = seg
    df["gps_cumdist_m"] = df["gps_dist_m"].cumsum()
    df["gps_speed_m_s_raw"] = np.where(
        df["gps_dt_s"] > 0,
        df["gps_dist_m"] / df["gps_dt_s"],
        0.0,
    )
    df["gps_speed_m_s"] = (
        pd.Series(df["gps_speed_m_s_raw"])
        .rolling(window=3, min_periods=1, center=True)
        .median()
    )
    df["gps_speed_kph"] = df["gps_speed_m_s"] * 3.6
    return df


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


def find_lap_boundaries_by_start_gate(
    gps: pd.DataFrame,
    start_index: int,
    laps: int,
    y_band_width: float = 5.0,
    x_window_width: float = 60.0,
    min_gap_points: int = 50,
    min_lap_distance_m: float = 2500.0,
    pre_race_max_distance_m: float = 1000.0,
) -> list[int]:
    """Detect lap boundaries at a localized start/finish gate.

    The current spike can occur while the car is still moving around before the
    real line.  First find that short pre-race return to the start Y band, then
    count only future re-entries near the same X/Y anchor.
    """
    gps_xy = add_xy(gps).reset_index(drop=True)
    if gps_xy.empty:
        return []
    if start_index < 0 or start_index >= len(gps_xy):
        raise IndexError("start_index is outside the GPS data range.")

    xy = gps_xy[["x", "y"]].to_numpy()
    seg_dists = np.zeros(len(xy))
    if len(xy) > 1:
        seg_dists[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    cum_dist = np.cumsum(seg_dists)

    y_start = float(gps_xy.loc[start_index, "y"])
    y = gps_xy["y"].to_numpy()

    anchor_idx = start_index
    escaped = False
    in_band = False
    crossing_count = 0
    for idx in range(start_index + 1, len(gps_xy)):
        near_y = abs(y[idx] - y_start) <= y_band_width

        if not escaped:
            if not near_y:
                escaped = True
                in_band = False
            continue

        if near_y and not in_band:
            in_band = True
            if idx - start_index < min_gap_points:
                continue
            crossing_count += 1
            if crossing_count % 2 == 0:
                pre_race_dist = cum_dist[idx] - cum_dist[start_index]
                if pre_race_dist <= pre_race_max_distance_m:
                    anchor_idx = idx
                break
        elif not near_y:
            in_band = False

    anchor_x = float(gps_xy.loc[anchor_idx, "x"])
    anchor_y = float(gps_xy.loc[anchor_idx, "y"])

    def inside_gate(idx: int) -> bool:
        return (
            abs(float(gps_xy.loc[idx, "y"]) - anchor_y) <= y_band_width
            and abs(float(gps_xy.loc[idx, "x"]) - anchor_x) <= x_window_width
        )

    boundaries = [anchor_idx]
    current_lap_start_idx = anchor_idx
    escaped_gate = False
    in_gate = True

    for idx in range(anchor_idx + 1, len(gps_xy)):
        inside = inside_gate(idx)

        if not escaped_gate:
            if not inside:
                escaped_gate = True
                in_gate = False
            continue

        if inside and not in_gate:
            in_gate = True
            if idx - current_lap_start_idx < min_gap_points:
                continue

            lap_dist = cum_dist[idx] - cum_dist[current_lap_start_idx]
            if lap_dist < min_lap_distance_m:
                print(
                    f"  Skipping short start-gate re-entry ({lap_dist:.0f}m) "
                    f"at GPS index {idx}."
                )
                continue

            boundaries.append(idx)
            current_lap_start_idx = idx
            if len(boundaries) >= laps + 1:
                break
        elif not inside:
            in_gate = False

    print(
        "Start-gate detection: "
        f"anchor=({anchor_x:.1f}m, {anchor_y:.1f}m), "
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

    if method in ("start", "gate"):
        raise ValueError(f"The '{method}' split method must be handled after GPS/telemetry alignment.")

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
    gps = add_gps_motion_features(
        gps.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    )
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

def _axis_series(df: pd.DataFrame, forward_axis: str) -> pd.Series:
    if forward_axis not in FORWARD_AXIS_CHOICES:
        raise ValueError(
            f"forward_axis must be one of {', '.join(FORWARD_AXIS_CHOICES)}. "
            f"Got: {forward_axis}"
        )

    sign = -1.0 if forward_axis.startswith("neg_") else 1.0
    axis = forward_axis.replace("neg_", "")
    return sign * pd.to_numeric(df[f"{axis}_x100"], errors="coerce")


def _sample_dt_seconds(df: pd.DataFrame) -> pd.Series:
    if "dt_s" in df.columns:
        return pd.to_numeric(df["dt_s"], errors="coerce").fillna(0.0).clip(lower=0)
    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
        return times.diff().dt.total_seconds().fillna(0.0).clip(lower=0)
    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        return (ts.diff() / 1000.0).fillna(0.0).clip(lower=0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def _window_samples(df: pd.DataFrame, window_s: float) -> int:
    if window_s <= 0:
        return 1
    dt_s = _sample_dt_seconds(df)
    positive = dt_s[dt_s > 0]
    if positive.empty:
        return max(1, int(round(window_s)))
    median_dt = float(positive.median())
    if median_dt <= 0:
        return max(1, int(round(window_s)))
    return max(1, int(round(window_s / median_dt)))


def _correlation(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat(
        [pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 5:
        return float("nan")
    left = pair.iloc[:, 0]
    right = pair.iloc[:, 1]
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(left.corr(right))


def derive_acceleration_features(
    df: pd.DataFrame,
    forward_axis: str = "ax",
    smooth_window: int | None = None,
    gravity_g: float = 1.0,
    accel_scale: float = 1000.0,
    imu_axis: str = "ax",
    imu_axis_sign: int = 1,
    bias_window_s: float = 30.0,
    smooth_window_s: float = 3.0,
) -> pd.DataFrame:
    """Add IMU-derived acceleration channels.

    The MPU-6050 fields are labelled *_x100 in old dumps, but observed
    magnitudes are milli-g: amag_x100 ~= 1000 means roughly 1 g.

    NOTE: The ``smooth_window`` parameter (sample count) is deprecated.
    Pass ``smooth_window_s`` (seconds) instead.  If ``smooth_window`` is
    provided it is converted to seconds for backward compatibility, but
    this will be removed in a future version.
    """
    df = df.copy()
    if accel_scale <= 0:
        raise ValueError("accel_scale must be positive.")
    if forward_axis not in FORWARD_AXIS_CHOICES:
        raise ValueError(
            f"forward_axis must be one of {', '.join(FORWARD_AXIS_CHOICES)}. "
            f"Got: {forward_axis}"
        )
    if imu_axis not in ("ax", "ay", "az"):
        raise ValueError("imu_axis must be one of ax, ay, az.")
    if imu_axis_sign not in (-1, 1):
        raise ValueError("imu_axis_sign must be -1 or 1.")
    if smooth_window is not None:
        import warnings
        warnings.warn(
            "smooth_window (sample count) is deprecated; pass smooth_window_s (seconds) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        smooth_window_s = float(smooth_window)

    ax_g = pd.to_numeric(df["ax_x100"], errors="coerce") / accel_scale
    ay_g = pd.to_numeric(df["ay_x100"], errors="coerce") / accel_scale
    az_g = pd.to_numeric(df["az_x100"], errors="coerce") / accel_scale
    df["imu_ax_g"] = ax_g
    df["imu_ay_g"] = ay_g
    df["imu_az_g"] = az_g
    if "amag_x100" in df.columns:
        df["imu_total_g_reported"] = (
            pd.to_numeric(df["amag_x100"], errors="coerce") / accel_scale
        )
    df["accel_total_g"] = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
    df["imu_total_g"] = df["accel_total_g"]
    df["accel_total_m_s2"] = df["accel_total_g"] * GRAVITY_M_S2
    df["accel_dynamic_g"] = df["accel_total_g"] - gravity_g
    df["accel_dynamic_m_s2"] = df["accel_dynamic_g"] * GRAVITY_M_S2
    for axis, values_g in (("ax", ax_g), ("ay", ay_g), ("az", az_g)):
        raw_m_s2 = values_g * GRAVITY_M_S2
        df[f"imu_{axis}_m_s2"] = raw_m_s2
        bias_window = _window_samples(df, bias_window_s)
        bias = raw_m_s2.rolling(
            window=bias_window,
            min_periods=1,
            center=True,
        ).median()
        dynamic = raw_m_s2 - bias
        smooth_window_samples = _window_samples(df, smooth_window_s)
        df[f"imu_{axis}_bias_m_s2"] = bias
        df[f"imu_{axis}_dynamic_m_s2"] = dynamic
        df[f"imu_{axis}_dynamic_smooth_m_s2"] = dynamic.rolling(
            window=smooth_window_samples,
            min_periods=1,
            center=True,
        ).median()

    legacy_axis = forward_axis.replace("neg_", "")
    legacy_sign = -1 if forward_axis.startswith("neg_") else 1
    df["accel_longitudinal_raw_g"] = (
        legacy_sign * pd.to_numeric(df[f"{legacy_axis}_x100"], errors="coerce")
        / accel_scale
    )
    df["accel_longitudinal_m_s2"] = (
        df["accel_longitudinal_raw_g"] * GRAVITY_M_S2
    )
    df["imu_forward_dynamic_m_s2"] = (
        imu_axis_sign * df[f"imu_{imu_axis}_dynamic_smooth_m_s2"]
    )
    # Backward-compatible alias: this is now bias-corrected MPU dynamic accel,
    # not raw longitudinal acceleration.
    df["accel_longitudinal_smooth_m_s2"] = df["imu_forward_dynamic_m_s2"]

    dt_s = _sample_dt_seconds(df)
    delta_accel = df["accel_longitudinal_smooth_m_s2"].diff().fillna(0.0)
    df["jerk_m_s3"] = np.where(dt_s > 0, delta_accel / dt_s, 0.0)

    # Backward-compatible names used by the heatmap script.
    df["accel_m_s2"] = df["accel_total_m_s2"]
    df["accel_mag"] = df["accel_total_g"]
    return df


def add_gps_acceleration_features(
    df: pd.DataFrame,
    smooth_window_s: float = 3.0,
) -> pd.DataFrame:
    df = df.copy()
    if {"gps_time", "gps_speed_m_s"}.issubset(df.columns):
        gps_motion = (
            df[["gps_time", "gps_speed_m_s"]]
            .dropna()
            .drop_duplicates(subset=["gps_time"])
            .sort_values("gps_time")
            .reset_index(drop=True)
        )
        gps_times = pd.to_datetime(gps_motion["gps_time"])
        gps_dt_s = gps_times.diff().dt.total_seconds().replace(0, np.nan)
        gps_speed = pd.to_numeric(gps_motion["gps_speed_m_s"], errors="coerce")
        raw = (gps_speed.diff() / gps_dt_s).replace([np.inf, -np.inf], np.nan)
        positive_dt = gps_dt_s[gps_dt_s > 0]
        if positive_dt.empty or smooth_window_s <= 0:
            smooth_window = 1
        else:
            smooth_window = max(1, int(round(smooth_window_s / float(positive_dt.median()))))
        smooth = (
            raw.rolling(window=smooth_window, min_periods=1, center=True)
            .median()
            .fillna(0.0)
        )
        gps_motion["gps_longitudinal_accel_raw_m_s2"] = raw.fillna(0.0)
        gps_motion["gps_longitudinal_accel_m_s2"] = smooth
        accel_lookup = gps_motion.set_index("gps_time")
        df["gps_longitudinal_accel_raw_m_s2"] = (
            df["gps_time"].map(accel_lookup["gps_longitudinal_accel_raw_m_s2"]).fillna(0.0)
        )
        df["gps_longitudinal_accel_m_s2"] = (
            df["gps_time"].map(accel_lookup["gps_longitudinal_accel_m_s2"]).fillna(0.0)
        )
    else:
        speed = pd.to_numeric(df["speed_m_s"], errors="coerce")
        dt_s = pd.to_numeric(df["dt_s"], errors="coerce").replace(0, np.nan)
        raw = speed.diff() / dt_s
        df["gps_longitudinal_accel_raw_m_s2"] = raw.replace([np.inf, -np.inf], np.nan)
        smooth_window = _window_samples(df, smooth_window_s)
        df["gps_longitudinal_accel_m_s2"] = (
            df["gps_longitudinal_accel_raw_m_s2"]
            .rolling(window=smooth_window, min_periods=1, center=True)
            .median()
            .fillna(0.0)
        )
    df["gps_longitudinal_accel_abs_m_s2"] = (
        pd.to_numeric(df["gps_longitudinal_accel_m_s2"], errors="coerce")
        .abs()
        .fillna(0.0)
    )
    return df


def compute_accel_candidate_scores(df: pd.DataFrame) -> list[dict[str, float | str]]:
    scores = []
    gps_accel = df.get("gps_longitudinal_accel_m_s2", pd.Series(dtype=float))
    current = pd.to_numeric(
        df.get("current_mA", pd.Series(dtype=float)), errors="coerce"
    ).abs()
    power = pd.to_numeric(df.get("power_w", pd.Series(dtype=float)), errors="coerce")

    for axis in ("ax", "ay", "az"):
        column = f"imu_{axis}_dynamic_smooth_m_s2"
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        for sign, name in ((1, axis), (-1, f"-{axis}")):
            candidate = sign * values
            scores.append({
                "candidate": name,
                "gps_corr": _correlation(candidate, gps_accel),
                "current_corr": _correlation(candidate, current),
                "power_corr": _correlation(candidate, power),
            })
    return scores


def derive_motion_energy(
    df: pd.DataFrame,
    forward_axis: str = "ax",
    accel_window: int = 5,
    accel_scale: float = 1000.0,
    imu_axis: str = "ax",
    imu_axis_sign: int = 1,
    accel_bias_window_sec: float = 30.0,
    accel_smooth_window_sec: float = 3.0,
) -> pd.DataFrame:
    """Add dt_s, dist_m, elev_diff_m, speed_m_s, speed_kph, grade_pct,
    power_w, energy_wh, energy_j, cum_energy_j, and cumdist_m columns to a
    merged lap DataFrame.

    Expects columns: time, lat, lon, elev, current_mA, voltage_mV.
    """
    df = add_xy(df.copy())
    df = df.sort_values("time").reset_index(drop=True)

    # Time delta
    times = pd.to_datetime(df["time"])
    df["dt_s"] = times.diff().dt.total_seconds().fillna(0.0).clip(lower=0)
    df = derive_acceleration_features(
        df,
        forward_axis=forward_axis,
        smooth_window=None,
        accel_scale=accel_scale,
        imu_axis=imu_axis,
        imu_axis_sign=imu_axis_sign,
        bias_window_s=accel_bias_window_sec,
        smooth_window_s=accel_smooth_window_sec,
    )

    # Point-to-point distance via Haversine (great-circle, returns metres)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    seg = np.zeros(len(df))
    if len(df) > 1:
        seg[1:] = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    df["dist_m"] = seg

    # Elevation change
    elev = pd.to_numeric(df["elev"], errors="coerce").fillna(0.0).to_numpy()
    elev_diff = np.zeros(len(elev))
    elev_diff[1:] = np.diff(elev)
    df["elev_diff_m"] = elev_diff

    # Speed comes from GPX-native timing when available. Computing speed after
    # the telemetry merge can inflate values because telemetry samples are more
    # frequent than GPS points.
    fallback_speed_m_s = np.where(df["dt_s"] > 0, df["dist_m"] / df["dt_s"], 0.0)
    if "gps_speed_m_s" in df.columns:
        df["speed_m_s"] = (
            pd.to_numeric(df["gps_speed_m_s"], errors="coerce")
            .fillna(pd.Series(fallback_speed_m_s, index=df.index))
        )
    else:
        df["speed_m_s"] = fallback_speed_m_s
    df["speed_kph"] = df["speed_m_s"] * 3.6
    df = add_gps_acceleration_features(df, smooth_window_s=accel_smooth_window_sec)

    # Grade (%). Raw point-to-point GPX elevation is noisy, so keep the raw
    # grade but also expose a centered smoothed grade for strategy modeling.
    df["grade_raw_pct"] = np.where(
        df["dist_m"] > 0.01,
        (df["elev_diff_m"] / df["dist_m"]) * 100.0,
        0.0,
    )
    grade_window = max(3, _window_samples(df, 8.0))
    df["grade_pct"] = (
        pd.Series(df["grade_raw_pct"], index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .rolling(window=grade_window, min_periods=1, center=True)
        .median()
        .clip(lower=-12.0, upper=12.0)
    )
    df["uphill_grade_pct"] = df["grade_pct"].clip(lower=0.0)
    df["downhill_grade_pct"] = (-df["grade_pct"]).clip(lower=0.0)

    # Power and energy
    df["power_w"] = (df["current_mA"].abs() / 1000.0) * (df["voltage_mV"] / 1000.0)
    df["energy_wh"] = df["power_w"] * df["dt_s"] / 3600.0
    df["energy_j"] = df["power_w"] * df["dt_s"]
    df["cum_energy_j"] = df["energy_j"].cumsum()

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


# ---------------------------------------------------------------------------
# Lap-building orchestration
# ---------------------------------------------------------------------------

def build_laps(
    gps_df: pd.DataFrame,
    telem_df: pd.DataFrame,
    *,
    laps: int,
    split_method: str = "start",
    start_time: "str | pd.Timestamp | None" = None,
    time_offset_ms: float = 0.0,
    tolerance_sec: float = 1.5,
    lap_times: "list[str] | None" = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    """Return (gps_laps, telem_laps, aligned_telem_df).

    Centralised lap-building so that gps_current_heatmap.py,
    analyze_strategy.py, build_interactive_dashboard.py, and
    simulate_speed_strategy.py all use identical alignment and
    lap-splitting logic.

    Parameters
    ----------
    gps_df:
        Raw DataFrame from read_gpx().
    telem_df:
        Raw DataFrame from read_telemetry().
    laps:
        Expected number of racing laps.
    split_method:
        One of ``"start"`` (current-spike plus large-track gate detector),
        ``"gate"`` (GPS begins at the gate, with course-scaled thresholds),
        ``"points"``, ``"time"``, or ``"line"``.
    start_time:
        ISO 8601 string or Timestamp to force-align telemetry start.
        ``None`` aligns to the first GPS point.
    time_offset_ms:
        Additional millisecond nudge applied after start alignment.
    tolerance_sec:
        Merge tolerance for ``merge_by_time``.
    lap_times:
        Optional list of elapsed MM:SS / H:MM:SS strings that override
        automatic lap detection.
    """
    if lap_times:
        track_start = gps_df["time"].iloc[0]
        lap_timestamps = [parse_lap_time(t, track_start) for t in lap_times]
        if len(lap_timestamps) < 2:
            raise ValueError("lap_times requires at least 2 timestamps.")
        spike_idx = find_start_spike(telem_df)
        spike_ms = float(telem_df.loc[spike_idx, "timestamp_ms"])
        telemetry_start = lap_timestamps[0] - pd.Timedelta(milliseconds=spike_ms)
        telem_df = align_telemetry(telem_df, gps_df, telemetry_start, time_offset_ms)

        gps_laps, telem_laps = [], []
        for i in range(len(lap_timestamps) - 1):
            ls, le = lap_timestamps[i], lap_timestamps[i + 1]
            gps_laps.append(
                gps_df[(gps_df["time"] >= ls) & (gps_df["time"] < le)]
                .copy().reset_index(drop=True)
            )
            telem_laps.append(
                telem_df[(telem_df["time"] >= ls) & (telem_df["time"] < le)]
                .copy().reset_index(drop=True)
            )
        return gps_laps, telem_laps, telem_df

    telem_df = align_telemetry(telem_df, gps_df, start_time, time_offset_ms)

    if split_method in ("start", "gate"):
        if split_method == "start":
            spike_idx = find_start_spike(telem_df)
            spike_time = telem_df.loc[spike_idx, "time"]
            gps_start_idx = find_nearest_gps_index(gps_df, spike_time)
            print(
                f"Start spike at telemetry index {spike_idx}, time {spike_time}, "
                f"matching GPS index {gps_start_idx}."
            )
            gps_df = gps_df.loc[gps_start_idx:].reset_index(drop=True)
            boundaries = find_lap_boundaries_by_start_gate(gps_df, 0, laps)
        else:
            gps_df = gps_df.reset_index(drop=True)
            gps_xy = add_xy(gps_df)
            x_span = float(gps_xy["x"].max() - gps_xy["x"].min())
            y_span = float(gps_xy["y"].max() - gps_xy["y"].min())
            estimated_lap_m = max(200.0, x_span + y_span)
            boundaries = find_lap_boundaries_by_start_gate(
                gps_df,
                0,
                laps,
                y_band_width=max(10.0, min(20.0, y_span * 0.10)),
                x_window_width=max(40.0, min(80.0, x_span * 0.40)),
                min_gap_points=5,
                min_lap_distance_m=estimated_lap_m,
                pre_race_max_distance_m=estimated_lap_m * 0.6,
            )
        if len(boundaries) < laps + 1:
            print(
                f"Warning: only found {len(boundaries) - 1} complete laps "
                f"(wanted {laps}). Last segment will be appended."
            )
        gps_laps = []
        for i in range(min(len(boundaries) - 1, laps)):
            gps_laps.append(
                gps_df.iloc[boundaries[i]: boundaries[i + 1]].reset_index(drop=True)
            )
        if len(gps_laps) < laps and boundaries:
            gps_laps.append(gps_df.iloc[boundaries[-1]:].reset_index(drop=True))
    else:
        gps_laps = split_gps_into_laps(gps_df, laps, split_method)

    telem_laps = []
    for lap_gps in gps_laps:
        if lap_gps.empty:
            telem_laps.append(pd.DataFrame())
            continue
        ls = lap_gps["time"].iloc[0]
        le = lap_gps["time"].iloc[-1]
        telem_laps.append(
            telem_df[
                (telem_df["time"] >= ls - pd.Timedelta(seconds=tolerance_sec))
                & (telem_df["time"] <= le + pd.Timedelta(seconds=tolerance_sec))
            ].copy().reset_index(drop=True)
        )

    return gps_laps, telem_laps, telem_df
