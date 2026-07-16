from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ACTION_CODES = {
    "unknown": 0,
    "accelerate": 1,
    "hold": 2,
    "coast": 3,
}


@dataclass(frozen=True)
class StrategyMapMeta:
    name: str
    lat0: float
    lon0: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float
    spacing_m: float
    offtrack_radius_m: float


def strategy_map_meta(samples_df: pd.DataFrame, *, name: str, spacing_m: float, offtrack_radius_m: float) -> StrategyMapMeta:
    if samples_df.empty:
        raise ValueError("samples_df must not be empty.")
    lat = pd.to_numeric(samples_df["lat"], errors="coerce")
    lon = pd.to_numeric(samples_df["lon"], errors="coerce")
    lat0 = float(lat.iloc[0])
    lon0 = float(lon.iloc[0])
    avg_lat_rad = math.radians(float(lat.mean()))
    return StrategyMapMeta(
        name=name,
        lat0=lat0,
        lon0=lon0,
        meters_per_deg_lat=110540.0,
        meters_per_deg_lon=111320.0 * math.cos(avg_lat_rad),
        spacing_m=float(spacing_m),
        offtrack_radius_m=float(offtrack_radius_m),
    )


def _finite_numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        raise ValueError(f"{name} has no finite values.")
    return out.interpolate(limit_direction="both").ffill().bfill()


def _distance_grid(total_dist_m: float, spacing_m: float) -> np.ndarray:
    if total_dist_m <= 0:
        raise ValueError("total_dist_m must be positive.")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive.")
    grid = np.arange(0.0, total_dist_m, spacing_m, dtype=float)
    if len(grid) == 0 or grid[-1] < total_dist_m:
        grid = np.append(grid, total_dist_m)
    return grid


def build_firmware_strategy_table(
    samples_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    *,
    name: str = "strategy",
    spacing_m: float = 10.0,
    offtrack_radius_m: float = 40.0,
) -> tuple[pd.DataFrame, StrategyMapMeta]:
    """Resample an optimized strategy into a compact GPS lookup table.

    The dashboard maps target speed by cumulative run distance. Firmware needs a
    table it can search using live GPS. This export keeps the route geometry
    from the processed samples, then attaches the optimized segment action and
    target speed at a fixed distance spacing.
    """
    if samples_df.empty:
        raise ValueError("samples_df must not be empty.")
    if profile_df.empty:
        raise ValueError("profile_df must not be empty.")

    required = {"lat", "lon", "x", "y", "run_cumdist_m"}
    missing = sorted(required - set(samples_df.columns))
    if missing:
        raise ValueError(f"samples_df is missing required columns: {', '.join(missing)}")
    required_profile = {"segment", "dist_end_m", "target_speed_kph", "action"}
    missing_profile = sorted(required_profile - set(profile_df.columns))
    if missing_profile:
        raise ValueError(f"profile_df is missing required columns: {', '.join(missing_profile)}")

    samples = samples_df.copy().sort_values("run_cumdist_m").reset_index(drop=True)
    dist = _finite_numeric(samples["run_cumdist_m"], "run_cumdist_m").to_numpy(dtype=float)
    _, unique_idx = np.unique(dist, return_index=True)
    samples = samples.iloc[np.sort(unique_idx)].reset_index(drop=True)
    dist = _finite_numeric(samples["run_cumdist_m"], "run_cumdist_m").to_numpy(dtype=float)
    total_dist = float(dist[-1])
    grid = _distance_grid(total_dist, spacing_m)

    profile = profile_df.copy().sort_values("dist_end_m").reset_index(drop=True)
    segment_edges = pd.to_numeric(profile["dist_end_m"], errors="coerce").to_numpy(dtype=float)
    segment_idx = np.searchsorted(segment_edges, grid, side="left")
    segment_idx = np.clip(segment_idx, 0, len(profile) - 1)
    mapped = profile.iloc[segment_idx].reset_index(drop=True)

    table = pd.DataFrame({
        "idx": np.arange(len(grid), dtype=int),
        "dist_m": grid,
        "lat": np.interp(grid, dist, _finite_numeric(samples["lat"], "lat")),
        "lon": np.interp(grid, dist, _finite_numeric(samples["lon"], "lon")),
        "x_m": np.interp(grid, dist, _finite_numeric(samples["x"], "x")),
        "y_m": np.interp(grid, dist, _finite_numeric(samples["y"], "y")),
        "segment": pd.to_numeric(mapped["segment"], errors="coerce").fillna(0).astype(int),
        "target_speed_kph": pd.to_numeric(mapped["target_speed_kph"], errors="coerce").fillna(0.0),
        "action": mapped["action"].astype(str),
    })
    table["action_code"] = table["action"].map(ACTION_CODES).fillna(0).astype(int)
    table["offtrack_radius_m"] = float(offtrack_radius_m)
    meta = strategy_map_meta(samples, name=name, spacing_m=spacing_m, offtrack_radius_m=offtrack_radius_m)
    return table, meta


def gps_to_local_xy(lat: float, lon: float, meta: StrategyMapMeta) -> tuple[float, float]:
    return (
        (float(lon) - meta.lon0) * meta.meters_per_deg_lon,
        (float(lat) - meta.lat0) * meta.meters_per_deg_lat,
    )


def nearest_strategy_recommendation(
    table: pd.DataFrame,
    meta: StrategyMapMeta,
    *,
    lat: float,
    lon: float,
    last_index: int | None = None,
    search_window: int = 24,
) -> dict[str, float | int | str | bool]:
    """Map a live GPS point to the nearest exported strategy segment."""
    if table.empty:
        raise ValueError("table must not be empty.")
    x, y = gps_to_local_xy(lat, lon, meta)
    if last_index is not None and 0 <= last_index < len(table):
        lo = max(0, int(last_index) - int(search_window))
        hi = min(len(table), int(last_index) + int(search_window) + 1)
        candidates = table.iloc[lo:hi].copy()
        offset = lo
    else:
        candidates = table
        offset = 0

    dx = pd.to_numeric(candidates["x_m"], errors="coerce").to_numpy(dtype=float) - x
    dy = pd.to_numeric(candidates["y_m"], errors="coerce").to_numpy(dtype=float) - y
    d2 = dx * dx + dy * dy
    local_idx = int(np.nanargmin(d2))
    best_idx = offset + local_idx
    distance_m = float(math.sqrt(float(d2[local_idx])))
    if distance_m > meta.offtrack_radius_m and last_index is not None:
        return nearest_strategy_recommendation(
            table,
            meta,
            lat=lat,
            lon=lon,
            last_index=None,
            search_window=search_window,
        )

    row = table.iloc[best_idx]
    return {
        "valid": bool(distance_m <= meta.offtrack_radius_m),
        "idx": int(best_idx),
        "distance_from_track_m": distance_m,
        "target_speed_kph": float(row["target_speed_kph"]),
        "segment": int(row["segment"]),
        "action": str(row["action"]),
    }


def _c_identifier(value: str) -> str:
    chars = []
    for ch in value.upper():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars).strip("_")
    return out or "STRATEGY"


def render_strategy_header(
    table: pd.DataFrame,
    meta: StrategyMapMeta,
    *,
    guard_name: str | None = None,
) -> str:
    ident = _c_identifier(guard_name or meta.name)
    lines = [
        "// Generated by export_firmware_strategy.py. Do not hand-edit table values.",
        f"#ifndef {ident}_H",
        f"#define {ident}_H",
        "",
        "#include <Arduino.h>",
        "",
        "enum StrategyAction : uint8_t {",
        "  STRATEGY_UNKNOWN = 0,",
        "  STRATEGY_ACCELERATE = 1,",
        "  STRATEGY_HOLD = 2,",
        "  STRATEGY_COAST = 3,",
        "};",
        "",
        "struct StrategyPoint {",
        "  int32_t lat_e7;",
        "  int32_t lon_e7;",
        "  int32_t x_cm;",
        "  int32_t y_cm;",
        "  uint16_t dist_m;",
        "  uint16_t segment;",
        "  uint16_t target_speed_kph_x10;",
        "  uint8_t action;",
        "};",
        "",
        f"static const double STRATEGY_LAT0 = {meta.lat0:.9f};",
        f"static const double STRATEGY_LON0 = {meta.lon0:.9f};",
        f"static const float STRATEGY_METERS_PER_DEG_LAT = {meta.meters_per_deg_lat:.6f}f;",
        f"static const float STRATEGY_METERS_PER_DEG_LON = {meta.meters_per_deg_lon:.6f}f;",
        f"static const float STRATEGY_SPACING_M = {meta.spacing_m:.3f}f;",
        f"static const float STRATEGY_OFFTRACK_RADIUS_M = {meta.offtrack_radius_m:.3f}f;",
        f"static const uint16_t STRATEGY_POINT_COUNT = {len(table)};",
        "",
        "static const StrategyPoint STRATEGY_POINTS[] PROGMEM = {",
    ]
    for row in table.itertuples(index=False):
        lat_e7 = int(round(float(row.lat) * 10_000_000))
        lon_e7 = int(round(float(row.lon) * 10_000_000))
        x_cm = int(round(float(row.x_m) * 100.0))
        y_cm = int(round(float(row.y_m) * 100.0))
        dist_m = max(0, min(65535, int(round(float(row.dist_m)))))
        segment = max(0, min(65535, int(row.segment)))
        target = max(0, min(65535, int(round(float(row.target_speed_kph) * 10.0))))
        action = max(0, min(255, int(row.action_code)))
        lines.append(
            f"  {{{lat_e7}, {lon_e7}, {x_cm}, {y_cm}, {dist_m}, {segment}, {target}, {action}}},"
        )
    lines.extend([
        "};",
        "",
        f"#endif  // {ident}_H",
        "",
    ])
    return "\n".join(lines)


def write_strategy_csv(table: pd.DataFrame, path: str) -> None:
    table.to_csv(path, index=False)


def write_strategy_header(table: pd.DataFrame, meta: StrategyMapMeta, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_strategy_header(table, meta, guard_name=meta.name))
