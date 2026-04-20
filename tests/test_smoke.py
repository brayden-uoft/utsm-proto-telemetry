"""Minimal smoke tests for the utsm_telemetry package.

Run with:  python -m pytest tests/ -v
Or simply: python tests/test_smoke.py
"""

import math
import os
import sys
import unittest
import io

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utsm_telemetry.core import (
    add_xy,
    compute_distance,
    parse_iso8601,
    parse_lap_time,
    align_telemetry,
    derive_motion_energy,
    merge_by_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gps(n: int = 10, lat0: float = 43.0, lon0: float = -79.0) -> pd.DataFrame:
    times = pd.date_range("2026-04-11T12:00:00Z", periods=n, freq="1s")
    return pd.DataFrame({
        "lat": lat0 + np.linspace(0, 0.001, n),
        "lon": lon0 + np.linspace(0, 0.001, n),
        "elev": np.linspace(100, 105, n),
        "time": times,
    })


def _make_telem(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_ms": np.arange(n) * 1000,
        "current_mA": np.abs(np.random.randn(n) * 500 + 2000),
        "voltage_mV": np.full(n, 24000.0),
        "ax_x100": np.zeros(n),
        "ay_x100": np.zeros(n),
        "az_x100": np.full(n, 100.0),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseISO8601(unittest.TestCase):
    def test_z_suffix(self):
        ts = parse_iso8601("2026-04-11T12:00:00Z")
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.month, 4)

    def test_offset(self):
        ts = parse_iso8601("2026-04-11T12:00:00+00:00")
        self.assertEqual(ts.hour, 12)


class TestParseLapTime(unittest.TestCase):
    def test_mmss(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        result = parse_lap_time("02:30", start)
        self.assertEqual(result, start + pd.Timedelta(minutes=2, seconds=30))

    def test_hmmss(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        result = parse_lap_time("1:02:30", start)
        self.assertEqual(result, start + pd.Timedelta(hours=1, minutes=2, seconds=30))

    def test_bad_format(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        with self.assertRaises(ValueError):
            parse_lap_time("bad", start)


class TestAddXY(unittest.TestCase):
    def test_origin_at_zero(self):
        gps = _make_gps(5)
        xy = add_xy(gps)
        self.assertAlmostEqual(xy["x"].iloc[0], 0.0, places=5)
        self.assertAlmostEqual(xy["y"].iloc[0], 0.0, places=5)

    def test_monotone_increasing(self):
        gps = _make_gps(5)
        xy = add_xy(gps)
        self.assertTrue((xy["x"].diff().dropna() >= 0).all())
        self.assertTrue((xy["y"].diff().dropna() >= 0).all())


class TestComputeDistance(unittest.TestCase):
    def test_nonzero(self):
        gps = _make_gps(10)
        dist = compute_distance(gps)
        self.assertGreater(dist, 0)

    def test_single_point(self):
        gps = _make_gps(1)
        dist = compute_distance(gps)
        self.assertEqual(dist, 0.0)


class TestAlignTelemetry(unittest.TestCase):
    def test_creates_time_column(self):
        gps = _make_gps(10)
        telem = _make_telem(10)
        # Read telemetry without coercion (raw)
        from utsm_telemetry.core import read_telemetry as rt
        aligned = align_telemetry(telem, gps, None, 0.0)
        self.assertIn("time", aligned.columns)
        self.assertEqual(len(aligned), len(telem))

    def test_offset_applied(self):
        gps = _make_gps(10)
        telem = _make_telem(10)
        aligned_base = align_telemetry(telem, gps, None, 0.0)
        aligned_off = align_telemetry(telem, gps, None, 1000.0)
        delta = (aligned_off["time"].iloc[0] - aligned_base["time"].iloc[0]).total_seconds()
        self.assertAlmostEqual(delta, 1.0, places=3)


class TestMergeByTime(unittest.TestCase):
    def test_merge_produces_lat_lon(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        self.assertIn("lat", merged.columns)
        self.assertIn("lon", merged.columns)
        self.assertGreater(len(merged), 0)


class TestDeriveMotionEnergy(unittest.TestCase):
    def test_columns_present(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        for col in ("dt_s", "dist_m", "speed_kph", "power_w", "energy_wh", "cumdist_m"):
            self.assertIn(col, derived.columns, f"Missing column: {col}")

    def test_energy_nonnegative(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        self.assertTrue((derived["energy_wh"] >= 0).all())

    def test_cumdist_monotone(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        diffs = derived["cumdist_m"].diff().dropna()
        self.assertTrue((diffs >= 0).all())


if __name__ == "__main__":
    unittest.main()
