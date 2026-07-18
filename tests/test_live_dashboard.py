import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_dashboard.app import TelemetryHub, TelemetryInput, TelemetryRecord


class TestLiveTelemetry(unittest.TestCase):
    def make_input(self, **overrides):
        values = {
            "device_id": "test-car",
            "source_boot_id": 123,
            "sequence": 9,
            "timestamp_ms": 42000,
            "current_mA": 2000,
            "voltage_mV": 24000,
            "ax_x100": 10,
            "ay_x100": 20,
            "az_x100": 980,
            "amag_x100": 981,
        }
        values.update(overrides)
        return TelemetryInput(**values)

    def test_derived_values(self):
        record = TelemetryRecord.from_input(self.make_input())
        self.assertAlmostEqual(record.power_W, 48.0)
        self.assertAlmostEqual(record.acceleration_mps2, 9.81)

    def test_optional_gps(self):
        record = TelemetryRecord.from_input(
            self.make_input(latitude=39.799, longitude=-86.238)
        )
        self.assertEqual(record.latitude, 39.799)
        self.assertEqual(record.longitude, -86.238)

    def test_recent_ring_limit(self):
        async def exercise():
            hub = TelemetryHub(max_records=2)
            for sequence in range(3):
                await hub.publish(
                    TelemetryRecord.from_input(self.make_input(sequence=sequence))
                )
            return await hub.recent()

        recent = asyncio.run(exercise())
        self.assertEqual([row["sequence"] for row in recent], [1, 2])


if __name__ == "__main__":
    unittest.main()

