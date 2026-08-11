import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_dashboard.app import (
    DynoTestManager,
    TelemetryHub,
    TelemetryInput,
    TelemetryRecord,
)


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

    def test_optional_wheel_speed(self):
        record = TelemetryRecord.from_input(
            self.make_input(wheel_speed_valid=True, wheel_speed_kph=23.45)
        )
        self.assertTrue(record.wheel_speed_valid)
        self.assertEqual(record.wheel_speed_kph, 23.45)

    def test_optional_motor_temperature(self):
        record = TelemetryRecord.from_input(
            self.make_input(
                motor_temperature_valid=True,
                motor_temperature_C=64.25,
            )
        )
        self.assertTrue(record.motor_temperature_valid)
        self.assertEqual(record.motor_temperature_C, 64.25)

    def test_legacy_payload_has_unavailable_wheel_speed(self):
        record = TelemetryRecord.from_input(self.make_input())
        self.assertFalse(record.wheel_speed_valid)
        self.assertIsNone(record.wheel_speed_kph)
        self.assertFalse(record.motor_temperature_valid)
        self.assertIsNone(record.motor_temperature_C)

    def test_dyno_uses_reported_power(self):
        record = TelemetryRecord.from_input(
            self.make_input(source_type="dyno", reported_power_W=37.25)
        )
        self.assertEqual(record.source_type, "dyno")
        self.assertAlmostEqual(record.power_W, 37.25)

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

    def test_dyno_test_integrates_each_source_and_stops(self):
        async def exercise():
            manager = DynoTestManager()
            await manager.start()
            for timestamp_ms in (0, 3_600_000):
                await manager.record(
                    TelemetryRecord.from_input(
                        self.make_input(
                            source_type="car",
                            timestamp_ms=timestamp_ms,
                            current_mA=4_000,
                            voltage_mV=25_000,
                        )
                    )
                )
                await manager.record(
                    TelemetryRecord.from_input(
                        self.make_input(
                            device_id="test-dyno",
                            source_type="dyno",
                            timestamp_ms=timestamp_ms,
                            reported_power_W=50.0,
                        )
                    )
                )
            return await manager.stop()

        result = asyncio.run(exercise())
        self.assertFalse(result["active"])
        self.assertAlmostEqual(result["input_energy_Wh"], 100.0)
        self.assertAlmostEqual(result["output_energy_Wh"], 50.0)
        self.assertAlmostEqual(result["efficiency_percent"], 50.0)
        self.assertAlmostEqual(result["car_current_A"], 4.0)
        self.assertAlmostEqual(result["car_voltage_V"], 25.0)
        self.assertAlmostEqual(result["dyno_current_A"], 2.0)
        self.assertAlmostEqual(result["dyno_voltage_V"], 24.0)


if __name__ == "__main__":
    unittest.main()
