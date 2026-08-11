from __future__ import annotations

import asyncio
import hmac
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field


STATIC_DIR = Path(__file__).resolve().parent / "static"
API_KEY = os.environ.get("UTSM_TELEMETRY_API_KEY", "change-me")
MAX_RECENT_RECORDS = int(os.environ.get("UTSM_LIVE_MAX_RECORDS", "500"))


class TelemetryInput(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    source_type: Literal["car", "dyno"] = "car"
    source_boot_id: int = Field(ge=0, le=0xFFFFFFFF)
    sequence: int = Field(ge=0, le=0xFFFFFFFF)
    timestamp_ms: int = Field(ge=0, le=0xFFFFFFFF)
    current_mA: int = Field(ge=-2_000_000, le=2_000_000)
    voltage_mV: int = Field(ge=0, le=100000)
    motor_temperature_valid: bool = False
    motor_temperature_C: float | None = Field(default=None, ge=-100, le=300)
    ax_x100: int = Field(ge=-32768, le=32767)
    ay_x100: int = Field(ge=-32768, le=32767)
    az_x100: int = Field(ge=-32768, le=32767)
    amag_x100: int = Field(ge=0, le=65535)
    wheel_speed_valid: bool = False
    wheel_speed_kph: float | None = Field(default=None, ge=0, le=200)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    reported_power_W: float | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    source_energy_Wh: float | None = Field(default=None, ge=0, le=1_000_000_000)
    dyno_state: int | None = Field(default=None, ge=0, le=2)


class TelemetryRecord(TelemetryInput):
    received_at: datetime
    power_W: float
    acceleration_mps2: float

    @classmethod
    def from_input(cls, value: TelemetryInput) -> "TelemetryRecord":
        values = value.model_dump()
        return cls(
            **values,
            received_at=datetime.now(timezone.utc),
            power_W=(
                value.reported_power_W
                if value.reported_power_W is not None
                else (value.current_mA * value.voltage_mV) / 1_000_000.0
            ),
            acceleration_mps2=value.amag_x100 / 100.0,
        )


class TelemetryHub:
    def __init__(self, max_records: int) -> None:
        self.records: deque[TelemetryRecord] = deque(maxlen=max_records)
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def publish(self, record: TelemetryRecord) -> None:
        async with self.lock:
            self.records.append(record)
            clients = tuple(self.clients)

        payload = {"type": "telemetry", "record": record.model_dump(mode="json")}
        failed: list[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(payload)
            except Exception:
                failed.append(client)

        if failed:
            async with self.lock:
                for client in failed:
                    self.clients.discard(client)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.clients.add(websocket)
            snapshot = [record.model_dump(mode="json") for record in self.records]
        await websocket.send_json({"type": "snapshot", "records": snapshot})

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(websocket)

    async def recent(self) -> list[dict[str, object]]:
        async with self.lock:
            return [record.model_dump(mode="json") for record in self.records]


class EnergyAccumulator:
    def __init__(self) -> None:
        self.energy_Wh = 0.0
        self.last_boot_id: int | None = None
        self.last_timestamp_ms: int | None = None
        self.last_power_W: float | None = None

    def add(self, record: TelemetryRecord) -> None:
        power_W = max(0.0, record.power_W)
        if (
            self.last_boot_id == record.source_boot_id
            and self.last_timestamp_ms is not None
            and self.last_power_W is not None
            and record.timestamp_ms > self.last_timestamp_ms
        ):
            elapsed_hours = (record.timestamp_ms - self.last_timestamp_ms) / 3_600_000.0
            self.energy_Wh += (self.last_power_W + power_W) * 0.5 * elapsed_hours

        self.last_boot_id = record.source_boot_id
        self.last_timestamp_ms = record.timestamp_ms
        self.last_power_W = power_W


class DynoTestSession:
    def __init__(self, test_id: int) -> None:
        self.test_id = test_id
        self.started_at = datetime.now(timezone.utc)
        self.stopped_at: datetime | None = None
        self.car = EnergyAccumulator()
        self.dyno = EnergyAccumulator()
        self.latest_car: TelemetryRecord | None = None
        self.latest_dyno: TelemetryRecord | None = None

    def add(self, record: TelemetryRecord) -> None:
        if record.source_type == "dyno":
            self.dyno.add(record)
            self.latest_dyno = record
        else:
            self.car.add(record)
            self.latest_car = record

    def snapshot(self) -> dict[str, object]:
        input_Wh = self.car.energy_Wh
        output_Wh = self.dyno.energy_Wh
        efficiency = output_Wh / input_Wh * 100.0 if input_Wh > 0 else None
        live_input = max(0.0, self.latest_car.power_W) if self.latest_car else None
        live_output = max(0.0, self.latest_dyno.power_W) if self.latest_dyno else None
        live_efficiency = (
            live_output / live_input * 100.0
            if live_input is not None and live_input > 0 and live_output is not None
            else None
        )
        return {
            "test_id": self.test_id,
            "active": self.stopped_at is None,
            "started_at": self.started_at.isoformat(),
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "input_power_W": live_input,
            "output_power_W": live_output,
            "live_efficiency_percent": live_efficiency,
            "input_energy_Wh": input_Wh,
            "output_energy_Wh": output_Wh,
            "efficiency_percent": efficiency,
            "car_current_A": (
                self.latest_car.current_mA / 1000.0 if self.latest_car else None
            ),
            "car_voltage_V": (
                self.latest_car.voltage_mV / 1000.0 if self.latest_car else None
            ),
            "dyno_current_A": (
                self.latest_dyno.current_mA / 1000.0 if self.latest_dyno else None
            ),
            "dyno_voltage_V": (
                self.latest_dyno.voltage_mV / 1000.0 if self.latest_dyno else None
            ),
            "car_received_at": (
                self.latest_car.received_at.isoformat() if self.latest_car else None
            ),
            "dyno_received_at": (
                self.latest_dyno.received_at.isoformat() if self.latest_dyno else None
            ),
        }


class DynoTestManager:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.session: DynoTestSession | None = None
        self.next_test_id = 1

    async def start(self) -> dict[str, object]:
        async with self.lock:
            if self.session is not None and self.session.stopped_at is None:
                raise ValueError("A dyno test is already running")
            self.session = DynoTestSession(self.next_test_id)
            self.next_test_id += 1
            return self.session.snapshot()

    async def stop(self) -> dict[str, object]:
        async with self.lock:
            if self.session is None or self.session.stopped_at is not None:
                raise ValueError("No dyno test is running")
            self.session.stopped_at = datetime.now(timezone.utc)
            return self.session.snapshot()

    async def record(self, record: TelemetryRecord) -> None:
        async with self.lock:
            if self.session is not None and self.session.stopped_at is None:
                self.session.add(record)

    async def current(self) -> dict[str, object] | None:
        async with self.lock:
            return self.session.snapshot() if self.session is not None else None


hub = TelemetryHub(MAX_RECENT_RECORDS)
dyno_tests = DynoTestManager()
app = FastAPI(title="UTSM Live Telemetry", version="1.0.0")


def verify_ingestion_key(
    x_telemetry_key: Annotated[str | None, Header()] = None,
) -> None:
    supplied = x_telemetry_key or ""
    if not hmac.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid telemetry API key")


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/live")


@app.get("/live")
async def live_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "live.html")


@app.get("/dyno")
async def dyno_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dyno.html")


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "stored_records": len(hub.records)}


@app.get("/api/live/recent")
async def recent() -> dict[str, object]:
    return {"records": await hub.recent()}


@app.get("/api/dyno-tests/current")
async def current_dyno_test() -> dict[str, object]:
    return {"test": await dyno_tests.current()}


@app.post("/api/dyno-tests/start", status_code=201)
async def start_dyno_test() -> dict[str, object]:
    try:
        return {"test": await dyno_tests.start()}
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/dyno-tests/stop")
async def stop_dyno_test() -> dict[str, object]:
    try:
        return {"test": await dyno_tests.stop()}
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/live/telemetry", status_code=202)
async def ingest_telemetry(
    telemetry: TelemetryInput,
    x_telemetry_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    verify_ingestion_key(x_telemetry_key)
    record = TelemetryRecord.from_input(telemetry)
    await dyno_tests.record(record)
    await hub.publish(record)
    return {"accepted": True, "sequence": record.sequence}


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        while True:
            # Client messages are ignored; receiving lets disconnects surface.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(websocket)
    except Exception:
        await hub.disconnect(websocket)
