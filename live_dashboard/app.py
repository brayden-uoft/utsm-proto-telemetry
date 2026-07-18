from __future__ import annotations

import asyncio
import hmac
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field


STATIC_DIR = Path(__file__).resolve().parent / "static"
API_KEY = os.environ.get("UTSM_TELEMETRY_API_KEY", "change-me")
MAX_RECENT_RECORDS = int(os.environ.get("UTSM_LIVE_MAX_RECORDS", "500"))


class TelemetryInput(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    source_boot_id: int = Field(ge=0, le=0xFFFFFFFF)
    sequence: int = Field(ge=0, le=0xFFFFFFFF)
    timestamp_ms: int = Field(ge=0, le=0xFFFFFFFF)
    current_mA: int = Field(ge=-32768, le=32767)
    voltage_mV: int = Field(ge=0, le=100000)
    ax_x100: int = Field(ge=-32768, le=32767)
    ay_x100: int = Field(ge=-32768, le=32767)
    az_x100: int = Field(ge=-32768, le=32767)
    amag_x100: int = Field(ge=0, le=65535)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


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
            power_W=(value.current_mA * value.voltage_mV) / 1_000_000.0,
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


hub = TelemetryHub(MAX_RECENT_RECORDS)
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


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "stored_records": len(hub.records)}


@app.get("/api/live/recent")
async def recent() -> dict[str, object]:
    return {"records": await hub.recent()}


@app.post("/api/live/telemetry", status_code=202)
async def ingest_telemetry(
    telemetry: TelemetryInput,
    x_telemetry_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    verify_ingestion_key(x_telemetry_key)
    record = TelemetryRecord.from_input(telemetry)
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
