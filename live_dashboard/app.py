from __future__ import annotations

import asyncio
import csv
import hmac
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from live_dashboard.run_library import ALLOWED_IMPORT_EXTENSIONS, MAX_CSV_VIEW_ROWS, ImportSource, MalformedRunError, RunLibrary
from utsm_telemetry.safe_archive import ArchiveValidationError


STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_RUNS_DIR = Path(os.environ.get("UTSM_PACKAGED_RUNS_DIR", REPO_ROOT / "data" / "runs"))
DASHBOARD_DATA_DIR = Path(os.environ.get("UTSM_DASHBOARD_DATA_DIR", REPO_ROOT / ".dashboard-data"))
USER_RUNS_DIR = DASHBOARD_DATA_DIR / "runs"
BLOB_DIR = DASHBOARD_DATA_DIR / "blobs"
STRATEGY_DIR = REPO_ROOT / "data" / "strategy"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"
API_KEY = os.environ.get("UTSM_TELEMETRY_API_KEY", "change-me")
OPERATOR_KEY = os.environ.get("UTSM_DASHBOARD_OPERATOR_KEY")
MAX_RECENT_RECORDS = int(os.environ.get("UTSM_LIVE_MAX_RECORDS", "500"))
MAX_UPLOAD_BYTES = int(os.environ.get("UTSM_RUN_UPLOAD_MAX_BYTES", str(500 * 1024 * 1024)))
MAX_UPLOAD_FILES = int(os.environ.get("UTSM_RUN_UPLOAD_MAX_FILES", "100"))
MAX_MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024
SESSION_COOKIE = "utsm_operator_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
MIN_OPERATOR_KEY_LENGTH = 16
STRATEGY_SPECS = {
    "indy": {
        "label": "Indianapolis run-derived strategy",
        "description": "Firmware-ready target speed and accelerate, hold, or coast guidance for the Indianapolis course.",
        "provenance": "Generated from the packaged Afternoon Run telemetry and GPX. This is run-derived strategy data, not a new measured run.",
        "files": {
            "map": STRATEGY_DIR / "indy_strategy_map.csv",
            "firmware": STRATEGY_DIR / "indy_strategy_map.h",
        },
    },
    "autodrome-chaudiere": {
        "label": "Autodrome Chaudière reference strategy",
        "description": "A model-derived starting profile on packaged reference-track geometry.",
        "provenance": "Transferred from the Afternoon Run vehicle model onto the Autodrome reference centerline. It is not measured Autodrome run data and must be refit after real laps.",
        "files": {
            "map": TRACKS_DIR / "autodrome-chaudiere" / "autodrome-chaudiere-efficiency-strategy.csv",
            "segments": TRACKS_DIR / "autodrome-chaudiere" / "autodrome-chaudiere-strategy-segments.csv",
            "report": TRACKS_DIR / "autodrome-chaudiere" / "autodrome-chaudiere-strategy-report.txt",
        },
    },
}


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

    async def latest_by_source(self) -> dict[str, dict[str, object]]:
        async with self.lock:
            latest: dict[str, dict[str, object]] = {}
            for record in reversed(self.records):
                if record.source_type not in latest:
                    latest[record.source_type] = record.model_dump(mode="json")
                if len(latest) == 2:
                    break
            return latest


class EnergyAccumulator:
    def __init__(self) -> None:
        self.energy_Wh = 0.0
        self.last_boot_id: int | None = None
        self.last_timestamp_ms: int | None = None
        self.last_power_W: float | None = None
        self.last_source_energy_Wh: float | None = None

    def add(self, record: TelemetryRecord) -> None:
        if record.source_energy_Wh is not None:
            source_energy_Wh = max(0.0, record.source_energy_Wh)
            if self.last_source_energy_Wh is not None:
                if self.last_boot_id == record.source_boot_id:
                    self.energy_Wh += max(
                        0.0, source_energy_Wh - self.last_source_energy_Wh
                    )
                else:
                    # A reboot resets the source counter. Count energy produced
                    # after the reboot before its first packet reached LTE.
                    self.energy_Wh += source_energy_Wh
            self.last_boot_id = record.source_boot_id
            self.last_source_energy_Wh = source_energy_Wh
            self.last_timestamp_ms = record.timestamp_ms
            self.last_power_W = max(0.0, record.power_W)
            return

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
run_library = RunLibrary(PACKAGED_RUNS_DIR, USER_RUNS_DIR, BLOB_DIR, STRATEGY_DIR)
import_lock = threading.Lock()
app = FastAPI(title="UTSM Telemetry Dashboard", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class OperatorUnlock(BaseModel):
    key: str = Field(min_length=MIN_OPERATOR_KEY_LENGTH, max_length=512)


class UploadTooLarge(ValueError):
    pass


class RequestUploadTooLarge(Exception):
    pass


class ImportBodyLimitMiddleware:
    """Authorize and bound imports before Starlette reads multipart bytes."""

    def __init__(self, wrapped_app) -> None:
        self.wrapped_app = wrapped_app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST" or scope.get("path") not in {"/api/runs/import", "/api/operator/unlock"}:
            await self.wrapped_app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        path = scope.get("path")
        marker = b"dashboard-import" if path == "/api/runs/import" else b"dashboard-unlock"
        if headers.get(b"x-utsm-request") != marker:
            await JSONResponse(status_code=403, content={"detail": "Missing dashboard request header"})(scope, receive, send)
            return
        origin = headers.get(b"origin")
        if origin:
            host = headers.get(b"host", b"").decode("latin-1")
            expected = f"{scope.get('scheme', 'http')}://{host}".rstrip("/")
            if origin.decode("latin-1").rstrip("/") != expected:
                await JSONResponse(status_code=403, content={"detail": "Cross-origin dashboard requests are not allowed"})(scope, receive, send)
                return
        if not operator_configured():
            await JSONResponse(status_code=503, content={"detail": "Operator access is not configured"})(scope, receive, send)
            return
        if path == "/api/runs/import":
            cookie_header = headers.get(b"cookie", b"").decode("latin-1")
            token = ""
            for part in cookie_header.split(";"):
                name, separator, value = part.strip().partition("=")
                if separator and name == SESSION_COOKIE:
                    token = value
                    break
            if not valid_session_token(token):
                await JSONResponse(status_code=401, content={"detail": "Unlock operator access to continue"})(scope, receive, send)
                return
        limit = (
            MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
            if path == "/api/runs/import"
            else 64 * 1024
        )
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    await JSONResponse(status_code=413, content={"detail": "Upload is too large."})(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})(scope, receive, send)
                return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestUploadTooLarge
            return message

        try:
            await self.wrapped_app(scope, limited_receive, send)
        except RequestUploadTooLarge:
            await JSONResponse(status_code=413, content={"detail": "Upload is too large."})(scope, receive, send)


app.add_middleware(ImportBodyLimitMiddleware)
unlock_attempts: dict[str, tuple[int, float]] = {}
unlock_attempts_lock = threading.Lock()


def verify_ingestion_key(
    x_telemetry_key: Annotated[str | None, Header()] = None,
) -> None:
    supplied = x_telemetry_key or ""
    if not hmac.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid telemetry API key")


def operator_configured() -> bool:
    placeholders = {"change-me", "change-me-operator", "replace-this-for-run-imports"}
    return bool(
        OPERATOR_KEY
        and len(OPERATOR_KEY.strip()) >= MIN_OPERATOR_KEY_LENGTH
        and OPERATOR_KEY.strip().lower() not in placeholders
    )


def public_import_error(error: ValueError) -> str:
    text = " ".join(str(error).split())[:400]
    if re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|tmp|users|var)/)", text, re.IGNORECASE):
        return "The uploaded files could not be imported. Check their format and required columns."
    return text or "The uploaded files could not be imported."


def _session_token() -> str:
    if not operator_configured():
        raise HTTPException(status_code=503, detail="Operator access is not configured")
    expires = int(time.time()) + SESSION_TTL_SECONDS
    nonce = secrets.token_urlsafe(18)
    payload = f"{expires}.{nonce}"
    signature = hmac.new(OPERATOR_KEY.encode(), payload.encode(), "sha256").hexdigest()
    return f"{payload}.{signature}"


def valid_session_token(token: str) -> bool:
    if not operator_configured():
        return False
    try:
        expires_text, nonce, signature = token.split(".", 2)
        payload = f"{expires_text}.{nonce}"
        expected = hmac.new(OPERATOR_KEY.encode(), payload.encode(), "sha256").hexdigest()
        return int(expires_text) >= int(time.time()) and hmac.compare_digest(signature, expected)
    except (ValueError, TypeError):
        return False


def has_operator_session(request: Request) -> bool:
    return valid_session_token(request.cookies.get(SESSION_COOKIE, ""))


def require_operator_session(request: Request) -> None:
    if not operator_configured():
        raise HTTPException(status_code=503, detail="Operator access is not configured")
    if not has_operator_session(request):
        raise HTTPException(status_code=401, detail="Unlock operator access to continue")


def _unlock_client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _unlock_retry_after(client: str) -> int:
    with unlock_attempts_lock:
        entry = unlock_attempts.get(client)
    if entry is None:
        return 0
    return max(0, math.ceil(entry[1] - time.monotonic()))


def _record_unlock_failure(client: str) -> int:
    now = time.monotonic()
    with unlock_attempts_lock:
        count, _ = unlock_attempts.get(client, (0, 0.0))
        count += 1
        delay = min(8.0, 0.5 * (2 ** min(count - 1, 4)))
        unlock_attempts[client] = (count, now + delay)
        if len(unlock_attempts) > 1000:
            for key in list(unlock_attempts)[:250]:
                if unlock_attempts[key][1] <= now:
                    unlock_attempts.pop(key, None)
    return max(1, math.ceil(delay))


async def get_visible_run(request: Request, run_id: str) -> dict[str, object]:
    try:
        if await asyncio.to_thread(run_library.is_user_run, run_id):
            require_operator_session(request)
        run = await asyncio.to_thread(run_library.get_run, run_id)
    except MalformedRunError as error:
        raise HTTPException(status_code=422, detail="Run metadata is malformed") from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("source") != "Packaged":
        require_operator_session(request)
    return run


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/live")
async def live_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "live.html")


@app.get("/dyno")
async def dyno_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dyno.html")


@app.get("/runs")
async def runs_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "runs.html")


@app.get("/runs/{run_id}")
async def run_dashboard(request: Request, run_id: str) -> FileResponse:
    try:
        if await asyncio.to_thread(run_library.is_user_run, run_id) and not has_operator_session(request):
            return FileResponse(STATIC_DIR / "error.html", status_code=401)
        found = await asyncio.to_thread(run_library.get_run, run_id)
    except MalformedRunError:
        return FileResponse(STATIC_DIR / "error.html", status_code=422)
    except ValueError:
        return FileResponse(STATIC_DIR / "error.html", status_code=404)
    if found is None:
        return FileResponse(STATIC_DIR / "error.html", status_code=404)
    return FileResponse(STATIC_DIR / "run.html")


@app.get("/import")
async def import_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "import.html")


@app.get("/replay/{run_id}")
async def replay_dashboard(run_id: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "replay.html")


@app.get("/compare")
async def compare_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/strategy")
async def strategy_catalog_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "strategy.html")


@app.get("/strategy/{strategy_id}")
async def strategy_dashboard(strategy_id: str) -> FileResponse:
    if strategy_id not in STRATEGY_SPECS:
        return FileResponse(STATIC_DIR / "error.html", status_code=404)
    return FileResponse(STATIC_DIR / "strategy.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "stored_records": len(hub.records),
    }


@app.get("/api/status")
async def dashboard_status(request: Request) -> dict[str, object]:
    latest = await hub.latest_by_source()
    runs = await asyncio.to_thread(run_library.list_runs)
    unlocked = has_operator_session(request)
    visible_runs = runs if unlocked else [run for run in runs if run.get("source") == "Packaged"]
    return {
        "ok": True,
        "stored_records": len(hub.records),
        "run_count": len(visible_runs),
        "locked_uploaded_count": 0 if unlocked else len(runs) - len(visible_runs),
        "latest": latest,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/operator/status")
async def operator_status(request: Request) -> dict[str, object]:
    return {"configured": operator_configured(), "unlocked": has_operator_session(request)}


@app.post("/api/operator/unlock")
async def operator_unlock(request: Request, value: OperatorUnlock) -> Response:
    client = _unlock_client(request)
    retry_after = _unlock_retry_after(client)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=f"Too many unlock attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
    if not hmac.compare_digest(value.key, OPERATOR_KEY):
        retry_after = _record_unlock_failure(client)
        raise HTTPException(
            status_code=401,
            detail="Invalid operator key",
            headers={"Retry-After": str(retry_after)},
        )
    with unlock_attempts_lock:
        unlock_attempts.pop(client, None)
    response = JSONResponse({"unlocked": True, "expires_in_seconds": SESSION_TTL_SECONDS})
    response.set_cookie(
        SESSION_COOKIE,
        _session_token(),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    return response


@app.get("/api/runs")
async def list_runs(request: Request, q: str = Query(default="", max_length=120)) -> dict[str, object]:
    all_runs = await asyncio.to_thread(run_library.list_runs)
    unlocked = has_operator_session(request)
    visible_runs = all_runs if unlocked else [run for run in all_runs if run.get("source") == "Packaged"]
    if q:
        needle = q.casefold()
        visible_runs = [
            run for run in visible_runs
            if needle in " ".join(
                str(run.get(field) or "")
                for field in ("label", "date", "source", "source_detail", "status")
            ).casefold()
        ]
    locked_count = 0 if unlocked else sum(run.get("source") != "Packaged" for run in all_runs)
    return {
        "runs": visible_runs,
        "count": len(visible_runs),
        "locked_uploaded_count": locked_count,
        "uploaded_locked": bool(locked_count),
    }


def _strategy_summary(strategy_id: str, spec: dict[str, object]) -> dict[str, object]:
    files = spec["files"]
    assert isinstance(files, dict)
    available_files = [key for key, path in files.items() if isinstance(path, Path) and path.is_file()]
    return {
        "id": strategy_id,
        "label": spec["label"],
        "description": spec["description"],
        "provenance": spec["provenance"],
        "available": "map" in available_files,
        "files": available_files,
        "href": f"/strategy/{strategy_id}",
    }


def _strategy_samples(path: Path, maximum: int = 500) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            try:
                distance = float(source.get("dist_m") or source.get("distance_m") or "")
                target = float(source.get("target_speed_kph") or "")
                latitude = float(source.get("lat") or "")
                longitude = float(source.get("lon") or "")
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (distance, target, latitude, longitude)):
                continue
            rows.append({
                "distance_m": distance,
                "target_speed_kph": target,
                "latitude": latitude,
                "longitude": longitude,
                "action": str(source.get("action") or "").lower()[:40],
            })
            if len(rows) > 20_000:
                break
    if len(rows) <= maximum:
        return rows
    indices = sorted({round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)})
    return [rows[index] for index in indices]


@app.get("/api/strategies")
async def list_strategies() -> dict[str, object]:
    strategies = [
        await asyncio.to_thread(_strategy_summary, strategy_id, spec)
        for strategy_id, spec in STRATEGY_SPECS.items()
    ]
    return {"strategies": strategies}


@app.get("/api/strategies/{strategy_id}")
async def get_strategy(strategy_id: str) -> dict[str, object]:
    spec = STRATEGY_SPECS.get(strategy_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    summary = await asyncio.to_thread(_strategy_summary, strategy_id, spec)
    files = spec["files"]
    assert isinstance(files, dict)
    map_path = files.get("map")
    if not isinstance(map_path, Path) or not map_path.is_file():
        raise HTTPException(status_code=404, detail="Strategy map not found")
    try:
        samples = await asyncio.to_thread(_strategy_samples, map_path)
    except (OSError, csv.Error) as error:
        raise HTTPException(status_code=422, detail="Strategy map could not be read") from error
    return {"strategy": summary, "samples": samples, "sample_count": len(samples)}


@app.get("/api/strategies/{strategy_id}/files/{file_key}")
async def download_strategy_file(strategy_id: str, file_key: str) -> FileResponse:
    spec = STRATEGY_SPECS.get(strategy_id)
    files = spec.get("files") if spec else None
    path = files.get(file_key) if isinstance(files, dict) else None
    if not isinstance(path, Path) or not path.is_file():
        raise HTTPException(status_code=404, detail="Strategy file not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> dict[str, object]:
    run = await get_visible_run(request, run_id)
    return {"run": run}


@app.get("/api/runs/{run_id}/csv")
async def run_csv(
    request: Request,
    run_id: str,
    file: str = Query(min_length=1, max_length=500),
    offset: int = Query(default=0, ge=0, le=MAX_CSV_VIEW_ROWS),
    limit: int = Query(default=100, ge=1, le=500),
    q: str = Query(default="", max_length=120),
    sort: str | None = Query(default=None, max_length=160),
    direction: Literal["asc", "desc"] = "asc",
) -> dict[str, object]:
    await get_visible_run(request, run_id)
    try:
        return await asyncio.to_thread(
            run_library.csv_page,
            run_id,
            file,
            offset=offset,
            limit=limit,
            query=q,
            sort=sort,
            direction=direction,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="CSV file not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}/gpx")
async def run_gpx(request: Request, run_id: str) -> dict[str, object]:
    await get_visible_run(request, run_id)
    try:
        return await asyncio.to_thread(run_library.gpx_points, run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="GPX route not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail="GPX route could not be read") from error


@app.get("/api/runs/{run_id}/files/{file_path:path}")
async def download_run_file(request: Request, run_id: str, file_path: str) -> FileResponse:
    await get_visible_run(request, run_id)
    try:
        path, download_name = await asyncio.gather(
            asyncio.to_thread(run_library.file_path, run_id, file_path),
            asyncio.to_thread(run_library.download_name, run_id, file_path),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return FileResponse(path, filename=download_name)


@app.get("/api/runs/{run_id}/replay")
async def replay_data(request: Request, run_id: str) -> dict[str, object]:
    await get_visible_run(request, run_id)
    try:
        return await asyncio.to_thread(run_library.replay_payload, run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/compare")
async def compare_runs(
    request: Request,
    run: Annotated[list[str], Query(min_length=1, max_length=80)],
) -> dict[str, object]:
    run_ids = list(dict.fromkeys(run))
    if not 1 <= len(run_ids) <= 2:
        raise HTTPException(status_code=400, detail="Choose one or two runs to compare.")
    for run_id in run_ids:
        await get_visible_run(request, run_id)
    try:
        metrics = await asyncio.gather(
            *(asyncio.to_thread(run_library.comparison_metrics, run_id) for run_id in run_ids)
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"runs": metrics}


def _save_uploads(files: list[UploadFile], directory: Path) -> list[ImportSource]:
    saved: list[ImportSource] = []
    used: set[str] = set()
    total = 0
    for upload in files:
        original = (upload.filename or "").replace("\\", "/")
        relative = PurePosixPath(original)
        if (
            not original
            or original.startswith("/")
            or re.match(r"^[A-Za-z]:", original)
            or any(part in {"", ".", ".."} for part in relative.parts)
            or len(original) > 500
        ):
            raise ValueError("An uploaded filename is unsafe.")
        name = relative.name
        if Path(name).suffix.lower() not in ALLOWED_IMPORT_EXTENSIONS:
            raise ValueError("Choose only CSV, GPX, or ZIP files.")
        suffix = Path(name).suffix
        normalized = original.casefold()
        if normalized in used:
            raise ValueError("Two selected files have the same relative name, ignoring case.")
        used.add(normalized)
        destination = directory / f"upload-{uuid.uuid4().hex}{suffix.lower()}"
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise UploadTooLarge(
                        f"Upload is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                    )
                handle.write(chunk)
        size = destination.stat().st_size
        if suffix.lower() == ".gpx" and size > 50 * 1024 * 1024:
            raise UploadTooLarge("A GPX file is larger than the 50 MB limit.")
        if suffix.lower() == ".csv" and size > 200 * 1024 * 1024:
            raise UploadTooLarge("A CSV file is larger than the 200 MB limit.")
        saved.append(ImportSource(destination, original))
    return saved


@app.post("/api/runs/import", status_code=201)
async def import_runs_api(
    request: Request,
    files: Annotated[list[UploadFile], File(description="CSV, GPX, or ZIP files")],
    name: Annotated[str, Form(min_length=1, max_length=80)],
    date: Annotated[str, Form(pattern=r"^20\d{2}-\d{2}-\d{2}$")],
    source: Annotated[Literal["upload", "sd-card"], Form()] = "upload",
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one file.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Choose no more than {MAX_UPLOAD_FILES} files.")
    with tempfile.TemporaryDirectory(prefix="utsm-dashboard-upload-") as temp:
        try:
            saved = await asyncio.to_thread(_save_uploads, files, Path(temp))
            while not import_lock.acquire(blocking=False):
                await asyncio.sleep(0.02)
            try:
                outcome = await asyncio.to_thread(
                    run_library.import_paths,
                    saved,
                    label=name,
                    date=date,
                    source_kind="SD card" if source == "sd-card" else "Browser upload",
                )
            finally:
                import_lock.release()
        except UploadTooLarge as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ArchiveValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=public_import_error(error)) from error
        except OSError as error:
            raise HTTPException(status_code=400, detail="The uploaded files could not be read safely.") from error
        finally:
            for upload in files:
                with suppress(Exception):
                    await upload.close()
    return {"runs": outcome.runs, "warnings": outcome.warnings}


@app.get("/api/live/recent")
async def recent() -> dict[str, object]:
    return {"records": await hub.recent()}


@app.get("/api/dyno-tests/current")
async def current_dyno_test() -> dict[str, object]:
    return {
        "test": await dyno_tests.current(),
        "latest": await hub.latest_by_source(),
    }


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
