from __future__ import annotations

import csv
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from defusedxml import ElementTree as SafeET

from import_firmware_run import REQUIRED_COLUMNS, import_runs
from utsm_telemetry.safe_archive import ArchiveValidationError, safe_extract_zip, validate_zip


ALLOWED_IMPORT_EXTENSIONS = {".csv", ".gpx", ".zip"}
MAX_CSV_VIEW_ROWS = 200_000
MAX_CSV_COLUMNS = 500
MAX_CSV_ROWS = 5_000_000
MAX_CSV_BYTES = 200 * 1024 * 1024
MAX_GPX_POINTS = 500_000
MAX_GPX_DEPTH = 64
MAX_GPX_BYTES = 50 * 1024 * 1024
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class MalformedRunError(ValueError):
    pass


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "run"


def pretty_name(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", value) if part)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("The requested file is outside the run folder.")
    return candidate


def _normalize_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"_manifest_invalid": True, "manifest_warning": "run.json must contain an object"}
    normalized: dict[str, Any] = {}
    invalid = False

    def optional_string(key: str, maximum: int = 500) -> None:
        nonlocal invalid
        if key not in payload:
            return
        value = payload.get(key)
        if isinstance(value, str):
            normalized[key] = value[:maximum]
        else:
            invalid = True

    for key, maximum in (
        ("label", 80), ("date", 10), ("source", 500), ("imported_at", 80),
        ("import_kind", 40), ("split_method", 40), ("batch_id", 32),
    ):
        optional_string(key, maximum)
    if "date" in normalized and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", normalized["date"]):
        normalized.pop("date")
        invalid = True
    for key in ("laps",):
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[key] = value
        else:
            invalid = True
    if "distance_m" in payload:
        value = payload.get("distance_m")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            normalized["distance_m"] = float(value)
        else:
            invalid = True
    for key in ("source_files", "lap_boundary_gpx_indices", "source_timestamp_ms"):
        if key not in payload:
            continue
        value = payload.get(key)
        if not isinstance(value, list):
            invalid = True
            continue
        if key == "source_files" and all(isinstance(item, str) for item in value):
            normalized[key] = [item[:500] for item in value[:1000]]
        elif key != "source_files" and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            normalized[key] = value[:1_000_000]
        else:
            invalid = True
    if "import_warnings" in payload:
        warning_value = payload.get("import_warnings")
        if isinstance(warning_value, list):
            warnings: list[dict[str, str]] = []
            for item in warning_value:
                if isinstance(item, str):
                    message = item[:500]
                    code = "no-gps" if re.search(r"\b(?:gpx|gps)\b", message, re.IGNORECASE) else f"warning-{len(warnings)}"
                    warnings.append({"code": code, "message": message})
                elif isinstance(item, dict) and isinstance(item.get("code"), str) and isinstance(item.get("message"), str):
                    warnings.append({"code": item["code"][:80], "message": item["message"][:500]})
                else:
                    invalid = True
            normalized["import_warnings"] = warnings
        else:
            invalid = True
    if "original_uploads" in payload:
        originals_value = payload.get("original_uploads")
        originals: list[dict[str, Any]] = []
        if not isinstance(originals_value, list):
            invalid = True
        else:
            for item in originals_value:
                if not isinstance(item, dict):
                    invalid = True
                    continue
                fields = {
                    "upload_id": item.get("upload_id"), "original_name": item.get("original_name"),
                    "size_bytes": item.get("size_bytes"), "sha256": item.get("sha256"),
                    "imported_at": item.get("imported_at"), "source": item.get("source"),
                }
                if not (
                    isinstance(fields["upload_id"], str)
                    and isinstance(fields["original_name"], str)
                    and isinstance(fields["size_bytes"], int) and fields["size_bytes"] >= 0
                    and isinstance(fields["sha256"], str)
                    and isinstance(fields["imported_at"], str)
                    and isinstance(fields["source"], str)
                ):
                    invalid = True
                    continue
                originals.append({
                    "upload_id": fields["upload_id"][:80], "original_name": fields["original_name"][:500],
                    "size_bytes": fields["size_bytes"], "sha256": fields["sha256"][:64],
                    "imported_at": fields["imported_at"][:80], "source": fields["source"][:80],
                })
            normalized["original_uploads"] = originals
    if invalid:
        normalized["_manifest_invalid"] = True
        normalized["manifest_warning"] = "run.json contains invalid field types"
    return normalized


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"_manifest_invalid": True, "manifest_warning": "run.json could not be read"}
    return _normalize_manifest(payload)


def _date_from_text(value: str) -> str | None:
    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", value)
    return "-".join(match.groups()) if match else None


def _gpx_summary(path: Path) -> dict[str, Any]:
    try:
        root = SafeET.parse(path, forbid_dtd=True, forbid_entities=True, forbid_external=True).getroot()
        points = list(root.iterfind(".//{*}trkpt"))
        times = [point.findtext("{*}time") for point in points]
        valid_times = [value for value in times if value]
        return {
            "gps_points": len(points),
            "timed_points": len(valid_times),
            "started_at": valid_times[0] if valid_times else None,
            "ended_at": valid_times[-1] if valid_times else None,
        }
    except (OSError, SafeET.ParseError, SafeET.DTDForbidden, SafeET.EntitiesForbidden, SafeET.ExternalReferenceForbidden):
        return {"gps_points": 0, "timed_points": 0, "gpx_warning": "GPX could not be read"}


def _file_record(run_dir: Path, path: Path, *, packaged: bool) -> dict[str, Any]:
    relative = path.relative_to(run_dir).as_posix()
    suffix = path.suffix.lower()
    data_kind = "packaged" if packaged else ("canonical" if suffix in {".csv", ".gpx"} else "metadata")
    return {
        "name": path.name,
        "display_name": path.name,
        "path": relative,
        "type": suffix.removeprefix(".").upper() or "FILE",
        "size_bytes": path.stat().st_size,
        "is_raw": False,
        "data_kind": data_kind,
        "viewable": suffix == ".csv",
    }


@dataclass(frozen=True)
class ImportOutcome:
    runs: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True)
class ImportSource:
    path: Path
    original_name: str


class RunLibrary:
    """Merged packaged and writable run catalog.

    Packaged runs are read-only inputs from ``data/runs``. Browser imports are
    committed under ``.dashboard-data/runs`` and original bytes live once in a
    content-addressed blob directory. Reference tracks are outside both roots.
    """

    def __init__(
        self,
        packaged_dir: Path,
        user_runs_dir: Path | None = None,
        blob_dir: Path | None = None,
        strategy_dir: Path | None = None,
    ) -> None:
        if user_runs_dir is None:
            self.packaged_dir: Path | None = None
            self.runs_dir = packaged_dir.resolve()
        else:
            self.packaged_dir = packaged_dir.resolve()
            self.runs_dir = user_runs_dir.resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir = (blob_dir or self.runs_dir.parent / "blobs").resolve()
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        if strategy_dir is not None:
            self.strategy_dir = strategy_dir.resolve()
        elif self.packaged_dir is not None:
            self.strategy_dir = self.packaged_dir.parent / "strategy"
        else:
            self.strategy_dir = None
        self.import_lock_path = self.runs_dir.parent / ".import.lock"
        self._cache_lock = threading.RLock()
        self._catalog_signature: tuple[Any, ...] | None = None
        self._catalog_cache: list[dict[str, Any]] | None = None
        self._gpx_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._csv_profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    @contextmanager
    def _cross_process_import_lock(self):
        """Serialize writers that use separate Uvicorn worker processes."""
        self.import_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.import_lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _strategy_for_run(self, run_dir: Path) -> dict[str, str] | None:
        if (
            self._is_packaged(run_dir)
            and run_dir.name == "afternoon-run"
            and self.strategy_dir is not None
            and (self.strategy_dir / "indy_strategy_map.csv").is_file()
        ):
            return {
                "id": "indy",
                "href": "/strategy/indy",
                "provenance": "Generated from the packaged Afternoon Run for the Indianapolis course.",
            }
        return None

    def _csv_profile(self, path: Path) -> dict[str, Any]:
        stat_result = path.stat()
        key = (str(path.resolve()), stat_result.st_mtime_ns, stat_result.st_size)
        with self._cache_lock:
            cached = self._csv_profile_cache.get(key)
        if cached is not None:
            return dict(cached)
        numeric_columns: set[str] = set()
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                for index, row in enumerate(reader):
                    if index >= 200:
                        break
                    for column in columns:
                        value = str(row.get(column) or "").strip()
                        if not value:
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        if math.isfinite(number):
                            numeric_columns.add(column)
        except (OSError, csv.Error):
            columns = []
        result = {"columns": columns, "numeric_columns": sorted(numeric_columns), "has_numeric_data": bool(numeric_columns)}
        with self._cache_lock:
            self._csv_profile_cache = {key: result, **{item: value for item, value in self._csv_profile_cache.items() if item[0] != key[0]}}
        return dict(result)

    def _validate_id(self, run_id: str) -> str:
        if run_id != slugify(run_id):
            raise ValueError("Invalid run ID.")
        return run_id

    def run_dir(self, run_id: str) -> Path:
        run_id = self._validate_id(run_id)
        user = safe_child(self.runs_dir, run_id)
        if user.is_dir():
            return user
        if self.packaged_dir is not None:
            packaged = safe_child(self.packaged_dir, run_id)
            if packaged.is_dir():
                return packaged
        return user

    def is_user_run(self, run_id: str) -> bool:
        """Check the protected writable namespace without reading run metadata."""
        run_id = self._validate_id(run_id)
        return safe_child(self.runs_dir, run_id).is_dir()

    def _is_packaged(self, run_dir: Path) -> bool:
        return self.packaged_dir is not None and self.packaged_dir in run_dir.resolve().parents

    def _cached_gpx_summary(self, path: Path) -> dict[str, Any]:
        stat_result = path.stat()
        key = (str(path.resolve()), stat_result.st_mtime_ns, stat_result.st_size)
        with self._cache_lock:
            cached = self._gpx_cache.get(key)
        if cached is None:
            cached = _gpx_summary(path)
            with self._cache_lock:
                self._gpx_cache = {key: cached, **{item: value for item, value in self._gpx_cache.items() if item[0] != key[0]}}
        return dict(cached)

    def _original_records(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in manifest.get("original_uploads", []):
            if not isinstance(item, dict):
                continue
            upload_id = str(item.get("upload_id", ""))
            digest = str(item.get("sha256", ""))
            if not re.fullmatch(r"upload-[0-9a-f]{16}", upload_id) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            original_name = str(item.get("original_name") or "Original upload")[:500]
            suffix = Path(original_name).suffix.lower()
            records.append({
                "name": Path(original_name.replace("\\", "/")).name or "Original upload",
                "display_name": original_name,
                "path": f"originals/{upload_id}",
                "type": suffix.removeprefix(".").upper() or "FILE",
                "size_bytes": int(item.get("size_bytes", 0)),
                "is_raw": True,
                "data_kind": "original",
                "viewable": suffix == ".csv",
                "sha256": digest,
            })
        return records

    def _summary(self, run_dir: Path, include_files: bool = False) -> dict[str, Any]:
        loaded_manifest = _read_manifest(run_dir)
        malformed = bool(loaded_manifest.get("_manifest_invalid"))
        manifest = {} if malformed else loaded_manifest
        packaged = self._is_packaged(run_dir)
        top_csv = sorted(run_dir.glob("*.csv"))
        original_records = self._original_records(manifest)
        original_csv = [record for record in original_records if record["viewable"]]
        gpx_files = sorted(run_dir.glob("*.gpx"))
        display_gpx = gpx_files[0] if gpx_files else None
        csv_count = len(top_csv) + len(original_csv)
        date = (
            str(manifest.get("date")) if manifest.get("date") else
            _date_from_text(str(manifest.get("source", ""))) or
            _date_from_text(run_dir.name)
        )
        modified = datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat()
        analysis_ready = bool(gpx_files and top_csv and not malformed)
        status = "Ready" if analysis_ready else "Raw data"
        warning_records: dict[str, str] = {}
        if malformed:
            warning_records["manifest"] = str(loaded_manifest.get("manifest_warning") or "run.json could not be read")
        if not csv_count:
            status = "Needs data"
            warning_records["no-csv"] = "No CSV file was found in this run."
        if csv_count and not display_gpx:
            warning_records["no-gps"] = "No GPX route is available. Data remains available for table review."
        for item in manifest.get("import_warnings", []):
            if isinstance(item, dict) and item.get("message"):
                warning_records[str(item.get("code") or f"warning-{len(warning_records)}")] = str(item["message"])
        warnings = list(warning_records.values())
        if packaged:
            source = "Packaged"
        elif str(manifest.get("source", "")).lower() == "sd card":
            source = "SD Card"
        else:
            source = "Uploaded"
        strategy = self._strategy_for_run(run_dir)
        data_available = bool(csv_count and not malformed)
        primary_value: Path | str | None = top_csv[0] if top_csv else (original_csv[0]["path"] if original_csv else None)
        profile_path: Path | None = primary_value if isinstance(primary_value, Path) else None
        if isinstance(primary_value, str) and original_csv:
            digest = str(original_csv[0].get("sha256") or "")
            candidate = safe_child(self.blob_dir, digest) if re.fullmatch(r"[0-9a-f]{64}", digest) else None
            profile_path = candidate if candidate is not None and candidate.is_file() else None
        csv_profile = self._csv_profile(profile_path) if profile_path is not None else {"has_numeric_data": False, "numeric_columns": []}
        charts_available = bool(data_available and csv_profile["has_numeric_data"])
        gpx_summary = self._cached_gpx_summary(display_gpx) if display_gpx is not None and not malformed else {}
        replay_available = bool(
            display_gpx and top_csv and not malformed
            and int(gpx_summary.get("timed_points") or 0) >= 2
            and any(column in csv_profile.get("columns", []) for column in ("timestamp_ms", "source_timestamp_ms"))
        )
        strategy_available = bool(strategy and not malformed)
        compare_available = bool(data_available)
        if malformed:
            status = "Needs attention"
        actions = {
            "data": {"available": data_available, "reason": None if data_available else "No readable CSV data is available.", "href": f"/runs/{run_dir.name}#dataSection" if data_available else None},
            "charts": {"available": charts_available, "reason": None if charts_available else ("Charts need at least one numeric CSV value." if data_available else "Charts need CSV data."), "href": f"/runs/{run_dir.name}#chartSection" if charts_available else None},
            "replay": {"available": replay_available, "reason": None if replay_available else "Replay needs canonical telemetry and a timed GPX route.", "href": f"/replay/{run_dir.name}" if replay_available else None},
            "strategy": {"available": strategy_available, "reason": None if strategy_available else "No run-derived strategy artifact is associated with this run.", "href": strategy["href"] if strategy_available else None},
            "compare": {"available": compare_available, "reason": None if compare_available else "Comparison needs CSV data.", "href": f"/compare?run={run_dir.name}" if compare_available else None},
        }
        result: dict[str, Any] = {
            "id": run_dir.name,
            "label": str(manifest.get("label") or pretty_name(run_dir.name)),
            "date": date,
            "source": source,
            "source_detail": str(manifest.get("source") or "Included with dashboard"),
            "imported_at": manifest.get("imported_at"),
            "modified_at": modified,
            "status": status,
            "analysis_ready": analysis_ready,
            "has_gps": display_gpx is not None,
            "csv_count": csv_count,
            "laps": manifest.get("laps"),
            "distance_m": manifest.get("distance_m"),
            "gps_points": None,
            "started_at": None,
            "ended_at": None,
            "warnings": warnings,
            "warning_codes": list(warning_records),
            "malformed": malformed,
            "actions": actions,
            "capabilities": {
                "view_data": data_available,
                "view_charts": charts_available,
                "view_map": display_gpx is not None and not malformed,
                "download_original": bool(original_records),
                "replay": replay_available,
                "strategy": strategy_available,
                "compare": compare_available,
            },
            "strategy": strategy,
            "numeric_columns": csv_profile.get("numeric_columns", []),
            "primary_csv": primary_value,
            "primary_gpx": display_gpx,
        }
        if gpx_summary:
            result.update(gpx_summary)
        if include_files:
            files = sorted(
                (path for path in run_dir.rglob("*") if path.is_file() and "raw" not in path.relative_to(run_dir).parts),
                key=lambda path: path.relative_to(run_dir).as_posix().lower(),
            )
            result["files"] = [_file_record(run_dir, path, packaged=packaged) for path in files] + original_records
            result["manifest"] = manifest
        result["primary_csv"] = (
            result["primary_csv"].relative_to(run_dir).as_posix()
            if isinstance(result["primary_csv"], Path) else result["primary_csv"]
        )
        result["primary_gpx"] = (
            result["primary_gpx"].relative_to(run_dir).as_posix()
            if isinstance(result["primary_gpx"], Path) else None
        )
        return result

    def _catalog_state(self) -> tuple[Any, ...]:
        state: list[Any] = []
        for root in (self.packaged_dir, self.runs_dir):
            if root is None or not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if path.is_dir() and not path.name.startswith((".", "_")):
                    stat_result = path.stat()
                    state.append((str(root), path.name, stat_result.st_mtime_ns))
        if self.runs_dir.is_dir():
            for marker in sorted(self.runs_dir.glob("_batch-*.pending")):
                if marker.is_file():
                    stat_result = marker.stat()
                    state.append(("pending-batch", marker.name, stat_result.st_mtime_ns, stat_result.st_size))
        return tuple(state)

    def _batch_pending(self, run_dir: Path) -> bool:
        batch_id = _read_manifest(run_dir).get("batch_id")
        return bool(
            isinstance(batch_id, str)
            and re.fullmatch(r"[0-9a-f]{32}", batch_id)
            and (self.runs_dir / f"_batch-{batch_id}.pending").is_file()
        )

    def invalidate(self) -> None:
        with self._cache_lock:
            self._catalog_signature = None
            self._catalog_cache = None

    def list_runs(self, query: str = "") -> list[dict[str, Any]]:
        query = query.strip().lower()
        signature = self._catalog_state()
        with self._cache_lock:
            cached = self._catalog_cache if signature == self._catalog_signature else None
        if cached is None:
            runs: list[dict[str, Any]] = []
            roots = [root for root in (self.packaged_dir, self.runs_dir) if root is not None and root.is_dir()]
            seen: set[str] = set()
            for root in roots:
                for path in sorted(root.iterdir()):
                    if not path.is_dir() or path.name.startswith((".", "_")) or path.name in seen or self._batch_pending(path):
                        continue
                    seen.add(path.name)
                    try:
                        runs.append(self._summary(path))
                    except (OSError, ValueError, TypeError) as error:
                        runs.append({
                            "id": path.name,
                            "label": pretty_name(path.name),
                            "date": None,
                            "source": "Packaged" if self._is_packaged(path) else "Uploaded",
                            "source_detail": "Run metadata could not be read",
                            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            "status": "Needs data",
                            "analysis_ready": False,
                            "has_gps": False,
                            "csv_count": 0,
                            "laps": None,
                            "distance_m": None,
                            "warnings": ["This run is malformed and was isolated from the rest of the catalog."],
                            "capabilities": {"view_data": False, "view_charts": False, "view_map": False, "download_original": False, "replay": False, "strategy": False, "compare": False},
                            "actions": {},
                            "primary_csv": None,
                            "primary_gpx": None,
                        })
            runs.sort(key=lambda item: (item.get("date") or "", item.get("modified_at") or ""), reverse=True)
            with self._cache_lock:
                self._catalog_signature = signature
                self._catalog_cache = runs
            cached = runs
        results = []
        for item in cached:
            haystack = " ".join(
                str(item.get(key) or "")
                for key in ("id", "label", "date", "source", "source_detail", "status")
            ).lower()
            if not query or query in haystack:
                results.append(dict(item))
        return results

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id)
        if not path.is_dir() or self._batch_pending(path):
            return None
        run = self._summary(path, include_files=True)
        if run.get("malformed"):
            raise MalformedRunError("Run metadata is malformed.")
        return run

    def file_path(self, run_id: str, relative: str) -> Path:
        if relative.startswith("originals/"):
            upload_id = relative.removeprefix("originals/")
            run = self.get_run(run_id)
            if run is None:
                raise FileNotFoundError(relative)
            for item in run.get("manifest", {}).get("original_uploads", []):
                if item.get("upload_id") == upload_id and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                    path = safe_child(self.blob_dir, str(item["sha256"]))
                    if path.is_file():
                        return path
            raise FileNotFoundError(relative)
        path = safe_child(self.run_dir(run_id), relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def download_name(self, run_id: str, relative: str) -> str:
        run = self.get_run(run_id)
        if run is None:
            raise FileNotFoundError(relative)
        record = next((item for item in run.get("files", []) if item.get("path") == relative), None)
        if record is None:
            raise FileNotFoundError(relative)
        original = Path(str(record.get("display_name", record.get("name", "download"))).replace("\\", "/")).name
        clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original).rstrip(" .")
        if not clean:
            clean = "download"
        stem = clean.split(".", 1)[0].rstrip(" .").upper()
        if stem in WINDOWS_RESERVED_NAMES:
            clean = f"_{clean}"
        if len(clean) > 180:
            suffix = Path(clean).suffix[:20]
            clean = clean[: max(1, 180 - len(suffix))] + suffix
        return clean

    def csv_page(
        self,
        run_id: str,
        relative: str,
        *,
        offset: int = 0,
        limit: int = 100,
        query: str = "",
        sort: str | None = None,
        direction: str = "asc",
    ) -> dict[str, Any]:
        path = self.file_path(run_id, relative)
        run = self.get_run(run_id)
        record = next((item for item in (run or {}).get("files", []) if item.get("path") == relative), None)
        if record is None or not record.get("viewable"):
            raise ValueError("Only CSV files can be viewed as a table.")
        page_rows: list[dict[str, str]] = []
        truncated = False
        limit = min(max(limit, 1), 500)
        offset = max(offset, 0)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if not columns:
                raise ValueError("The CSV has no header row.")
            if sort and sort not in columns:
                raise ValueError("The requested sort column does not exist.")
            needle = query.strip().lower()
            if not sort:
                total = 0
                for index, row in enumerate(reader):
                    if index >= MAX_CSV_VIEW_ROWS:
                        truncated = True
                        break
                    normalized = {column: str(row.get(column) or "") for column in columns}
                    if needle and not any(needle in value.lower() for value in normalized.values()):
                        continue
                    if offset <= total < offset + limit:
                        page_rows.append(normalized)
                    total += 1
            else:
                with tempfile.TemporaryDirectory(prefix="utsm-csv-sort-") as temp:
                    database = sqlite3.connect(Path(temp) / "sort.db")
                    try:
                        database.execute(
                            "CREATE TABLE rows (numeric_kind INTEGER, numeric_value REAL, text_value TEXT, payload TEXT)"
                        )
                        total = 0
                        batch: list[tuple[int, float | None, str, str]] = []
                        for index, row in enumerate(reader):
                            if index >= MAX_CSV_VIEW_ROWS:
                                truncated = True
                                break
                            normalized = {column: str(row.get(column) or "") for column in columns}
                            if needle and not any(needle in value.lower() for value in normalized.values()):
                                continue
                            value = normalized[sort]
                            try:
                                key = (0, float(value), "")
                            except ValueError:
                                key = (1, None, value.lower())
                            batch.append((*key, json.dumps(normalized, separators=(",", ":"))))
                            total += 1
                            if len(batch) >= 1000:
                                database.executemany("INSERT INTO rows VALUES (?, ?, ?, ?)", batch)
                                batch.clear()
                        if batch:
                            database.executemany("INSERT INTO rows VALUES (?, ?, ?, ?)", batch)
                        order = "DESC" if direction == "desc" else "ASC"
                        cursor = database.execute(
                            f"SELECT payload FROM rows ORDER BY numeric_kind ASC, numeric_value {order}, text_value {order} LIMIT ? OFFSET ?",
                            (limit, offset),
                        )
                        page_rows = [json.loads(payload) for (payload,) in cursor]
                    finally:
                        database.close()
        return {
            "file": relative,
            "columns": columns,
            "rows": page_rows,
            "offset": offset,
            "limit": limit,
            "total": total,
            "truncated": truncated,
        }

    def is_imported(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        return bool(run and run.get("source") != "Packaged")

    def gpx_points(self, run_id: str, maximum: int = 10_000) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None or not run.get("primary_gpx"):
            raise FileNotFoundError("GPX route not found")
        path = self.file_path(run_id, str(run["primary_gpx"]))
        try:
            root = SafeET.parse(path, forbid_dtd=True, forbid_entities=True, forbid_external=True).getroot()
            all_points = [
                [float(point.attrib["lat"]), float(point.attrib["lon"])]
                for point in root.iterfind(".//{*}trkpt")
            ]
        except (
            OSError,
            SafeET.ParseError,
            SafeET.DTDForbidden,
            SafeET.EntitiesForbidden,
            SafeET.ExternalReferenceForbidden,
            KeyError,
            ValueError,
        ) as error:
            raise ValueError("GPX route could not be read.") from error
        step = max(1, len(all_points) // max(maximum, 1))
        points = all_points[::step]
        if all_points and points[-1] != all_points[-1]:
            points.append(all_points[-1])
        return {"points": points, "total_points": len(all_points)}

    def _sample_csv(self, run_id: str, relative: str, maximum: int = 2_000) -> list[dict[str, str]]:
        path = self.file_path(run_id, relative)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            total = sum(1 for _ in csv.DictReader(handle))
        if total == 0 or maximum <= 0:
            return []
        target_count = min(total, maximum)
        targets = (
            [0] if target_count == 1
            else sorted({round(index * (total - 1) / (target_count - 1)) for index in range(target_count)})
        )
        sampled: list[dict[str, str]] = []
        target_index = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if target_index < len(targets) and index == targets[target_index]:
                    sampled.append({str(key): str(value or "") for key, value in row.items() if key is not None})
                    target_index += 1
                if target_index >= len(targets):
                    break
        return sampled

    def _timed_gpx_points(self, run_id: str) -> list[tuple[float, float, float]]:
        run = self.get_run(run_id)
        if run is None or not run.get("primary_gpx"):
            raise FileNotFoundError("GPX route not found")
        path = self.file_path(run_id, str(run["primary_gpx"]))
        try:
            root = SafeET.parse(path, forbid_dtd=True, forbid_entities=True, forbid_external=True).getroot()
            raw: list[tuple[float, float, datetime]] = []
            for point in root.iterfind(".//{*}trkpt"):
                time_text = point.findtext("{*}time")
                if not time_text:
                    continue
                parsed = datetime.fromisoformat(time_text.replace("Z", "+00:00"))
                raw.append((float(point.attrib["lat"]), float(point.attrib["lon"]), parsed))
        except (OSError, SafeET.ParseError, SafeET.DTDForbidden, SafeET.EntitiesForbidden, SafeET.ExternalReferenceForbidden, KeyError, TypeError, ValueError) as error:
            raise ValueError("Timed GPX route could not be read.") from error
        if len(raw) < 2:
            raise ValueError("Replay needs at least two timed GPX points.")
        first = raw[0][2]
        return [(lat, lon, (time_value - first).total_seconds() * 1000.0) for lat, lon, time_value in raw]

    def replay_payload(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise FileNotFoundError("Run not found")
        if not run.get("capabilities", {}).get("replay"):
            raise ValueError("Replay needs canonical telemetry and a GPX route.")
        points = self._timed_gpx_points(run_id)
        rows = self._sample_csv(run_id, str(run["primary_csv"]), maximum=2_000)
        if not rows:
            raise ValueError("Replay has no telemetry rows.")
        row_times: list[float] = []
        first_row_time: float | None = None
        for row in rows:
            timestamp: float | None = None
            for name in ("timestamp_ms", "source_timestamp_ms"):
                try:
                    timestamp = float(row[name])
                    break
                except (KeyError, TypeError, ValueError):
                    continue
            if timestamp is None:
                raise ValueError("Replay needs a telemetry timestamp column.")
            first_row_time = timestamp if first_row_time is None else first_row_time
            row_times.append(timestamp - first_row_time)
        point_times = [point[2] for point in points]
        samples: list[dict[str, Any]] = []
        for index, (row, elapsed_ms) in enumerate(zip(rows, row_times)):
            insertion = bisect.bisect_left(point_times, elapsed_ms)
            candidates = [value for value in (insertion - 1, insertion) if 0 <= value < len(points)]
            point_index = min(candidates, key=lambda value: abs(point_times[value] - elapsed_ms))
            if index < len(rows) - 1:
                point_index = min(point_index, len(points) - 2)
            else:
                point_index = len(points) - 1
            point = points[point_index]

            def number(*names: str) -> float | None:
                for name in names:
                    try:
                        return float(row[name]) if row.get(name, "") != "" else None
                    except (KeyError, ValueError):
                        continue
                return None

            samples.append({
                "lat": point[0],
                "lon": point[1],
                "route_progress_percent": point_index / (len(points) - 1) * 100.0,
                "timestamp_ms": number("timestamp_ms", "source_timestamp_ms"),
                "speed_kph": number("wheel_speed_kph", "wheel_speed_kmph", "gps_speed_kmph", "speed_kph"),
                "current_A": (value / 1000.0) if (value := number("current_mA")) is not None else None,
                "voltage_V": (value / 1000.0) if (value := number("voltage_mV")) is not None else None,
                "temperature_C": number("motor_temperature_C", "temperature_C"),
            })
        return {"run": {"id": run["id"], "label": run["label"]}, "samples": samples}

    def comparison_metrics(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None or not run.get("primary_csv"):
            raise ValueError("Comparison needs CSV data.")
        path = self.file_path(run_id, str(run["primary_csv"]))
        row_count = 0
        aggregates = {
            "current": {"total": 0.0, "count": 0, "peak": None},
            "speed": {"total": 0.0, "count": 0, "peak": None},
            "power": {"total": 0.0, "count": 0, "peak": None},
        }
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if row_count >= MAX_CSV_VIEW_ROWS:
                    break
                row_count += 1
                for names, key, scale in (
                    (("current_mA",), "current", 0.001),
                    (("wheel_speed_kph", "wheel_speed_kmph", "speed_kph"), "speed", 1.0),
                ):
                    for name in names:
                        try:
                            value = float(row[name]) * scale
                            aggregate = aggregates[key]
                            aggregate["total"] += value
                            aggregate["count"] += 1
                            aggregate["peak"] = value if aggregate["peak"] is None else max(aggregate["peak"], value)
                            break
                        except (KeyError, TypeError, ValueError):
                            continue
                power_value: float | None = None
                for name in ("power_W", "power_w"):
                    try:
                        power_value = float(row[name])
                        break
                    except (KeyError, TypeError, ValueError):
                        continue
                if power_value is None:
                    try:
                        power_value = float(row["current_mA"]) * float(row["voltage_mV"]) / 1_000_000.0
                    except (KeyError, TypeError, ValueError):
                        pass
                if power_value is not None and math.isfinite(power_value):
                    aggregate = aggregates["power"]
                    aggregate["total"] += power_value
                    aggregate["count"] += 1
                    aggregate["peak"] = power_value if aggregate["peak"] is None else max(aggregate["peak"], power_value)

        def average(key: str) -> float | None:
            aggregate = aggregates[key]
            return aggregate["total"] / aggregate["count"] if aggregate["count"] else None

        return {
            "id": run["id"],
            "label": run["label"],
            "date": run.get("date"),
            "source": run.get("source"),
            "laps": run.get("laps"),
            "distance_m": run.get("distance_m"),
            "rows": row_count,
            "average_speed_kph": average("speed"),
            "peak_speed_kph": aggregates["speed"]["peak"],
            "average_current_A": average("current"),
            "peak_current_A": aggregates["current"]["peak"],
            "average_power_W": average("power"),
            "peak_power_W": aggregates["power"]["peak"],
        }

    @staticmethod
    def validate_zip(path: Path) -> None:
        validate_zip(path)

    @staticmethod
    def _looks_like_firmware_csv(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            return False
        return REQUIRED_COLUMNS.issubset(set(header))

    @staticmethod
    def validate_csv(path: Path) -> None:
        try:
            if path.stat().st_size > MAX_CSV_BYTES:
                raise ValueError(f"CSV is larger than the {MAX_CSV_BYTES // (1024 * 1024)} MB limit: {path.name}")
            with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header or not any(value.strip() for value in header):
                    raise ValueError(f"CSV has no header row: {path.name}")
                if len(header) > MAX_CSV_COLUMNS:
                    raise ValueError(f"CSV has more than {MAX_CSV_COLUMNS} columns: {path.name}")
                normalized = [value.strip().lower() for value in header]
                if len(normalized) != len(set(normalized)):
                    raise ValueError(f"CSV has duplicate column names: {path.name}")
                for index, row in enumerate(reader, start=1):
                    if index > MAX_CSV_ROWS:
                        raise ValueError(f"CSV has more than {MAX_CSV_ROWS:,} rows: {path.name}")
                    if len(row) != len(header):
                        raise ValueError(f"CSV row {index + 1} has the wrong number of columns: {path.name}")
                    if any("\x00" in value for value in row):
                        raise ValueError(f"CSV contains a null byte: {path.name}")
        except UnicodeDecodeError as error:
            raise ValueError(f"CSV is not valid UTF-8: {path.name}") from error
        except csv.Error as error:
            raise ValueError(f"CSV could not be parsed: {path.name}") from error

    @staticmethod
    def validate_gpx(path: Path) -> None:
        try:
            if path.stat().st_size > MAX_GPX_BYTES:
                raise ValueError(f"GPX is larger than the {MAX_GPX_BYTES // (1024 * 1024)} MB limit: {path.name}")
            root = SafeET.parse(
                path,
                forbid_dtd=True,
                forbid_entities=True,
                forbid_external=True,
            ).getroot()
            point_count = 0
            stack: list[tuple[Any, int]] = [(root, 1)]
            while stack:
                element, depth = stack.pop()
                if depth > MAX_GPX_DEPTH:
                    raise ValueError(f"GPX exceeds the {MAX_GPX_DEPTH}-level depth limit: {path.name}")
                if element.tag.endswith("trkpt"):
                    point_count += 1
                    if point_count > MAX_GPX_POINTS:
                        raise ValueError(f"GPX has more than {MAX_GPX_POINTS:,} points: {path.name}")
                    latitude = float(element.attrib["lat"])
                    longitude = float(element.attrib["lon"])
                    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                        raise ValueError(f"GPX contains an invalid coordinate: {path.name}")
                stack.extend((child, depth + 1) for child in element)
            if point_count == 0:
                raise ValueError(f"GPX contains no track points: {path.name}")
        except (
            SafeET.ParseError,
            SafeET.DTDForbidden,
            SafeET.EntitiesForbidden,
            SafeET.ExternalReferenceForbidden,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, ValueError) and str(error).startswith("GPX "):
                raise
            raise ValueError(f"GPX could not be parsed: {path.name}") from error

    def _store_sources(
        self,
        sources: Iterable[ImportSource],
        source_kind: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source in sources:
            digest = hashlib.sha256()
            size = 0
            with source.path.open("rb") as input_handle:
                while chunk := input_handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            hexdigest = digest.hexdigest()
            destination = safe_child(self.blob_dir, hexdigest)
            if not destination.exists():
                temporary = safe_child(self.blob_dir, f"_blob-{uuid.uuid4().hex}")
                try:
                    with source.path.open("rb") as input_handle, temporary.open("xb") as output_handle:
                        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                    if temporary.stat().st_size != size:
                        raise OSError("Blob size verification failed.")
                    try:
                        os.replace(temporary, destination)
                    except FileExistsError:
                        temporary.unlink(missing_ok=True)
                finally:
                    temporary.unlink(missing_ok=True)
            records.append({
                "upload_id": f"upload-{uuid.uuid4().hex[:16]}",
                "original_name": source.original_name[:500],
                "size_bytes": size,
                "sha256": hexdigest,
                "imported_at": utc_now(),
                "source": source_kind,
            })
        return records

    @staticmethod
    def _sources(paths: Iterable[Path | ImportSource]) -> list[ImportSource]:
        return [
            item if isinstance(item, ImportSource) else ImportSource(Path(item), Path(item).name)
            for item in paths
        ]

    @staticmethod
    def _combine_csv(paths: list[Path], output: Path) -> None:
        header: list[str] | None = None
        with output.open("w", encoding="utf-8", newline="") as target:
            writer: csv.writer | None = None
            for path in paths:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
                    reader = csv.reader(source)
                    current = next(reader, None)
                    if not current:
                        continue
                    if header is None:
                        header = current
                        writer = csv.writer(target)
                        writer.writerow(header)
                    elif current != header:
                        raise ValueError("Uploaded CSV files do not have matching columns.")
                    assert writer is not None
                    writer.writerows(reader)
        if header is None:
            raise ValueError("No CSV header was found.")

    def _write_manifest(self, run_dir: Path, payload: dict[str, Any]) -> None:
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _stage(self) -> Path:
        path = self.runs_dir / f"_staging-{uuid.uuid4().hex}"
        path.mkdir()
        return path

    def _commit(self, stage: Path) -> Path:
        for _ in range(10):
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            destination = self.runs_dir / run_id
            if not destination.exists():
                os.replace(stage, destination)
                self.invalidate()
                return destination
        raise RuntimeError("Could not allocate a run ID.")

    def _import_gpx_csv(
        self,
        gpx: Path,
        csv_paths: list[Path],
        all_sources: list[ImportSource],
        label: str,
        date: str,
        source_kind: str,
    ) -> dict[str, Any]:
        run_dir = self._stage()
        try:
            shutil.copy2(gpx, run_dir / "route.gpx")
            self._combine_csv(csv_paths, run_dir / "telemetry.csv")
            originals = self._store_sources(all_sources, source_kind)
            self._write_manifest(run_dir, {
                "label": label,
                "date": date,
                "source": source_kind,
                "source_files": [source.original_name for source in all_sources],
                "imported_at": utc_now(),
                "import_kind": "gpx_csv",
                "original_uploads": originals,
            })
            run_dir = self._commit(run_dir)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return self._summary(run_dir)

    def _import_raw(
        self,
        sources: list[ImportSource],
        label: str,
        date: str,
        source_kind: str,
        warning: str,
        derived_files: list[Path] | None = None,
    ) -> dict[str, Any]:
        run_dir = self._stage()
        try:
            originals = self._store_sources(sources, source_kind)
            if derived_files:
                csv_index = 0
                gpx_index = 0
                for path in derived_files:
                    if path.suffix.lower() == ".csv":
                        csv_index += 1
                        shutil.copy2(path, run_dir / f"extracted-{csv_index:03d}.csv")
                    elif path.suffix.lower() == ".gpx" and gpx_index == 0:
                        gpx_index += 1
                        shutil.copy2(path, run_dir / "route.gpx")
            self._write_manifest(run_dir, {
                "label": label,
                "date": date,
                "source": source_kind,
                "source_files": [source.original_name for source in sources],
                "imported_at": utc_now(),
                "import_kind": "raw",
                "import_warnings": [{
                    "code": "no-gps" if re.search(r"\b(?:gpx|gps)\b", warning, re.IGNORECASE) else "raw-import",
                    "message": warning,
                }],
                "original_uploads": originals,
            })
            run_dir = self._commit(run_dir)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return self._summary(run_dir)

    def _import_firmware(
        self,
        source: Path,
        raw_sources: list[ImportSource],
        label: str,
        date: str,
        source_kind: str,
    ) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="utsm-dashboard-import-") as temp:
            output = Path(temp) / "runs"
            import_source = source
            if source.suffix.lower() == ".zip":
                extracted_tree = Path(temp) / "safe-tree"
                extracted_paths = safe_extract_zip(source, extracted_tree)
                csv_files = [path for path in extracted_paths if path.suffix.lower() == ".csv"]
                if not csv_files:
                    raise ValueError("The ZIP does not contain a CSV file.")
                extracted = Path(temp) / "safe-csv"
                extracted.mkdir()
                for index, path in enumerate(csv_files, start=1):
                    destination = extracted / f"part-{index:04d}.csv"
                    shutil.copy2(path, destination)
                    self.validate_csv(destination)
                import_source = extracted
            args = Namespace(
                input=str(import_source),
                output_dir=str(output),
                name=slugify(label),
                date=date,
                session_gap_sec=60.0,
                min_session_distance_m=250.0,
            )
            generated = import_runs(args)
            originals = self._store_sources(raw_sources, source_kind)
            stages: list[Path] = []
            destinations: list[Path] = []
            batch_id = uuid.uuid4().hex if len(generated) > 1 else None
            batch_marker = self.runs_dir / f"_batch-{batch_id}.pending" if batch_id else None
            if batch_marker is not None:
                batch_marker.write_text("pending\n", encoding="ascii")
            try:
                for index, generated_dir in enumerate(generated, start=1):
                    suffix = f" Run {index}" if len(generated) > 1 else ""
                    stage = self._stage()
                    shutil.rmtree(stage)
                    shutil.copytree(generated_dir, stage)
                    manifest = _read_manifest(stage)
                    manifest.update({
                        "label": f"{label}{suffix}",
                        "date": date,
                        "source": source_kind,
                        "source_files": [source.original_name for source in raw_sources],
                        "imported_at": utc_now(),
                        "import_kind": "firmware",
                        "original_uploads": originals,
                    })
                    if batch_id:
                        manifest["batch_id"] = batch_id
                    self._write_manifest(stage, manifest)
                    stages.append(stage)
                for stage in stages:
                    destinations.append(self._commit(stage))
                if batch_marker is not None:
                    batch_marker.unlink(missing_ok=True)
                    self.invalidate()
                return [self._summary(destination) for destination in destinations]
            except Exception:
                for stage in stages:
                    shutil.rmtree(stage, ignore_errors=True)
                for destination in destinations:
                    shutil.rmtree(destination, ignore_errors=True)
                if batch_marker is not None:
                    batch_marker.unlink(missing_ok=True)
                self.invalidate()
                raise

    def _import_paths_unlocked(
        self,
        paths: list[Path | ImportSource],
        *,
        label: str,
        date: str,
        source_kind: str,
    ) -> ImportOutcome:
        sources = self._sources(paths)
        if not sources:
            raise ValueError("Choose at least one CSV, GPX, or ZIP file.")
        clean_label = " ".join(label.split()).strip()[:80]
        if not clean_label:
            raise ValueError("Run name is required.")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError("Date must use YYYY-MM-DD.") from error
        unsupported = [source.original_name for source in sources if source.path.suffix.lower() not in ALLOWED_IMPORT_EXTENSIONS]
        if unsupported:
            raise ValueError(f"Unsupported file type: {', '.join(unsupported)}")
        zip_sources = [source for source in sources if source.path.suffix.lower() == ".zip"]
        gpx_sources = [source for source in sources if source.path.suffix.lower() == ".gpx"]
        csv_sources = [source for source in sources if source.path.suffix.lower() == ".csv"]
        zip_paths = [source.path for source in zip_sources]
        gpx_paths = [source.path for source in gpx_sources]
        csv_paths = [source.path for source in csv_sources]
        for path in csv_paths:
            self.validate_csv(path)
        for path in gpx_paths:
            self.validate_gpx(path)
        if zip_paths and len(sources) != 1:
            raise ValueError("Upload one ZIP at a time, or upload the CSV and GPX files directly.")
        warnings: list[str] = []
        if zip_paths:
            source = zip_paths[0]
            try:
                runs = self._import_firmware(source, zip_sources, clean_label, date, source_kind)
            except ArchiveValidationError:
                raise
            except ValueError:
                with tempfile.TemporaryDirectory(prefix="utsm-dashboard-zip-") as temp:
                    extracted = Path(temp)
                    extracted_files = safe_extract_zip(source, extracted)
                    if not extracted_files:
                        raise ValueError("The ZIP does not contain CSV or GPX files.")
                    for path in extracted_files:
                        self.validate_csv(path) if path.suffix.lower() == ".csv" else self.validate_gpx(path)
                    warning = "Saved for raw review because automatic lap import found no complete GPS laps."
                    warnings.append(warning)
                    runs = [self._import_raw(zip_sources, clean_label, date, source_kind, warning, extracted_files)]
            return ImportOutcome(runs, warnings)
        if len(gpx_paths) > 1:
            raise ValueError("Choose one GPX route for each import.")
        if gpx_paths:
            if not csv_paths:
                raise ValueError("A GPX import also needs at least one CSV telemetry file.")
            return ImportOutcome([
                self._import_gpx_csv(gpx_paths[0], csv_paths, sources, clean_label, date, source_kind)
            ], warnings)
        if csv_paths and all(self._looks_like_firmware_csv(path) for path in csv_paths):
            source: Path
            with tempfile.TemporaryDirectory(prefix="utsm-dashboard-csv-") as temp:
                if len(csv_paths) == 1:
                    source = csv_paths[0]
                    try:
                        runs = self._import_firmware(source, csv_sources, clean_label, date, source_kind)
                    except ValueError:
                        warning = "Saved for raw review because no complete GPS laps were found."
                        warnings.append(warning)
                        runs = [self._import_raw(csv_sources, clean_label, date, source_kind, warning)]
                else:
                    folder = Path(temp) / "csv"
                    folder.mkdir()
                    for path in csv_paths:
                        shutil.copy2(path, folder / path.name)
                    try:
                        runs = self._import_firmware(folder, csv_sources, clean_label, date, source_kind)
                    except ValueError:
                        warning = "Saved for raw review because no complete GPS laps were found."
                        warnings.append(warning)
                        runs = [self._import_raw(csv_sources, clean_label, date, source_kind, warning)]
            return ImportOutcome(runs, warnings)
        if csv_paths:
            warning = "No GPX route was included. Data is available for table review."
            warnings.append(warning)
            return ImportOutcome([
                self._import_raw(csv_sources, clean_label, date, source_kind, warning)
            ], warnings)
        raise ValueError("Choose at least one CSV file, or a ZIP containing firmware CSV files.")

    def import_paths(
        self,
        paths: list[Path | ImportSource],
        *,
        label: str,
        date: str,
        source_kind: str,
    ) -> ImportOutcome:
        with self._cross_process_import_lock():
            return self._import_paths_unlocked(
                paths,
                label=label,
                date=date,
                source_kind=source_kind,
            )
