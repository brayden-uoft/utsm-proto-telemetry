from __future__ import annotations

import hashlib
import asyncio
import json
import os
import stat
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import live_dashboard.app as app_module
import live_dashboard.run_library as library_module
import utsm_telemetry.safe_archive as archive_module
from live_dashboard.run_library import ImportSource, RunLibrary
from utsm_telemetry.safe_archive import safe_extract_zip


CSV_BYTES = b"timestamp_ms,current_mA,voltage_mV\n2,2000,24000\n1,1000,24000\n3,3000,24000\n"
GPX_BYTES = b"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
<trkpt lat="45.500000" lon="-73.600000"><time>2026-08-06T12:00:00Z</time></trkpt>
<trkpt lat="45.500100" lon="-73.599900"><time>2026-08-06T12:00:01Z</time></trkpt>
</trkseg></trk></gpx>"""


class TestRunLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "runs"
        self.library = RunLibrary(self.root)
        self.sources = Path(self.temp.name) / "sources"
        self.sources.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def source(self, name: str, content: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(content)
        return path

    def test_raw_import_is_persistent_and_byte_exact(self) -> None:
        source = self.source("original.csv", CSV_BYTES)
        outcome = self.library.import_paths(
            [source], label="Campus Raw", date="2026-08-06", source_kind="SD card"
        )
        self.assertEqual(len(outcome.runs), 1)
        run_id = outcome.runs[0]["id"]
        self.assertRegex(run_id, r"^run-[0-9a-f]{12}$")
        detail = self.library.get_run(run_id)
        assert detail is not None
        original = detail["manifest"]["original_uploads"][0]
        self.assertEqual(original["original_name"], "original.csv")
        self.assertEqual(original["size_bytes"], len(CSV_BYTES))
        self.assertEqual(original["sha256"], hashlib.sha256(CSV_BYTES).hexdigest())
        virtual_path = f"originals/{original['upload_id']}"
        self.assertEqual(self.library.file_path(run_id, virtual_path).read_bytes(), CSV_BYTES)
        self.assertEqual(self.library.file_path(run_id, virtual_path).name, original["sha256"])
        restarted = RunLibrary(self.root)
        self.assertEqual(restarted.get_run(run_id)["label"], "Campus Raw")

    def test_gpx_csv_import_is_analysis_ready_and_preserves_sources(self) -> None:
        csv_path = self.source("car.csv", CSV_BYTES)
        gpx_path = self.source("route.gpx", GPX_BYTES)
        run = self.library.import_paths(
            [gpx_path, csv_path], label="Pair", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        self.assertTrue(run["analysis_ready"])
        self.assertTrue(run["capabilities"]["view_map"])
        detail = self.library.get_run(run["id"])
        self.assertEqual(len(detail["manifest"]["original_uploads"]), 2)

    def test_failed_pair_import_leaves_no_run_or_staging_folder(self) -> None:
        gpx = self.source("route.gpx", GPX_BYTES)
        first = self.source("first.csv", b"a,b\n1,2\n")
        second = self.source("second.csv", b"a,c\n1,3\n")
        with self.assertRaisesRegex(ValueError, "matching columns"):
            self.library.import_paths(
                [gpx, first, second], label="Bad Pair", date="2026-08-06", source_kind="Browser upload"
            )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_malformed_csv_and_gpx_are_rejected(self) -> None:
        malformed_csv = self.source("bad.csv", b"a,b\n1\n")
        with self.assertRaisesRegex(ValueError, "wrong number of columns"):
            self.library.import_paths(
                [malformed_csv], label="Bad", date="2026-08-06", source_kind="Browser upload"
            )
        malformed_gpx = self.source("bad.gpx", b"<gpx><trkpt")
        good_csv = self.source("good.csv", b"a,b\n1,2\n")
        with self.assertRaisesRegex(ValueError, "could not be parsed"):
            self.library.import_paths(
                [malformed_gpx, good_csv], label="Bad", date="2026-08-06", source_kind="Browser upload"
            )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_csv_viewer_paginates_filters_and_sorts(self) -> None:
        source = self.source("values.csv", CSV_BYTES)
        run = self.library.import_paths(
            [source], label="Values", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        detail = self.library.get_run(run["id"])
        relative = detail["primary_csv"]
        page = self.library.csv_page(run["id"], relative, offset=1, limit=1, sort="timestamp_ms", direction="desc")
        self.assertEqual(page["total"], 3)
        self.assertEqual(page["rows"][0]["timestamp_ms"], "2")
        filtered = self.library.csv_page(run["id"], relative, query="24000", limit=2)
        self.assertEqual(filtered["total"], 3)
        with self.assertRaisesRegex(ValueError, "sort column"):
            self.library.csv_page(run["id"], relative, sort="missing")

    def test_raw_only_capabilities_gate_unavailable_actions(self) -> None:
        source = self.source("raw.csv", CSV_BYTES)
        run = self.library.import_paths(
            [source], label="Raw", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        self.assertTrue(run["capabilities"]["view_data"])
        self.assertTrue(run["capabilities"]["download_original"])
        self.assertFalse(run["capabilities"]["view_map"])
        self.assertFalse(run["capabilities"]["replay"])
        self.assertTrue(run["actions"]["data"]["href"].endswith("#dataSection"))
        self.assertTrue(run["actions"]["charts"]["href"].endswith("#chartSection"))
        self.assertEqual(run["actions"]["compare"]["href"], f"/compare?run={run['id']}")
        self.assertFalse(run["actions"]["replay"]["available"])
        self.assertIn("timed GPX", run["actions"]["replay"]["reason"])
        self.assertFalse(run["actions"]["strategy"]["available"])

    def test_run_id_and_file_traversal_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.library.get_run("../tracks")
        source = self.source("raw.csv", CSV_BYTES)
        run = self.library.import_paths(
            [source], label="Raw", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        with self.assertRaises(ValueError):
            self.library.file_path(run["id"], "../../secret")

    def make_zip(self, name: str, entries: list[tuple[zipfile.ZipInfo | str, bytes]]) -> Path:
        path = self.sources / name
        with zipfile.ZipFile(path, "w") as archive:
            for info, content in entries:
                archive.writestr(info, content)
        return path

    def test_zip_traversal_symlink_unsupported_and_expansion_are_rejected(self) -> None:
        traversal = self.make_zip("traversal.zip", [("../escape.csv", CSV_BYTES)])
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.library.validate_zip(traversal)
        link = zipfile.ZipInfo("link.csv")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        symlink = self.make_zip("symlink.zip", [(link, b"target.csv")])
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.library.validate_zip(symlink)
        unsupported = self.make_zip("unsupported.zip", [("readme.exe", b"bad")])
        with self.assertRaisesRegex(ValueError, "unsupported file"):
            self.library.validate_zip(unsupported)
        expanded = self.make_zip("expanded.zip", [("one.csv", CSV_BYTES)])
        with patch.object(archive_module, "MAX_ARCHIVE_EXPANDED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "500 MB safety limit"):
                self.library.validate_zip(expanded)

    def test_zip_rejects_collisions_devices_ads_modes_and_compression_bombs(self) -> None:
        cases: list[tuple[str, list[tuple[zipfile.ZipInfo | str, bytes]], str]] = [
            ("collision.zip", [("A.csv", CSV_BYTES), ("a.CSV", CSV_BYTES)], "colliding filenames"),
            ("device.zip", [("CON.csv", CSV_BYTES)], "reserved device"),
            ("ads.zip", [("data:secret.csv", CSV_BYTES)], "ADS path"),
            ("trailing.zip", [("data.csv.", CSV_BYTES)], "trailing dot"),
        ]
        fifo = zipfile.ZipInfo("pipe.csv")
        fifo.create_system = 3
        fifo.external_attr = (stat.S_IFIFO | 0o644) << 16
        cases.append(("fifo.zip", [(fifo, CSV_BYTES)], "non-regular"))
        for name, entries, expected in cases:
            with self.subTest(name=name):
                archive = self.make_zip(name, entries)
                with self.assertRaisesRegex(ValueError, expected):
                    self.library.validate_zip(archive)
        compressed = self.sources / "compression.zip"
        with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("zeros.csv", b"a\n" + b"0\n" * 600_000)
        with self.assertRaisesRegex(ValueError, "compression ratio"):
            self.library.validate_zip(compressed)

    def test_zip_rejects_encrypted_flag_and_actual_extraction_overflow(self) -> None:
        encrypted = self.make_zip("encrypted.zip", [("data.csv", CSV_BYTES)])
        payload = bytearray(encrypted.read_bytes())
        local = payload.index(b"PK\x03\x04")
        central = payload.index(b"PK\x01\x02")
        payload[local + 6] |= 1
        payload[central + 8] |= 1
        encrypted.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "encrypted"):
            self.library.validate_zip(encrypted)
        archive = self.make_zip("actual.zip", [("data.csv", CSV_BYTES)])
        with patch.object(archive_module, "MAX_ARCHIVE_EXPANDED_BYTES", len(CSV_BYTES) - 1):
            with self.assertRaisesRegex(ValueError, "500 MB safety limit|extraction limit"):
                safe_extract_zip(archive, self.sources / "actual-output")

    def test_zip_counts_directories_toward_the_entry_limit(self) -> None:
        archive = self.make_zip(
            "many-entries.zip",
            [("first/", b""), ("first/data.csv", CSV_BYTES)],
        )
        with patch.object(archive_module, "MAX_ARCHIVE_FILES", 1):
            with self.assertRaisesRegex(ValueError, "more than 1 entries"):
                self.library.validate_zip(archive)

    def test_gpx_rejects_dtd_entities_and_depth(self) -> None:
        dtd = self.source(
            "entity.gpx",
            b'<!DOCTYPE gpx [<!ENTITY x "bad">]><gpx><trk><trkseg><trkpt lat="45" lon="-73">&x;</trkpt></trkseg></trk></gpx>',
        )
        csv_path = self.source("entity.csv", CSV_BYTES)
        with self.assertRaises(ValueError):
            self.library.import_paths([dtd, csv_path], label="DTD", date="2026-08-06", source_kind="Browser upload")
        deep = self.source(
            "deep.gpx",
            ("<gpx>" + "<x>" * 70 + '<trkpt lat="45" lon="-73"/>' + "</x>" * 70 + "</gpx>").encode(),
        )
        with self.assertRaisesRegex(ValueError, "depth limit"):
            self.library.import_paths([deep, csv_path], label="Deep", date="2026-08-06", source_kind="Browser upload")

    def test_csv_and_gpx_byte_limits_apply_outside_the_web_endpoint(self) -> None:
        csv_path = self.source("limit.csv", CSV_BYTES)
        gpx_path = self.source("limit.gpx", GPX_BYTES)
        with patch.object(library_module, "MAX_CSV_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "CSV is larger"):
                self.library.validate_csv(csv_path)
        with patch.object(library_module, "MAX_GPX_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "GPX is larger"):
                self.library.validate_gpx(gpx_path)

    def test_packaged_and_uploaded_roots_are_merged_without_writing_packaged(self) -> None:
        packaged = Path(self.temp.name) / "packaged"
        uploaded = Path(self.temp.name) / "dashboard" / "runs"
        blobs = Path(self.temp.name) / "dashboard" / "blobs"
        packaged_run = packaged / "included-run"
        packaged_run.mkdir(parents=True)
        (packaged_run / "data.csv").write_bytes(CSV_BYTES)
        before = sorted(path.relative_to(packaged).as_posix() for path in packaged.rglob("*"))
        library = RunLibrary(packaged, uploaded, blobs)
        source = self.source("new.csv", CSV_BYTES)
        imported = library.import_paths([source], label="Uploaded", date="2026-08-06", source_kind="Browser upload").runs[0]
        after = sorted(path.relative_to(packaged).as_posix() for path in packaged.rglob("*"))
        self.assertEqual(before, after)
        catalog = library.list_runs()
        self.assertEqual({item["source"] for item in catalog}, {"Packaged", "Uploaded"})
        self.assertTrue((uploaded / imported["id"]).is_dir())

    def test_malformed_manifest_is_isolated_from_catalog(self) -> None:
        broken = self.root / "broken-run"
        valid = self.root / "valid-run"
        broken.mkdir()
        valid.mkdir()
        (broken / "run.json").write_text("{not json", encoding="utf-8")
        (valid / "data.csv").write_bytes(CSV_BYTES)
        runs = self.library.list_runs()
        self.assertEqual(len(runs), 2)
        self.assertTrue(any("run.json could not be read" in " ".join(run["warnings"]) for run in runs))

    def test_missing_packaged_root_is_catalog_safe(self) -> None:
        library = RunLibrary(
            Path(self.temp.name) / "does-not-exist",
            Path(self.temp.name) / "writable" / "runs",
            Path(self.temp.name) / "writable" / "blobs",
        )
        self.assertEqual(library.list_runs(), [])

    def test_text_only_csv_has_an_exact_charts_unavailable_reason(self) -> None:
        source = self.source("notes.csv", b"name,state\nalice,ready\nbob,waiting\n")
        run = self.library.import_paths(
            [source], label="Notes", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        self.assertTrue(run["actions"]["data"]["available"])
        self.assertFalse(run["actions"]["charts"]["available"])
        self.assertEqual(run["actions"]["charts"]["reason"], "Charts need at least one numeric CSV value.")

    def test_raw_no_gps_warning_is_structurally_deduplicated(self) -> None:
        source = self.source("raw.csv", CSV_BYTES)
        run = self.library.import_paths(
            [source], label="Raw", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        self.assertEqual(run["warning_codes"].count("no-gps"), 1)
        self.assertEqual(len([warning for warning in run["warnings"] if "GPX" in warning or "GPS" in warning]), 1)

    def test_pending_multi_run_batch_is_hidden_until_complete(self) -> None:
        source = self.source("firmware.csv", CSV_BYTES)
        generated_root = Path(self.temp.name) / "batch-generated"
        generated: list[Path] = []
        for index in range(2):
            run_dir = generated_root / f"run-{index}"
            run_dir.mkdir(parents=True)
            (run_dir / "telemetry.csv").write_bytes(CSV_BYTES)
            (run_dir / "route.gpx").write_bytes(GPX_BYTES)
            (run_dir / "run.json").write_text('{"laps": 1}', encoding="utf-8")
            generated.append(run_dir)
        original_commit = self.library._commit
        visible_during_commit: list[list[str]] = []

        def observed_commit(stage: Path) -> Path:
            destination = original_commit(stage)
            visible_during_commit.append([run["id"] for run in self.library.list_runs()])
            return destination

        with patch.object(library_module, "import_runs", return_value=generated), patch.object(
            self.library, "_commit", side_effect=observed_commit
        ):
            runs = self.library._import_firmware(
                source, [ImportSource(source, "firmware.csv")], "Batch", "2026-08-06", "SD card"
            )
        self.assertEqual(visible_during_commit, [[], []])
        self.assertEqual({run["id"] for run in self.library.list_runs()}, {run["id"] for run in runs})

    def test_shared_catalog_cache_refreshes_when_batch_marker_is_removed(self) -> None:
        source = self.source("shared-firmware.csv", CSV_BYTES)
        generated_root = Path(self.temp.name) / "shared-generated"
        generated: list[Path] = []
        for index in range(2):
            run_dir = generated_root / f"run-{index}"
            run_dir.mkdir(parents=True)
            (run_dir / "telemetry.csv").write_bytes(CSV_BYTES)
            (run_dir / "route.gpx").write_bytes(GPX_BYTES)
            (run_dir / "run.json").write_text('{"laps": 1}', encoding="utf-8")
            generated.append(run_dir)
        observer = RunLibrary(self.root)
        original_commit = self.library._commit
        observer_during_commit: list[list[str]] = []

        def observed_commit(stage: Path) -> Path:
            destination = original_commit(stage)
            observer_during_commit.append([run["id"] for run in observer.list_runs()])
            return destination

        with patch.object(library_module, "import_runs", return_value=generated), patch.object(
            self.library, "_commit", side_effect=observed_commit
        ):
            completed = self.library._import_firmware(
                source,
                [ImportSource(source, "shared-firmware.csv")],
                "Shared batch",
                "2026-08-06",
                "SD card",
            )
        self.assertEqual(observer_during_commit, [[], []])
        self.assertEqual(
            {run["id"] for run in observer.list_runs()},
            {run["id"] for run in completed},
        )

    def test_multi_session_firmware_references_one_blob(self) -> None:
        source = self.source("firmware.csv", CSV_BYTES)
        generated_root = Path(self.temp.name) / "generated"
        generated: list[Path] = []
        for index in range(2):
            run_dir = generated_root / f"generated-{index}"
            run_dir.mkdir(parents=True)
            (run_dir / "telemetry.csv").write_bytes(CSV_BYTES)
            (run_dir / "route.gpx").write_bytes(GPX_BYTES)
            (run_dir / "run.json").write_text('{"laps": 1}', encoding="utf-8")
            generated.append(run_dir)
        with patch.object(library_module, "import_runs", return_value=generated):
            runs = self.library._import_firmware(
                source,
                [ImportSource(source, "firmware.csv")],
                "Firmware",
                "2026-08-06",
                "SD card",
            )
        self.assertEqual(len(runs), 2)
        blobs = [path for path in self.library.blob_dir.iterdir() if not path.name.startswith("_")]
        self.assertEqual(len(blobs), 1)
        digests = {
            self.library.get_run(run["id"])["manifest"]["original_uploads"][0]["sha256"]
            for run in runs
        }
        self.assertEqual(len(digests), 1)
        self.assertTrue(all(not (self.library.run_dir(run["id"]) / "raw").exists() for run in runs))

    def test_front_campus_replay_keeps_final_timestamp_and_finishes_route_once(self) -> None:
        library = RunLibrary(
            app_module.PACKAGED_RUNS_DIR,
            Path(self.temp.name) / "front-campus-user" / "runs",
            Path(self.temp.name) / "front-campus-user" / "blobs",
            app_module.STRATEGY_DIR,
        )
        payload = library.replay_payload("front-campus-2026-08-06-run-01")
        samples = payload["samples"]
        self.assertEqual(samples[-1]["timestamp_ms"], 1_309_596.0)
        self.assertEqual(samples[-1]["route_progress_percent"], 100.0)
        self.assertTrue(all(sample["route_progress_percent"] < 100 for sample in samples[:-1]))
        self.assertEqual(samples[0]["timestamp_ms"], 0.0)

    def test_front_campus_comparison_derives_power_from_current_and_voltage(self) -> None:
        library = RunLibrary(
            app_module.PACKAGED_RUNS_DIR,
            Path(self.temp.name) / "front-campus-compare" / "runs",
            Path(self.temp.name) / "front-campus-compare" / "blobs",
            app_module.STRATEGY_DIR,
        )
        metrics = library.comparison_metrics("front-campus-2026-08-06-run-01")
        self.assertAlmostEqual(metrics["average_power_W"], 84.7925806, places=5)
        self.assertGreater(metrics["peak_power_W"], metrics["average_power_W"])

    def test_packaged_afternoon_run_links_run_derived_strategy(self) -> None:
        library = RunLibrary(
            app_module.PACKAGED_RUNS_DIR,
            Path(self.temp.name) / "strategy-user" / "runs",
            Path(self.temp.name) / "strategy-user" / "blobs",
            app_module.STRATEGY_DIR,
        )
        run = library.get_run("afternoon-run")
        self.assertTrue(run["actions"]["strategy"]["available"])
        self.assertEqual(run["actions"]["strategy"]["href"], "/strategy/indy")
        self.assertIn("packaged Afternoon Run", run["strategy"]["provenance"])


class TestDashboardAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_library = app_module.run_library
        self.original_key = app_module.OPERATOR_KEY
        self.original_limit = app_module.MAX_UPLOAD_BYTES
        self.original_file_limit = app_module.MAX_UPLOAD_FILES
        app_module.run_library = RunLibrary(Path(self.temp.name) / "runs")
        app_module.OPERATOR_KEY = "operator-test-key"
        with app_module.unlock_attempts_lock:
            app_module.unlock_attempts.clear()
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        app_module.run_library = self.original_library
        app_module.OPERATOR_KEY = self.original_key
        app_module.MAX_UPLOAD_BYTES = self.original_limit
        app_module.MAX_UPLOAD_FILES = self.original_file_limit
        with app_module.unlock_attempts_lock:
            app_module.unlock_attempts.clear()
        self.temp.cleanup()

    @property
    def headers(self) -> dict[str, str]:
        return {"X-UTSM-Request": "dashboard-import"}

    @property
    def form(self) -> dict[str, str]:
        return {"name": "API run", "date": "2026-08-06", "source": "upload"}

    def unlock(self, client: TestClient | None = None) -> None:
        response = (client or self.client).post(
            "/api/operator/unlock",
            json={"key": "operator-test-key"},
            headers={"X-UTSM-Request": "dashboard-unlock"},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_every_page_and_shared_asset_loads(self) -> None:
        for path in ("/", "/live", "/runs", "/import", "/dyno", "/compare", "/strategy", "/strategy/indy", "/replay/example", "/favicon.ico", "/static/app.css", "/static/shell.js"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_import_requires_operator_key_marker_and_same_origin(self) -> None:
        files = [("files", ("run.csv", CSV_BYTES, "text/csv"))]
        self.assertEqual(
            self.client.post("/api/operator/unlock", json={"key": "operator-test-key"}).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                "/api/operator/unlock",
                json={"key": "operator-test-key"},
                headers={"X-UTSM-Request": "dashboard-unlock", "Origin": "https://attacker.example"},
            ).status_code,
            403,
        )
        self.assertEqual(self.client.post("/api/runs/import", data=self.form, files=files).status_code, 403)
        self.assertEqual(self.client.post("/api/runs/import", data=self.form, files=files, headers=self.headers).status_code, 401)
        self.unlock()
        cross_origin = {**self.headers, "Origin": "https://attacker.example"}
        self.assertEqual(self.client.post("/api/runs/import", data=self.form, files=files, headers=cross_origin).status_code, 403)

    def test_unlock_cookie_is_http_only_same_site_and_bounded(self) -> None:
        response = self.client.post(
            "/api/operator/unlock",
            json={"key": "operator-test-key"},
            headers={"X-UTSM-Request": "dashboard-unlock"},
        )
        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        self.assertIn("max-age=28800", cookie)

    def test_operator_access_fails_closed_when_key_is_missing_or_default(self) -> None:
        files = [("files", ("run.csv", CSV_BYTES, "text/csv"))]
        for value in (None, "", "short", "change-me", "change-me-operator", "replace-this-for-run-imports"):
            with self.subTest(value=value):
                app_module.OPERATOR_KEY = value
                status = self.client.get("/api/operator/status")
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json(), {"configured": False, "unlocked": False})
                unlock = self.client.post(
                    "/api/operator/unlock",
                    json={"key": "anything"},
                    headers={"X-UTSM-Request": "dashboard-unlock"},
                )
                self.assertEqual(unlock.status_code, 503)
                response = self.client.post(
                    "/api/runs/import", data=self.form, files=files, headers=self.headers
                )
                self.assertEqual(response.status_code, 503)
                self.assertNotIn(str(Path(self.temp.name)), response.text)
        import_page = self.client.get("/import").text
        self.assertIn("UTSM_DASHBOARD_OPERATOR_KEY", import_page)

    def test_import_list_detail_rows_and_original_download(self) -> None:
        self.unlock()
        response = self.client.post(
            "/api/runs/import", data=self.form,
            files=[("files", ("run.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
        )
        self.assertEqual(response.status_code, 201, response.text)
        run = response.json()["runs"][0]
        run_id = run["id"]
        self.assertEqual(self.client.get("/api/runs").json()["count"], 1)
        detail = self.client.get(f"/api/runs/{run_id}").json()["run"]
        raw = next(file for file in detail["files"] if file["is_raw"] and file["name"] == "run.csv")
        rows = self.client.get(f"/api/runs/{run_id}/csv", params={"file": raw["path"], "limit": 1})
        self.assertEqual(rows.status_code, 200)
        self.assertEqual(len(rows.json()["rows"]), 1)
        downloaded = self.client.get(f"/api/runs/{run_id}/files/{raw['path']}")
        self.assertEqual(downloaded.content, CSV_BYTES)
        manifest = detail["manifest"]["original_uploads"][0]
        self.assertEqual(manifest["sha256"], hashlib.sha256(CSV_BYTES).hexdigest())
        self.assertNotIn(str(Path(self.temp.name)), json.dumps(detail))

    def test_uploaded_run_data_requires_operator_session(self) -> None:
        self.unlock()
        response = self.client.post(
            "/api/runs/import", data=self.form,
            files=[("files", ("run.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
        )
        run_id = response.json()["runs"][0]["id"]
        detail = self.client.get(f"/api/runs/{run_id}").json()["run"]
        raw = next(file for file in detail["files"] if file["is_raw"])
        locked = TestClient(app_module.app)
        self.assertEqual(locked.get(f"/api/runs/{run_id}").status_code, 401)
        self.assertEqual(
            locked.get(f"/api/runs/{run_id}/csv", params={"file": raw["path"]}).status_code,
            401,
        )
        self.assertEqual(
            locked.get(f"/api/runs/{run_id}/files/{raw['path']}").status_code,
            401,
        )

    def test_locked_catalog_redacts_every_uploaded_metadata_field(self) -> None:
        packaged = Path(self.temp.name) / "packaged"
        packaged_run = packaged / "public-run"
        packaged_run.mkdir(parents=True)
        (packaged_run / "data.csv").write_bytes(CSV_BYTES)
        app_module.run_library = RunLibrary(
            packaged,
            Path(self.temp.name) / "protected" / "runs",
            Path(self.temp.name) / "protected" / "blobs",
        )
        self.unlock()
        response = self.client.post(
            "/api/runs/import", data={**self.form, "name": "Secret run label"},
            files=[("files", ("private.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
        )
        self.assertEqual(response.status_code, 201, response.text)
        run_id = response.json()["runs"][0]["id"]
        locked = TestClient(app_module.app)
        payload = locked.get("/api/runs").json()
        self.assertEqual([run["id"] for run in payload["runs"]], ["public-run"])
        self.assertEqual(payload["locked_uploaded_count"], 1)
        serialized = json.dumps(payload)
        for secret in (run_id, "Secret run label", "private.csv", "2026-08-06", "Browser upload"):
            self.assertNotIn(secret, serialized)
        status = locked.get("/api/status").json()
        self.assertEqual(status["run_count"], 1)
        self.assertEqual(status["locked_uploaded_count"], 1)
        self.unlock(locked)
        unlocked = locked.get("/api/runs").json()
        self.assertEqual(unlocked["count"], 2)
        self.assertIn(run_id, {run["id"] for run in unlocked["runs"]})

    def test_hostile_display_filename_roundtrips_without_becoming_a_disk_path(self) -> None:
        self.unlock()
        original_name = "SD Card/Folder/<script> & telemetry.csv"
        response = self.client.post(
            "/api/runs/import",
            data=self.form,
            files=[("files", (original_name, CSV_BYTES, "text/csv"))],
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 201, response.text)
        run_id = response.json()["runs"][0]["id"]
        detail = self.client.get(f"/api/runs/{run_id}").json()["run"]
        manifest = detail["manifest"]["original_uploads"][0]
        self.assertEqual(manifest["original_name"], original_name)
        raw = next(file for file in detail["files"] if file["is_raw"])
        physical = app_module.run_library.file_path(run_id, raw["path"])
        self.assertRegex(physical.name, r"^[0-9a-f]{64}$")
        self.assertNotIn("script", physical.name)
        download = self.client.get(f"/api/runs/{run_id}/files/{raw['path']}")
        self.assertEqual(download.content, CSV_BYTES)
        disposition = download.headers["content-disposition"]
        self.assertIn("telemetry.csv", disposition)
        self.assertNotIn("SD Card/Folder", disposition)

    def test_windows_reserved_download_name_is_sanitized_without_changing_manifest(self) -> None:
        self.unlock()
        response = self.client.post(
            "/api/runs/import", data=self.form,
            files=[("files", ("CON.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
        )
        self.assertEqual(response.status_code, 201, response.text)
        run_id = response.json()["runs"][0]["id"]
        detail = self.client.get(f"/api/runs/{run_id}").json()["run"]
        self.assertEqual(detail["manifest"]["original_uploads"][0]["original_name"], "CON.csv")
        raw = next(file for file in detail["files"] if file["is_raw"])
        download = self.client.get(f"/api/runs/{run_id}/files/{raw['path']}")
        self.assertEqual(download.content, CSV_BYTES)
        self.assertIn("_CON.csv", download.headers["content-disposition"])

    def test_duplicate_case_names_and_oversized_upload_are_rejected_atomically(self) -> None:
        self.unlock()
        files = [
            ("files", ("RUN.csv", CSV_BYTES, "text/csv")),
            ("files", ("run.CSV", CSV_BYTES, "text/csv")),
        ]
        response = self.client.post("/api/runs/import", data=self.form, files=files, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(app_module.run_library.list_runs(), [])
        app_module.MAX_UPLOAD_BYTES = 4
        response = self.client.post(
            "/api/runs/import", data=self.form,
            files=[("files", ("large.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(app_module.run_library.list_runs(), [])

    def test_upload_file_count_is_bounded(self) -> None:
        self.unlock()
        app_module.MAX_UPLOAD_FILES = 1
        response = self.client.post(
            "/api/runs/import",
            data=self.form,
            files=[
                ("files", ("one.csv", CSV_BYTES, "text/csv")),
                ("files", ("two.csv", CSV_BYTES, "text/csv")),
            ],
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(app_module.run_library.list_runs(), [])

    def test_streamed_request_body_is_bounded_without_content_length(self) -> None:
        called = False

        async def inner(scope, receive, send) -> None:
            nonlocal called
            called = True
            await receive()

        messages = [{"type": "http.request", "body": b"12345", "more_body": False}]
        sent: list[dict[str, object]] = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        token = app_module._session_token()
        scope = {
            "type": "http", "method": "POST", "path": "/api/runs/import",
            "headers": [
                (b"host", b"testserver"),
                (b"x-utsm-request", b"dashboard-import"),
                (b"cookie", f"{app_module.SESSION_COOKIE}={token}".encode()),
            ], "scheme": "http", "query_string": b"", "root_path": "",
            "http_version": "1.1", "server": ("testserver", 80), "client": ("tester", 1),
        }
        with patch.object(app_module, "MAX_UPLOAD_BYTES", 4), patch.object(app_module, "MAX_MULTIPART_OVERHEAD_BYTES", 0):
            asyncio.run(app_module.ImportBodyLimitMiddleware(inner)(scope, receive, send))
        self.assertTrue(called)
        self.assertEqual(next(message["status"] for message in sent if message["type"] == "http.response.start"), 413)

    def test_rejected_mutations_do_not_consume_the_asgi_body(self) -> None:
        async def exercise(headers: list[tuple[bytes, bytes]], configured_key: str | None) -> tuple[int, int, bool]:
            receive_calls = 0
            entered = False
            sent: list[dict[str, object]] = []

            async def receive():
                nonlocal receive_calls
                receive_calls += 1
                raise AssertionError("Rejected requests must not read the body")

            async def send(message):
                sent.append(message)

            async def inner(scope, receive, send):
                nonlocal entered
                entered = True

            scope = {
                "type": "http", "method": "POST", "path": "/api/runs/import",
                "headers": headers, "scheme": "http", "query_string": b"", "root_path": "",
                "http_version": "1.1", "server": ("testserver", 80), "client": ("tester", 1),
            }
            with patch.object(app_module, "OPERATOR_KEY", configured_key):
                await app_module.ImportBodyLimitMiddleware(inner)(scope, receive, send)
            status = next(message["status"] for message in sent if message["type"] == "http.response.start")
            return status, receive_calls, entered

        valid_base = [(b"host", b"testserver"), (b"x-utsm-request", b"dashboard-import")]
        cases = [
            ([], "operator-test-key", 403),
            (valid_base, None, 503),
            (valid_base + [(b"origin", b"https://attacker.example")], "operator-test-key", 403),
            (valid_base, "operator-test-key", 401),
        ]
        for headers, key, expected in cases:
            with self.subTest(expected=expected, headers=headers):
                status, receive_calls, entered = asyncio.run(exercise(headers, key))
                self.assertEqual(status, expected)
                self.assertEqual(receive_calls, 0)
                self.assertFalse(entered)

    def test_unlock_attempts_are_throttled(self) -> None:
        headers = {"X-UTSM-Request": "dashboard-unlock"}
        wrong = self.client.post("/api/operator/unlock", json={"key": "wrong-key-value-00"}, headers=headers)
        self.assertEqual(wrong.status_code, 401)
        self.assertIn("retry-after", wrong.headers)
        immediate = self.client.post("/api/operator/unlock", json={"key": "operator-test-key"}, headers=headers)
        self.assertEqual(immediate.status_code, 429)
        with app_module.unlock_attempts_lock:
            app_module.unlock_attempts.clear()
        self.assertEqual(
            self.client.post("/api/operator/unlock", json={"key": "operator-test-key"}, headers=headers).status_code,
            200,
        )

    @staticmethod
    def zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
        with tempfile.SpooledTemporaryFile() as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, content in entries:
                    archive.writestr(name, content)
            handle.seek(0)
            return handle.read()

    def post_zip(self, payload: bytes, filename: str = "run.zip"):
        return self.client.post(
            "/api/runs/import",
            data=self.form,
            files=[("files", (filename, payload, "application/zip"))],
            headers=self.headers,
        )

    def test_malformed_encrypted_and_corrupt_archives_are_safe_4xx_errors(self) -> None:
        self.unlock()
        malformed = self.post_zip(b"this is not a zip")
        self.assertEqual(malformed.status_code, 400, malformed.text)
        self.assertNotIn(str(Path(self.temp.name)), malformed.text)

        encrypted = bytearray(self.zip_bytes([("data.csv", CSV_BYTES)]))
        encrypted[encrypted.index(b"PK\x03\x04") + 6] |= 1
        encrypted[encrypted.index(b"PK\x01\x02") + 8] |= 1
        encrypted_response = self.post_zip(bytes(encrypted), "encrypted.zip")
        self.assertEqual(encrypted_response.status_code, 400, encrypted_response.text)
        self.assertIn("encrypted", encrypted_response.text.lower())

        corrupt = bytearray(self.zip_bytes([("data.csv", CSV_BYTES)]))
        local = corrupt.index(b"PK\x03\x04")
        name_length = int.from_bytes(corrupt[local + 26:local + 28], "little")
        extra_length = int.from_bytes(corrupt[local + 28:local + 30], "little")
        data_start = local + 30 + name_length + extra_length
        corrupt[data_start] ^= 0xFF
        corrupt_response = self.post_zip(bytes(corrupt), "corrupt.zip")
        self.assertEqual(corrupt_response.status_code, 400, corrupt_response.text)
        self.assertNotIn(str(Path(self.temp.name)), corrupt_response.text)
        self.assertEqual(app_module.run_library.list_runs(), [])

    def test_health_remains_responsive_during_blocking_import(self) -> None:
        worker = TestClient(app_module.app)
        self.unlock(worker)
        started = threading.Event()
        release = threading.Event()
        original = app_module.run_library.import_paths

        def blocked(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test import timed out")
            return original(*args, **kwargs)

        result: list[int] = []

        def do_import() -> None:
            response = worker.post(
                "/api/runs/import", data=self.form,
                files=[("files", ("run.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
            )
            result.append(response.status_code)

        with patch.object(app_module.run_library, "import_paths", side_effect=blocked):
            thread = threading.Thread(target=do_import)
            thread.start()
            self.assertTrue(started.wait(1))
            before = time.perf_counter()
            heartbeat = self.client.get("/health")
            elapsed = time.perf_counter() - before
            release.set()
            thread.join(3)
        self.assertEqual(heartbeat.status_code, 200)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(result, [201])

    def test_imports_are_serialized(self) -> None:
        first = TestClient(app_module.app)
        second = TestClient(app_module.app)
        self.unlock(first)
        self.unlock(second)
        guard = threading.Lock()
        active = 0
        maximum = 0
        original = app_module.run_library.import_paths

        def observed(*args, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.08)
                return original(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        statuses: list[int] = []

        def send(client: TestClient, name: str) -> None:
            response = client.post(
                "/api/runs/import", data={**self.form, "name": name},
                files=[("files", (f"{name}.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
            )
            statuses.append(response.status_code)

        with patch.object(app_module.run_library, "import_paths", side_effect=observed):
            threads = [
                threading.Thread(target=send, args=(first, "first")),
                threading.Thread(target=send, args=(second, "second")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(4)
        self.assertEqual(sorted(statuses), [201, 201])
        self.assertEqual(maximum, 1)

    def test_replay_compare_and_missing_run_pages(self) -> None:
        csv_one = Path(self.temp.name) / "one.csv"
        csv_two = Path(self.temp.name) / "two.csv"
        gpx_one = Path(self.temp.name) / "one.gpx"
        gpx_two = Path(self.temp.name) / "two.gpx"
        csv_one.write_bytes(CSV_BYTES)
        csv_two.write_bytes(CSV_BYTES.replace(b"3000", b"4000"))
        gpx_one.write_bytes(GPX_BYTES)
        gpx_two.write_bytes(GPX_BYTES)
        first = app_module.run_library.import_paths(
            [gpx_one, csv_one], label="First", date="2026-08-06", source_kind="Browser upload"
        ).runs[0]
        second = app_module.run_library.import_paths(
            [gpx_two, csv_two], label="Second", date="2026-08-07", source_kind="Browser upload"
        ).runs[0]
        self.unlock()
        replay = self.client.get(f"/api/runs/{first['id']}/replay")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(len(replay.json()["samples"]), 3)
        self.assertEqual(replay.json()["samples"][-1]["timestamp_ms"], 3.0)
        self.assertEqual(replay.json()["samples"][-1]["route_progress_percent"], 100.0)
        self.assertTrue(all(sample["route_progress_percent"] < 100 for sample in replay.json()["samples"][:-1]))
        compared = self.client.get("/api/compare", params=[("run", first["id"]), ("run", second["id"])])
        self.assertEqual(compared.status_code, 200, compared.text)
        self.assertEqual(len(compared.json()["runs"]), 2)
        missing = self.client.get("/runs/not-a-run")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("Run unavailable", missing.text)

    def test_typed_malformed_manifests_return_controlled_errors(self) -> None:
        variants = [
            None,
            [],
            {"label": ["wrong"]},
            {"laps": "six"},
            {"distance_m": {"value": 10}},
            {"import_warnings": "not-a-list"},
            {"original_uploads": None},
        ]
        self.unlock()
        for index, manifest in enumerate(variants):
            run_id = f"broken-{index}"
            run_dir = app_module.run_library.runs_dir / run_id
            run_dir.mkdir()
            (run_dir / "data.csv").write_bytes(CSV_BYTES)
            (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        app_module.run_library.invalidate()
        catalog = self.client.get("/api/runs")
        self.assertEqual(catalog.status_code, 200, catalog.text)
        self.assertEqual(catalog.json()["count"], len(variants))
        for index in range(len(variants)):
            run_id = f"broken-{index}"
            with self.subTest(run_id=run_id):
                detail = self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(detail.status_code, 422, detail.text)
                self.assertEqual(detail.json()["detail"], "Run metadata is malformed")
                page = self.client.get(f"/runs/{run_id}")
                self.assertEqual(page.status_code, 422)
                self.assertIn("Run unavailable", page.text)

    def test_strategy_endpoints_explain_artifact_provenance(self) -> None:
        catalog = self.client.get("/api/strategies")
        self.assertEqual(catalog.status_code, 200, catalog.text)
        by_id = {item["id"]: item for item in catalog.json()["strategies"]}
        self.assertTrue(by_id["indy"]["available"])
        self.assertTrue(by_id["autodrome-chaudiere"]["available"])
        detail = self.client.get("/api/strategies/indy")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertGreater(detail.json()["sample_count"], 10)
        self.assertIn("packaged Afternoon Run", detail.json()["strategy"]["provenance"])
        reference = self.client.get("/api/strategies/autodrome-chaudiere").json()["strategy"]
        self.assertIn("not measured Autodrome run data", reference["provenance"])
        download = self.client.get("/api/strategies/indy/files/map")
        self.assertEqual(download.status_code, 200)
        self.assertIn("indy_strategy_map.csv", download.headers["content-disposition"])
        self.assertEqual(self.client.get("/api/strategies/../secret").status_code, 404)

    def test_api_bounds_and_traversal_responses(self) -> None:
        self.assertEqual(self.client.get("/api/runs/../tracks").status_code, 404)
        response = self.client.get("/api/runs/no-run/csv", params={"file": "x.csv", "limit": 501})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.client.get("/api/runs/no-run/csv", params={"file": "x.csv", "offset": 200_001}).status_code,
            422,
        )
        self.assertEqual(self.client.get("/api/runs", params={"q": "x" * 121}).status_code, 422)

    def test_import_validation_does_not_expose_server_paths(self) -> None:
        self.unlock()
        with patch.object(app_module.run_library, "import_paths", side_effect=ValueError(r"bad input at C:\\private\\secret.csv")):
            response = self.client.post(
                "/api/runs/import", data=self.form,
                files=[("files", ("run.csv", CSV_BYTES, "text/csv"))], headers=self.headers,
            )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("private", response.text)
        self.assertNotIn("secret.csv", response.text)

    def test_ui_contains_accessibility_status_mode_and_safe_rendering_contracts(self) -> None:
        shell = self.client.get("/static/shell.js").text
        live = self.client.get("/live").text
        home = self.client.get("/").text
        runs = self.client.get("/runs").text
        strategy = self.client.get("/strategy/autodrome-chaudiere").text
        import_page = self.client.get("/import").text
        dyno = self.client.get("/dyno").text
        css = self.client.get("/static/app.css").text
        run = Path(app_module.STATIC_DIR / "run.html").read_text(encoding="utf-8")
        self.assertIn("aria-current", shell)
        self.assertIn("utsm-dashboard-mode", shell)
        self.assertIn("data-mode=\"simple\"", live)
        self.assertIn("role=\"status\"", live)
        self.assertIn("new Date(record.received_at).getTime()", live)
        self.assertIn("source_type||'car'", live)
        self.assertIn("lastStatusKey", live)
        self.assertIn("Dashboard connected, waiting for car", live)
        self.assertIn("Car signal was lost", live)
        self.assertIn("Car live, GPS not available", live)
        self.assertIn('aria-live="off"', live)
        self.assertIn("liveChartSummary", live)
        self.assertIn("webkitdirectory", import_page)
        self.assertIn("dataTransfer.files", import_page)
        self.assertIn('id="message" role="status" aria-live="polite"', import_page)
        self.assertIn("message.setAttribute('aria-live',isError?'assertive':'polite')", import_page)
        self.assertIn("Unlock saved run data", import_page)
        self.assertIn("Common analysis columns not found", import_page)
        self.assertIn("webkitRelativePath", import_page)
        self.assertIn("entry.selected", import_page)
        self.assertNotIn("toISOString().slice(0,10)", import_page)
        self.assertIn('role="status"', dyno)
        self.assertIn('id="runTime" class="muted small" aria-live="off"', dyno)
        self.assertIn("lastRunState", dyno)
        self.assertIn('id="systemAge" class="muted small" aria-live="off"', home)
        self.assertIn("lastStatusKey", home)
        self.assertIn("catalogAction(run,key,label)", runs)
        self.assertIn("locked_uploaded_count", runs)
        self.assertIn('href="/strategy">Strategy library</a>', runs)
        self.assertIn('href="/strategy">All strategies</a>', strategy)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn('html[data-mode="simple"] [data-advanced]', css)
        self.assertIn("textContent=row[column]", run)
        self.assertIn("Search every value", run)
        self.assertIn("data_kind", run)
        self.assertIn("Imported telemetry", run)
        self.assertIn("Imported route", run)
        self.assertIn("chartSummary", run)
        for action in ("Data", "Charts", "Replay", "Strategy", "Compare"):
            self.assertIn(action, run)
        self.assertNotIn("innerHTML", run)


if __name__ == "__main__":
    unittest.main()
