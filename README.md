# UTSM Proto Telemetry

Python tools for reading car telemetry, aligning it with GPX track data, splitting laps, analyzing energy use, and replaying runs in an interactive dashboard.

The current analysis path is centered on:

1. `dumper.py` for serial telemetry capture.
2. `utsm_telemetry/` for shared parsing, alignment, lap detection, motion, energy, and acceleration helpers.
3. `analyze_strategy.py` for lap, sector, speed-bin, and strategy reports.
4. `build_interactive_dashboard.py` for the multi-run HTML replay dashboard and strategy overlay.

Generated artifacts are reproducible and ignored by Git. Regenerate reports and dashboards into `outputs/` when needed.

## Strategy Plan

See `PLAN.md` for the current project state, bootcamp alignment, and remaining gaps toward a data-driven Shell Eco-marathon Indianapolis driver strategy. The plan is based on the Schmid Elektronik 2026 bootcamp flow: telemetry, analytics, live data, modelling, and holistic race strategy.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Smoke tests can run without pytest:

```powershell
python tests\test_smoke.py
```

## Data Inputs

Telemetry CSVs must include:

- `timestamp_ms`
- `current_mA`
- `voltage_mV`
- `ax_x100`
- `ay_x100`
- `az_x100`

Some dumps also include `amag_x100`. Despite the column names, the MPU-6050 acceleration values in the 2026 afternoon data behave like milli-g units: about `1000` means `1 g`.

GPX files must include latitude, longitude, elevation, and timestamps. Speed is derived from GPX point-to-point movement on the GPX sampling clock, not by integrating noisy accelerometer data.

## Runs

The dashboard auto-discovers runs from `data\runs\`. Each run is just a
subfolder containing exactly one `.gpx` file and one telemetry `.csv` file -
folder and file names can be anything (e.g. `data\runs\june-20-test\` holding
`june-20-2026.gpx` and `Telemetry_001.csv`). To add a new run, drop a new
subfolder in there; no code changes are needed and it will show up in the
dashboard's run-switcher automatically. The subfolder name becomes the run's
label (e.g. `june-20-test` -> "June 20 Test").

Currently packaged:

- `data\runs\morning-run\` - `Utsm.gpx` + `telemetry_20260411_112302.csv`
- `data\runs\afternoon-run\` - `Utsm-2.gpx` + `telemetry_20260411_122713.csv`

Reference tracks without telemetry live separately under `data\tracks\`. The
dashboard discovers one `*-centerline.gpx` file per track folder and adds it to
the **Map** selector without treating it as a recorded run. Currently packaged:

- `data\tracks\autodrome-chaudiere\` - a smoothed 399.3 m centerline and
  model-derived efficiency strategy for Autodrome Chaudière near Québec City,
  resampled to 80 points at approximately 5 m spacing

Use `--laps 3 --split-method start` as the standard replay/strategy path. The fourth recorded pass is currently treated as unreliable for strategy work, so the dashboard and simulator default to the first three clean laps.

To point the dashboard at a different folder of runs, pass `--runs-dir`. To
build a one-off dashboard from a single pair of files outside that structure,
pass `--gps` and `--telemetry` directly (this skips auto-discovery and
produces a single "Custom run" entry).

## Interactive Dashboard

Build the dashboard:

```powershell
python build_interactive_dashboard.py --laps 3 --output outputs\telemetry_strategy_dashboard.html
```

Useful strategy knobs:

```powershell
python build_interactive_dashboard.py --laps 3 --strategy-step-m 50 --display-region-min-m 150 --strategy-time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output outputs\telemetry_strategy_dashboard.html
```

Open `outputs\telemetry_strategy_dashboard.html` in a browser. It is a self-contained HTML file with:

- run switcher for the morning and afternoon datasets
- map switcher for recorded-run GPS or packaged geometry-only reference tracks
- one manual time slider per selected run
- play/pause replay
- full-course gray reference trace
- current-lap colored trail
- explicit action-colored strategy regions on the map (`accelerate`, `hold`, `coast`)
- segment labels on the map
- synchronized current, speed, GPS acceleration, MPU dynamic acceleration, power, and cumulative total-energy charts
- current, speed, power, and cumulative-energy prediction overlays
- visible `20 A` fuse threshold on the current chart
- map mode switch between action-region view and metric-colored trail view
- merged strategy labels, while the underlying optimizer can switch actions every `50 m`

The strategy layer uses the same optimized profile as `simulate_speed_strategy.py`. In the dashboard:

- `Strategy` toggle shows or hides the simulated action regions
- `Labels` toggle shows or hides per-segment labels
- `Strategy target speed` can still be selected as a map coloring metric
- the speed chart overlays actual speed against optimized target speed
- the current chart overlays predicted current and marks the fuse threshold
- the total-energy chart overlays actual cumulative joules against predicted cumulative joules
- the live readout shows current segment, action, target speed, predicted current, and predicted power

Acceleration is split into two separate channels:

- `GPS acceleration`: derived from GPS speed changes, smoother and more physically interpretable for vehicle speed trend.
- `MPU dynamic acceleration`: MPU-6050 axis data scaled as milli-g, bias/gravity corrected with a rolling median, and kept as a diagnostic vibration/response channel.

The dashboard payload also includes MPU axis/sign diagnostic correlations for `ax`, `-ax`, `ay`, `-ay`, `az`, and `-az`.

When a reference track is selected, the charts continue to describe the
selected recorded run. A reference track can also include a separately
generated model strategy: the Autodrome map colors its centerline by target
speed and clearly labels the result as model-derived rather than measured
telemetry.

## Autodrome Maximum-Efficiency Strategy

Regenerate the initial Autodrome strategy from the packaged centerline and the
existing afternoon vehicle telemetry model:

```powershell
python generate_reference_strategy.py `
  data\tracks\autodrome-chaudiere\autodrome-chaudiere-centerline.gpx `
  data\runs\afternoon-run\Utsm-2.gpx `
  data\runs\afternoon-run\telemetry_20260411_122713.csv `
  --output-prefix data\tracks\autodrome-chaudiere\autodrome-chaudiere `
  --preview outputs\autodrome-chaudiere-efficiency-strategy.png `
  --model-laps 3 `
  --target-lap-time-sec 60 `
  --strategy-step-m 20 `
  --speed-min-kph 8 `
  --speed-max-kph 35 `
  --start-speed-kph 24
```

The optimizer minimizes predicted electrical energy subject to the 60-second
maximum lap time, closed-loop start/end speed, per-segment speed-change, motor,
and fuse constraints. A lap-time constraint is required for a useful result:
without one, maximum efficiency degenerates to driving at the minimum allowed
speed.

To run the demo:

```powershell
python build_interactive_dashboard.py --laps 3 --output outputs\telemetry_strategy_dashboard.html
Start-Process outputs\telemetry_strategy_dashboard.html
```

Choose **Autodrome Chaudière** from the dashboard's **Map** selector. The
colored centerline is the target-speed curve; purple is slower and yellow is
faster. The map readout shows the predicted lap time and energy.

This is an initial transferred-model strategy, not validated Autodrome
telemetry. The generator removes the source circuit's absolute-position model
term before transfer, but the vehicle and surface assumptions still need to be
refit after the first real Autodrome laps.

## Reference Track Preprocessing

The original Autodrome Chaudière Google Earth export contains two coarse
closed paths named `Outer Ring` and `Inner Ring`, not a drivable centerline.
Regenerate the packaged track and a visual audit image with:

```powershell
python preprocess_track.py `
  data\tracks\autodrome-chaudiere\google-earth-boundaries.gpx `
  data\tracks\autodrome-chaudiere\autodrome-chaudiere-centerline.gpx `
  --preview outputs\autodrome-chaudiere-preprocessing.png `
  --spacing-m 5 `
  --smooth-window-m 12
```

The preprocessing stages are:

1. Parse the inner and outer GPX tracks and remove duplicate closing points.
2. Project latitude/longitude into a local metre coordinate system.
3. Normalize ring direction and cyclically align their start phases.
4. Resample both boundaries uniformly and average paired points.
5. Apply a periodic Gaussian smoother so the closed seam remains continuous.
6. Resample the smoothed centerline at fixed spacing and export an untimed,
   geometry-only GPX.

For the supplied source this produces a 399.3 m closed centerline from a
450.6 m outer boundary and 357.2 m inner boundary. The result closely matches
the published 0.4 km track length, but it remains satellite-traced geometry;
replace or calibrate it against a real driven GPS lap before using metre-level
driver guidance.

The total-energy chart is cumulative run joules versus elapsed time. It spans the whole run and does not reset at lap boundaries.

For the morning run, later samples can become telemetry-sparse. The standard 3-lap workflow avoids the broken fourth-lap interpretation for now.

## Strategy Analysis

Run the corrected afternoon analysis:

```powershell
python analyze_strategy.py data\runs\afternoon-run\Utsm-2.gpx data\runs\afternoon-run\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
```

This writes:

- `PREFIX_laps.csv`
- `PREFIX_sectors.csv`
- `PREFIX_speed_bins.csv`
- `PREFIX_report.txt`

The analysis computes:

- lap duration and distance
- GPX-derived speed
- current, voltage, power, and integrated Wh
- Wh/km efficiency
- elevation gain/loss and grade
- GPS acceleration and MPU dynamic acceleration
- equal-distance sector summaries
- flat-road speed efficiency bins

## Speed Strategy Simulation

Run the empirical 3-state optimizer:

```powershell
python simulate_speed_strategy.py data\runs\afternoon-run\Utsm-2.gpx data\runs\afternoon-run\telemetry_20260411_122713.csv --laps 3 --split-method start --strategy-step-m 50 --time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output-prefix outputs\speed_strategy
```

This writes:

- `PREFIX_strategy_profile.csv`
- `PREFIX_strategy_samples.csv`
- `PREFIX_strategy_report.txt`

The dashboard generator runs the same optimizer internally so the HTML stays in sync with the standalone strategy report.

The current optimizer is empirical and deterministic. It fits current and power models from historical samples, simulates explicit `accelerate` / pulse-and-coast `hold` / zero-throttle `coast` behavior by short equal-distance segment, minimizes predicted joules near the recorded 3-lap pace, and rejects strategies that stay above the fuse current threshold for too long. `--segments` is still accepted as a legacy fixed-count override, but `--strategy-step-m 50` is the default path because the driver can change accel/hold/coast state within `100 m`.

## Optional Animation Fallback

The interactive dashboard is the main visualization. The older animation scripts are kept as optional fallback/demo tools:

```powershell
python animate_run.py --help
python build_animation_gallery.py --help
```

Use them only when a pre-rendered GIF/HTML gallery is specifically needed. They are slower and less useful for data inspection than the interactive dashboard.

## Common Commands

Capture telemetry from the serial device:

```powershell
python dumper.py --port COM13
```

Generate legacy current heatmaps:

```powershell
python gps_current_heatmap.py data\runs\morning-run\Utsm.gpx data\runs\morning-run\telemetry_20260411_112302.csv --laps 3 --split-method start --output outputs\current_heatmap.png
```

Run smoke tests and regenerate the multi-run dashboard:

```powershell
python tests\test_smoke.py
python build_interactive_dashboard.py --laps 3 --strategy-step-m 50 --output outputs\telemetry_strategy_dashboard.html
python analyze_strategy.py data\runs\afternoon-run\Utsm-2.gpx data\runs\afternoon-run\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
python simulate_speed_strategy.py data\runs\afternoon-run\Utsm-2.gpx data\runs\afternoon-run\telemetry_20260411_122713.csv --laps 3 --split-method start --strategy-step-m 50 --time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output-prefix outputs\speed_strategy
```

## Notes And Limits

- XY position is a local flat-earth approximation, which is fine for this track scale.
- Nearest-time merging assumes telemetry and GPX clocks can be aligned closely enough.
- Energy is electrical energy estimated from current and voltage; it is not drivetrain output energy.
- GPS acceleration is low bandwidth because it comes from GPX speed changes.
- MPU dynamic acceleration is useful for diagnostics, but sensor orientation and gravity compensation are still imperfect without gyro fusion or a known mounting calibration.
- Generated outputs, caches, and local scratch artifacts should stay out of Git.

## Live LTE Dashboard

The live page is separate from the generated historical replay dashboard. It
runs a small FastAPI server that accepts one telemetry record at a time, keeps
the most recent records in memory, and sends them to connected browsers over a
WebSocket.

Start it in PowerShell:

```powershell
$env:UTSM_TELEMETRY_API_KEY = "replace-this-for-real-tests"
python -m uvicorn live_dashboard.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/live`. Test the complete page without hardware in a
second PowerShell window:

```powershell
python send_live_test.py --api-key "replace-this-for-real-tests" --gps
```

The page includes:

- current, voltage, power, and acceleration gauges
- four rolling live charts
- a table preserving the current telemetry CSV column names
- an optional live map and trail when latitude/longitude arrive
- a stale-data indicator when the car has not reported for five seconds

The API endpoint is `POST /api/live/telemetry` and requires the
`X-Telemetry-Key` header. Records retain the existing seven CSV fields and add
packet identity plus optional `latitude` and `longitude` fields.

### Reaching the local server from LTE

`localhost` and private Wi-Fi addresses are not reachable from a cellular
modem. For a development test, expose port 8000 with a temporary HTTPS tunnel.
For example, after installing `cloudflared`:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Copy the generated `https://...trycloudflare.com` hostname into the relay's
`TELEMETRY_ENDPOINT`, followed by `/api/live/telemetry`. Use the same API key in
the relay configuration and the `UTSM_TELEMETRY_API_KEY` environment variable.
Quick tunnel hostnames change when the tunnel restarts and are intended only
for development.

### Verified ESP-NOW to LTE demo

The live page was verified end to end with an ESP32-C3 SuperMini sending dummy
telemetry over ESP-NOW channel 1 to a LILYGO T-A7670G R2. The A7670G relayed
the packets over a Public Mobile LTE connection through a Cloudflare quick
tunnel. Packet sequence, current, voltage, acceleration, and fake GPS values
arrived intact at the dashboard.

Use the matching firmware sketches from
[`UTSM-proto/utsm-telem-firmware`](https://github.com/UTSM-proto/utsm-telem-firmware):

- `lte_relay/lte_relay.ino` on the A7670G, with `LTE_DUMMY_TEST_MODE = false`
- `telem-v1/espnow_dummy_sender/espnow_dummy_sender.ino` on the C3 for the
  full-path dummy test
- `telem-v1/telemetry_gpio1_led_sd_per_session.ino` on the C3 to return to
  real sensor and SD-backed telemetry

The successful full-path serial indicators are:

```text
C3: C3 ESP-NOW queued seq=N
Relay: Dashboard POST status=202
Relay: LIVE seq=N delivered
```

The relay and server must use the same API key. The relay endpoint must contain
the complete quick-tunnel URL ending in `/api/live/telemetry`.
