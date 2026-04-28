# UTSM Proto Telemetry

Python tools for reading car telemetry, aligning it with GPX track data, splitting laps, analyzing energy use, and replaying runs in an interactive dashboard.

The current analysis path is centered on:

1. `dumper.py` for serial telemetry capture.
2. `utsm_telemetry/` for shared parsing, alignment, lap detection, motion, energy, and acceleration helpers.
3. `analyze_strategy.py` for lap, sector, speed-bin, and strategy reports.
4. `build_interactive_dashboard.py` for the multi-run HTML replay dashboard and strategy overlay.

Generated artifacts are reproducible and ignored by Git. Regenerate reports and dashboards into `outputs/` when needed.

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

## Canonical Runs

The main dashboard now packages both known April 11 runs:

- `Utsm.gpx` + `telemetry_dumps\telemetry_20260411_112302.csv`
- `Utsm-2.gpx` + `telemetry_dumps\telemetry_20260411_122713.csv`

Use `--laps 3 --split-method start` as the standard replay/strategy path. The fourth recorded pass is currently treated as unreliable for strategy work, so the dashboard and simulator default to the first three clean laps.

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

The total-energy chart is cumulative run joules versus elapsed time. It spans the whole run and does not reset at lap boundaries.

For the morning run, later samples can become telemetry-sparse. The standard 3-lap workflow avoids the broken fourth-lap interpretation for now.

## Strategy Analysis

Run the corrected afternoon analysis:

```powershell
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
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
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --strategy-step-m 50 --time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output-prefix outputs\speed_strategy
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
python gps_current_heatmap.py Utsm.gpx telemetry_dumps\telemetry_20260411_112302.csv --laps 3 --split-method start --output outputs\current_heatmap.png
```

Run smoke tests and regenerate the multi-run dashboard:

```powershell
python tests\test_smoke.py
python build_interactive_dashboard.py --laps 3 --strategy-step-m 50 --output outputs\telemetry_strategy_dashboard.html
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --strategy-step-m 50 --time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output-prefix outputs\speed_strategy
```

## Notes And Limits

- XY position is a local flat-earth approximation, which is fine for this track scale.
- Nearest-time merging assumes telemetry and GPX clocks can be aligned closely enough.
- Energy is electrical energy estimated from current and voltage; it is not drivetrain output energy.
- GPS acceleration is low bandwidth because it comes from GPX speed changes.
- MPU dynamic acceleration is useful for diagnostics, but sensor orientation and gravity compensation are still imperfect without gyro fusion or a known mounting calibration.
- Generated outputs, caches, and local scratch artifacts should stay out of Git.
