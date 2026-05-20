# UTSM Telemetry Strategy Plan

## Mission

Build a data-driven strategy workflow for the Shell Eco-marathon Indianapolis track and for general driver coaching. The target is aligned with the Schmid Elektronik 2026 Data and Telemetry Bootcamp: turn telemetry into knowledge, then into driver action.

Bootcamp references:

- https://schmid-elektronik.ch/bootcamp-sem26/
- https://mybinder.org/v2/gh/LuAmma/sem_racebootcamp/HEAD?labpath=%2FSEM-Python-BE_Package%2FJupyterShowcase_BE.ipynb
- https://mybinder.org/v2/gh/LuAmma/Graph-speed-optimisation/HEAD?urlpath=%2Fdoc%2Ftree%2FBE_SpeedProfileOptimisation.ipynb
- https://mybinder.org/v2/gh/LuAmma/Graph-speed-optimisation/ai-model?urlpath=%2Fdoc%2Ftree%2FAI_SpeedProfileOptimisation.ipynb

The end product should help answer:

- What speed should the driver target on each part of the track?
- Where should the driver accelerate, hold, coast, or avoid surging?
- Where are the biggest energy leaks?
- How do lap time, energy efficiency, fuse limits, and safety trade off?
- What improvement do we expect versus the current baseline?

## Current State

The current `master` checkout is ahead of the older local cleanup branch. It already contains the newer technical strategy work:

- `utsm_telemetry/` shared package for parsing, GPS alignment, lap detection, motion, energy, acceleration, and simulation helpers.
- `analyze_strategy.py` for lap, sector, acceleration, speed-bin, and efficiency reporting.
- `simulate_speed_strategy.py` for empirical accelerate/hold/coast optimization.
- `build_interactive_dashboard.py` for the self-contained multi-run HTML strategy dashboard.
- `tests/test_smoke.py` for regression checks without requiring `pytest`.
- `telemetry_dumps/` containing the preserved April 11 telemetry CSV files.
- `Utsm.gpx` and `Utsm-2.gpx` as the canonical GPS tracks.

The repo is strongest in bootcamp levels 1 and 2: telemetry capture and analytics. It has a meaningful start on level 4 because it can fit an empirical energy model and optimize local speed/action targets, but it is not yet a physics-complete digital twin.

## Bootcamp Mapping

The Schmid bootcamp progression is:

1. Telemetry: create relevant race data on track.
2. Data analytics: recognize correlations and patterns.
3. Live data: send data from the track to IoT.
4. Modelling: maintain a physics-based digital twin.
5. Holistic: advanced data-driven racing.

Current status:

- Level 1 is partially complete through serial dumps and replayable telemetry datasets.
- Level 2 is functional through lap summaries, sector analysis, speed bins, dashboard charts, and current/energy heatmaps.
- Level 3 is not implemented yet. The workflow is offline, not live.
- Level 4 is partially implemented empirically through `simulate_speed_strategy.py`, but still needs vehicle physics and calibrated parameters.
- Level 5 is the main target: driver-facing strategy that links data, vehicle constraints, route context, and race execution.

## Technical Gaps

### Data and calibration

- Add a run manifest that ties each run to GPX, telemetry, driver, vehicle setup, weather, and notes.
- Document units and calibration for every telemetry channel.
- Add official energy sensor integration planning for Joulemeter, Liquid Flowmeter, or Gas Flowmeter data.
- Add weather/context inputs: wind, rain, temperature, track condition.
- Add setup context: vehicle mass, driver mass, tire pressure, gearing, battery state, and mechanical changes.

### Track model

- Build an Indianapolis/Shell Eco-marathon route model with cumulative distance.
- Label straights, corners, start/finish, elevation-sensitive sections, and caution zones.
- Replace equal-distance-only sectors with named track zones for driver coaching.
- Keep equal-distance 25 m to 100 m optimizer steps internally, but merge them into human-readable dashboard regions.

### Vehicle model

- Estimate rolling resistance, aerodynamic drag, drivetrain efficiency, vehicle mass, and power/current limits.
- Fit acceleration behaviour and coast deceleration from real telemetry, using only realistic low-current deceleration samples for coast.
- Include grade-aware and wind-aware speed prediction.
- Add constraints for corner speed, traction, rain, voltage sag, and fuse limits.

### Optimizer

- Preserve local pulse-and-coast decisions at roughly 50 m resolution.
- Optimize for energy subject to lap-time, fuse-current, speed, and safety constraints.
- Report predicted lap time, total energy, Wh/km, fuse-risk duration, coast time, and target action regions.
- Compare optimized strategy against actual run baselines with error metrics.

### Driver interface

- Produce a simple pre-run strategy sheet: track zone, target speed, action, and warning.
- Produce a post-run review: followed plan, missed plan, largest energy leaks, and next adjustment.
- Keep the dashboard useful for engineering review, but simplify the driver-facing output.

### Award evidence

- Build a 10-page Data and Telemetry Off-Track Award story from repo outputs.
- Include architecture diagrams for capture, processing, analysis, and future live telemetry.
- Include baseline-vs-strategy tables with quantified expected gains.
- Clearly separate observed data, inferred model outputs, and proposed race strategy.

## Immediate Next Steps

1. Add a manifest file for the morning and afternoon April 11 runs.
2. Build a track-zone CSV from the canonical GPX/reference track.
3. Add a vehicle-parameter config file for mass, wheel size, current limit, drag, rolling resistance, and drivetrain assumptions.
4. Extend the simulator report with baseline-vs-optimized tables suitable for the award writeup.
5. Generate one approved dashboard demo and one concise driver strategy sheet.
6. Add tests around any new manifest, track-zone, and config parsing.

## Demo Approval Path

Before committing future strategy changes, run:

```powershell
python tests\test_smoke.py
python build_interactive_dashboard.py --laps 3 --strategy-step-m 50 --display-region-min-m 150 --strategy-time-tolerance-pct 3 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output outputs\telemetry_strategy_dashboard.html
```

Then open `outputs\telemetry_strategy_dashboard.html` and confirm the dashboard, replay, charts, and strategy overlay look right.

