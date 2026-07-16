.venv\Scripts\Activate.ps1# Acceleration, Throttle Proxy, and Strategy Simulation Plan

## Slack Context

This plan is based on the `#proto-telemetry` discussion on 2026-04-26. The goal is to move telemetry strategy from intuition toward data-driven testing for the Schmid Elektronik telemetry award. Rango's fork already added cleanup work, a reusable `utsm_telemetry` package, tests, generated-output organization, and sector-delta plotting.

The next feature direction is:

1. Use the GPS and MPU-6050 acceleration data already present in telemetry dumps to understand vehicle response.
2. Use acceleration as one possible input to a throttle or driver-demand proxy.
3. Compare throttle-like demand against current draw and vehicle response.
4. Build a simulation layer that can replay or alter throttle inputs and estimate current, speed, and energy outcomes.

## Current Repository Baseline

The repo already has enough structure for a first implementation:

- `telemetry_dumps/*.csv` includes `timestamp_ms`, `current_mA`, `voltage_mV`, `ax_x100`, `ay_x100`, `az_x100`, and `amag_x100`.
- `utsm_telemetry/core.py` reads telemetry, aligns it with GPX, merges by time, and derives motion/energy fields.
- `analyze_strategy.py` creates lap, sector, and speed-bin outputs.
- `plot_sector_deltas.py` visualizes sector differences across laps.
- `tests/test_smoke.py` covers the basic helper package.

Important cleanup note: `gps_current_heatmap.py` still duplicates helper logic that now exists in `utsm_telemetry/core.py`. Before adding much more analysis code, it should be refactored to import the package helpers so acceleration/throttle logic only lives in one place.

## Feature 1: Better Acceleration Channels

The current implementation separates acceleration into a GPS-derived channel and an MPU-derived diagnostic channel. This is intentional: the MPU data has gravity offset, unknown mounting orientation, and vibration, while GPS acceleration is lower bandwidth but easier to interpret physically.

Derived channels in `utsm_telemetry/core.py`:

- `imu_ax_m_s2`, `imu_ay_m_s2`, `imu_az_m_s2`: MPU axes converted using milli-g scale, so raw `1000` means about `1 g`.
- `imu_total_g`: vector magnitude in g units from the MPU axes.
- `imu_*_bias_m_s2`: slow rolling-median gravity/bias estimate per axis.
- `imu_*_dynamic_m_s2`: axis value after removing the rolling bias estimate.
- `imu_forward_dynamic_m_s2`: selected MPU axis/sign after dynamic correction.
- `gps_longitudinal_accel_m_s2`: acceleration derived from GPX speed changes.
- `jerk_m_s3`: derivative of smoothed longitudinal acceleration.

The dashboard should show GPS acceleration and MPU dynamic acceleration separately. The MPU axis/sign score table compares `ax`, `-ax`, `ay`, `-ay`, `az`, and `-az` against GPS acceleration and current/power pulses, but it should not be treated as a final mounting calibration.

## Feature 2: Throttle Proxy

There is no direct throttle-position sensor in the current dump, so the first version should call this a throttle proxy, not measured throttle.

Recommended first-pass model:

1. Smooth GPS acceleration and MPU dynamic acceleration separately.
2. Ignore low-speed GPS jitter and samples with unrealistic acceleration.
3. Estimate resistive load from grade and speed.
4. Mark positive driver demand when GPS acceleration plus estimated resistance is positive.
5. Use MPU dynamic acceleration as a diagnostic confidence/response signal, not the primary throttle source.
6. Normalize that demand to a 0 to 1 `throttle_proxy` channel using robust percentiles.

Initial formula:

```text
demand_raw = gps_longitudinal_accel_m_s2
           + rolling_resistance_term
           + aero_drag_term
           + grade_term

throttle_proxy = clipped_percentile_scale(demand_raw, p05, p95)
```

For the first implementation, the resistance terms can be optional and conservative:

- `grade_term = 9.81 * sin(arctan(grade_pct / 100))`
- `rolling_resistance_term = crr * 9.81`, with `crr` defaulting to `0.006`
- `aero_drag_term` can be disabled until mass, CdA, and air density assumptions are documented

This keeps the first pass honest: acceleration alone is a rough proxy, acceleration plus grade compensation is better, and a calibrated model is the later target.

## Feature 3: Current Response Analysis

Once `throttle_proxy` exists, add a report that answers:

- How much current is drawn at different throttle-proxy levels?
- How quickly does current respond after a throttle increase?
- Which sectors show high throttle demand but poor acceleration response?
- Which sectors show coasting opportunities, where current is low and speed is stable?
- Are there repeated spikes where acceleration does not justify the current draw?

Implementation target:

- Add `build_throttle_summary()` in a new module such as `utsm_telemetry/throttle.py`.
- Add throttle columns to the merged/derived lap dataframe.
- Output `PREFIX_throttle_bins.csv` from `analyze_strategy.py`.
- Add a text-report section named `Throttle Proxy Findings`.

Useful output columns:

- `throttle_bin_lo`
- `throttle_bin_hi`
- `sample_count`
- `avg_current_mA`
- `avg_power_w`
- `avg_gps_accel_m_s2`
- `avg_imu_dynamic_accel_m_s2`
- `avg_speed_kph`
- `energy_wh`
- `distance_m`
- `efficiency_wh_per_km`

## Feature 4: Simulation Strategy Layer

The simulation should start simple and empirical. It should not pretend to be a full vehicle dynamics model before calibration data exists.

Version 1 should replay a lap and replace current draw using a learned current-response table:

1. Build bins from historical data: speed, grade, and throttle proxy.
2. For each sample in a proposed strategy, look up expected current/power.
3. Integrate power over time to estimate Wh.
4. Compare total energy and rough lap-time impact against the original lap.

Candidate simulated strategies:

- Smooth throttle: reduce rapid throttle-proxy spikes and distribute demand over a longer window.
- Cap peak demand: clip throttle proxy above a configurable maximum.
- Coast zones: force low throttle in selected sectors.
- Constant target speed: approximate throttle needed to hold a speed on flat or graded sections.
- Sector-specific strategy: apply different caps or smoothing levels per sector.

Add this as a separate script first:

```powershell
python simulate_strategy.py Utsm-2.gpx telemetry_dumps/telemetry_20260411_122713.csv --laps 3 --strategy smooth --output-prefix outputs/sim_smooth
```

Expected outputs:

- `PREFIX_sim_samples.csv`
- `PREFIX_sim_laps.csv`
- `PREFIX_sim_report.txt`

## Implementation Phases

### Phase 0: Repo Hygiene

- Refactor `gps_current_heatmap.py` to import from `utsm_telemetry.core`.
- Fix encoding artifacts in report strings where arrows/dashes render as mojibake.
- Add `pytest` to `requirements.txt` or document that smoke tests can run with `python tests/test_smoke.py`.

### Phase 1: Acceleration Features

- Correct MPU scaling to milli-g units.
- Add GPS acceleration and MPU dynamic acceleration channels.
- Add rolling bias/gravity removal for MPU data.
- Add diagnostic axis/sign scoring.
- Add unit tests for signs, units, smoothing, GPS acceleration, and raw-column preservation.

### Phase 2: Throttle Proxy Analysis

- Add `utsm_telemetry/throttle.py`.
- Compute `throttle_proxy` on merged lap data.
- Add throttle bins and throttle-response findings to the strategy report.
- Add tests for percentile scaling, binning, and no-data behavior.

### Phase 3: Empirical Simulation

- Add `utsm_telemetry/simulation.py`.
- Build an empirical current model from prior lap data.
- Add `simulate_strategy.py` with strategies for smoothing, caps, and sector overrides.
- Compare simulated energy against the original lap using the same report style as `analyze_strategy.py`.

### Phase 4: Calibration and Sensor Improvements

- Confirm IMU orientation on the car.
- Add a real throttle-position channel if firmware/hardware can expose it.
- Record dyno runs with controlled throttle steps.
- Fit a better current-response model using speed, grade, acceleration, and throttle.
- Validate simulation predictions against held-out real laps.

## First Code Change Recommendation

The current useful PR implements the Phase 1 foundation:

1. Correct lap detection for the 3-lap afternoon run.
2. Add the interactive dashboard as the main demo artifact.
3. Split acceleration into GPS acceleration and MPU dynamic acceleration.
4. Add diagnostics and tests around the acceleration processing.

That gives the team honest acceleration traces immediately and creates the foundation for throttle proxy work without overbuilding the simulator before calibration exists.
