from __future__ import annotations

import math

import numpy as np
import pandas as pd

ACTION_ACCELERATE = "accelerate"
ACTION_HOLD = "hold"
ACTION_COAST = "coast"
ACTION_ORDER = (ACTION_ACCELERATE, ACTION_HOLD, ACTION_COAST)
DEFAULT_MOTOR_PROFILE = {
    "name": "Koford 60 mm 24 V",
    "nominal_voltage_v": 24.0,
    "nominal_output_w": 256.0,
    "nominal_speed_rpm": 7560.0,
    "top_speed_kph": 39.0,
    "wheel_diameter_m": 0.50,
    "vehicle_mass_kg": 100.0,
    "rolling_resistance_coeff": 0.008,
    "drivetrain_efficiency": 0.82,
}


def build_full_run_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    if "run_cumdist_m" not in df.columns:
        df["run_cumdist_m"] = pd.to_numeric(df["dist_m"], errors="coerce").fillna(0.0).cumsum()
    if "cum_energy_j" not in df.columns:
        df["cum_energy_j"] = pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0).cumsum()
    return df


def classify_strategy_action(delta_kph: float, hold_delta_kph: float = 1.0) -> str:
    if delta_kph > hold_delta_kph:
        return ACTION_ACCELERATE
    if delta_kph < -hold_delta_kph:
        return ACTION_COAST
    return ACTION_HOLD


def build_motor_config(
    wheel_diameter_m: float = 0.50,
    vehicle_mass_kg: float = 100.0,
    rolling_resistance_coeff: float = 0.008,
    drivetrain_efficiency: float = 0.82,
) -> dict[str, float | str]:
    if wheel_diameter_m <= 0:
        raise ValueError("wheel_diameter_m must be positive.")
    if vehicle_mass_kg <= 0:
        raise ValueError("vehicle_mass_kg must be positive.")
    config = dict(DEFAULT_MOTOR_PROFILE)
    config["wheel_diameter_m"] = float(wheel_diameter_m)
    config["vehicle_mass_kg"] = float(vehicle_mass_kg)
    config["rolling_resistance_coeff"] = float(rolling_resistance_coeff)
    config["drivetrain_efficiency"] = float(drivetrain_efficiency)
    config["inferred_gear_ratio"] = infer_gear_ratio(
        top_speed_kph=float(config["top_speed_kph"]),
        motor_top_rpm=float(config["nominal_speed_rpm"]),
        wheel_diameter_m=wheel_diameter_m,
    )
    return config


def infer_gear_ratio(top_speed_kph: float, motor_top_rpm: float, wheel_diameter_m: float) -> float:
    if top_speed_kph <= 0 or motor_top_rpm <= 0 or wheel_diameter_m <= 0:
        raise ValueError("top_speed_kph, motor_top_rpm, and wheel_diameter_m must be positive.")
    vehicle_top_speed_m_s = top_speed_kph / 3.6
    wheel_rpm = vehicle_top_speed_m_s / (math.pi * wheel_diameter_m) * 60.0
    return float(motor_top_rpm / wheel_rpm)


def build_strategy_segments(df: pd.DataFrame, segments: int) -> pd.DataFrame:
    if segments < 2:
        raise ValueError("segments must be at least 2.")
    df = build_full_run_distance(df)
    total_dist = float(df["run_cumdist_m"].iloc[-1])
    if total_dist <= 0:
        raise ValueError("run_cumdist_m must be positive.")

    edges = np.linspace(0.0, total_dist, segments + 1)
    rows = []
    for idx in range(segments):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == segments - 1:
            seg = df[(df["run_cumdist_m"] >= lo) & (df["run_cumdist_m"] <= hi)].copy()
        else:
            seg = df[(df["run_cumdist_m"] >= lo) & (df["run_cumdist_m"] < hi)].copy()
        if seg.empty:
            continue
        length_m = hi - lo
        speed = pd.to_numeric(seg["speed_kph"], errors="coerce").dropna()
        grade = pd.to_numeric(seg["grade_pct"], errors="coerce").dropna()
        current = pd.to_numeric(seg["current_mA"], errors="coerce").dropna()
        power = pd.to_numeric(seg["power_w"], errors="coerce").dropna()
        energy_j = pd.to_numeric(seg["energy_j"], errors="coerce").fillna(0.0).sum()
        dt_s = pd.to_numeric(seg["dt_s"], errors="coerce").fillna(0.0).sum()
        rows.append({
            "segment": idx + 1,
            "dist_start_m": lo,
            "dist_end_m": hi,
            "length_m": length_m,
            "center_frac": ((lo + hi) * 0.5) / total_dist,
            "baseline_speed_kph": float(speed.median()) if not speed.empty else 0.0,
            "baseline_grade_pct": float(grade.mean()) if not grade.empty else 0.0,
            "baseline_current_mA": float(current.mean()) if not current.empty else 0.0,
            "baseline_power_w": float(power.mean()) if not power.empty else 0.0,
            "baseline_energy_j": float(energy_j),
            "baseline_time_s": float(dt_s) if dt_s > 0 else _segment_time_s(
                length_m,
                float(speed.median()) if not speed.empty else 0.0,
            ),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No strategy segments could be built.")
    return out


def build_strategy_segments_by_distance(df: pd.DataFrame, strategy_step_m: float = 50.0) -> pd.DataFrame:
    if strategy_step_m <= 0:
        raise ValueError("strategy_step_m must be positive.")
    df = build_full_run_distance(df)
    total_dist = float(df["run_cumdist_m"].iloc[-1])
    if total_dist <= 0:
        raise ValueError("run_cumdist_m must be positive.")
    segments = max(int(math.ceil(total_dist / strategy_step_m)), 2)
    return build_strategy_segments(df, segments=segments)


def fit_empirical_energy_model(df: pd.DataFrame, ridge: float = 1e-3) -> dict[str, object]:
    df = build_full_run_distance(df)
    fit = pd.DataFrame({
        "speed_kph": pd.to_numeric(df.get("speed_kph"), errors="coerce"),
        "grade_pct": pd.to_numeric(df.get("grade_pct"), errors="coerce"),
        "gps_accel_m_s2": pd.to_numeric(df.get("gps_longitudinal_accel_m_s2"), errors="coerce"),
        "current_mA": pd.to_numeric(df.get("current_mA"), errors="coerce"),
        "power_w": pd.to_numeric(df.get("power_w"), errors="coerce"),
        "run_cumdist_m": pd.to_numeric(df.get("run_cumdist_m"), errors="coerce"),
    }).dropna()
    if len(fit) < 10:
        raise ValueError("Not enough samples to fit the empirical strategy model.")

    total_dist = max(float(fit["run_cumdist_m"].iloc[-1]), 1.0)
    fit["pos_accel_m_s2"] = fit["gps_accel_m_s2"].clip(lower=0.0)
    fit["neg_accel_m_s2"] = (-fit["gps_accel_m_s2"]).clip(lower=0.0)
    fit["uphill_grade_pct"] = fit["grade_pct"].clip(lower=0.0)
    fit["downhill_grade_pct"] = (-fit["grade_pct"]).clip(lower=0.0)
    fit["position_frac"] = fit["run_cumdist_m"] / total_dist
    fit["action"] = fit["gps_accel_m_s2"].map(classify_strategy_action)
    fit["is_accelerate"] = (fit["action"] == ACTION_ACCELERATE).astype(float)
    fit["is_hold"] = (fit["action"] == ACTION_HOLD).astype(float)
    fit["is_coast"] = (fit["action"] == ACTION_COAST).astype(float)

    design = _design_matrix(fit)
    current_coeffs = _solve_ridge(design, fit["current_mA"].to_numpy(dtype=float), ridge)
    power_coeffs = _solve_ridge(design, fit["power_w"].to_numpy(dtype=float), ridge)
    positive_accel = fit["gps_accel_m_s2"] > max(float(fit["gps_accel_m_s2"].quantile(0.70)), 0.02)
    high_current = fit["current_mA"] > float(fit["current_mA"].quantile(0.75))
    throttle_like = fit[positive_accel | high_current]
    if throttle_like.empty:
        throttle_like = fit
    on_current_mA = float(throttle_like["current_mA"].quantile(0.90))
    cruise_current_mA = float(fit.loc[fit["gps_accel_m_s2"].abs() <= 0.03, "current_mA"].median())
    if not math.isfinite(cruise_current_mA):
        cruise_current_mA = float(fit["current_mA"].median())
    median_voltage_v = float(
        pd.to_numeric(df.get("voltage_mV"), errors="coerce").dropna().median() / 1000.0
    ) if "voltage_mV" in df.columns else 24.0
    if not math.isfinite(median_voltage_v) or median_voltage_v <= 0:
        median_voltage_v = 24.0
    coast_current_threshold_mA = max(500.0, float(fit["current_mA"].quantile(0.20)))
    coast_fit = fit[
        (fit["current_mA"] <= coast_current_threshold_mA)
        & (fit["speed_kph"] >= 2.0)
        & (fit["gps_accel_m_s2"] < -0.03)
    ].copy()
    if len(coast_fit) < 8:
        coast_fit = fit[(fit["gps_accel_m_s2"] < -0.03) & (fit["speed_kph"] >= 2.0)].copy()
    if len(coast_fit) >= 4:
        coast_design = _coast_design_matrix(coast_fit)
        coast_coeffs = _solve_ridge(coast_design, coast_fit["gps_accel_m_s2"].to_numpy(dtype=float), ridge)
    else:
        coast_coeffs = np.array([-0.25, 0.0, 0.0, -0.098])
    return {
        "current_coeffs": current_coeffs,
        "power_coeffs": power_coeffs,
        "coast_coeffs": coast_coeffs,
        "ridge": float(ridge),
        "on_current_mA": max(on_current_mA, cruise_current_mA, 1000.0),
        "cruise_current_mA": max(cruise_current_mA, 0.0),
        "median_voltage_v": median_voltage_v,
        "coast_current_threshold_mA": coast_current_threshold_mA,
        "coast_sample_count": int(len(coast_fit)),
    }


def predict_current_mA(
    model: dict[str, object],
    speed_kph: float,
    accel_m_s2: float,
    grade_pct: float,
    position_frac: float,
    action: str,
) -> float:
    return predict_strategy_electrical(
        model,
        speed_kph=speed_kph,
        accel_m_s2=accel_m_s2,
        grade_pct=grade_pct,
        position_frac=position_frac,
        action=action,
    )["avg_current_mA"]


def predict_power_w(
    model: dict[str, object],
    speed_kph: float,
    accel_m_s2: float,
    grade_pct: float,
    position_frac: float,
    action: str,
) -> float:
    return predict_strategy_electrical(
        model,
        speed_kph=speed_kph,
        accel_m_s2=accel_m_s2,
        grade_pct=grade_pct,
        position_frac=position_frac,
        action=action,
    )["avg_power_w"]


def predict_strategy_electrical(
    model: dict[str, object],
    speed_kph: float,
    accel_m_s2: float,
    grade_pct: float,
    position_frac: float,
    action: str,
    motor_config: dict[str, float | str] | None = None,
) -> dict[str, float]:
    if action == ACTION_COAST:
        return {
            "avg_current_mA": 0.0,
            "avg_power_w": 0.0,
            "peak_current_mA": 0.0,
            "on_current_mA": 0.0,
            "throttle_duty": 0.0,
            "throttle_level": 0.0,
            "fuse_risk_duration_s": 0.0,
        }

    raw_current = max(
        _predict_linear(
            model["current_coeffs"],
            speed_kph=speed_kph,
            accel_m_s2=accel_m_s2,
            grade_pct=grade_pct,
            position_frac=position_frac,
            action=action,
        ),
        0.0,
    )
    raw_power = max(
        _predict_linear(
            model["power_coeffs"],
            speed_kph=speed_kph,
            accel_m_s2=accel_m_s2,
            grade_pct=grade_pct,
            position_frac=position_frac,
            action=action,
        ),
        0.0,
    )
    on_current = float(model.get("on_current_mA", max(raw_current, 1000.0)))
    cruise_current = float(model.get("cruise_current_mA", raw_current))
    voltage_v = float(model.get("median_voltage_v", 24.0))
    physics_power_w = _physics_propulsion_power_w(speed_kph, accel_m_s2, grade_pct, motor_config)
    physics_current_mA = physics_power_w / max(voltage_v, 1.0) * 1000.0
    if action == ACTION_ACCELERATE:
        duty = _clamp(0.22 + max(accel_m_s2, 0.0) * 1.8, 0.18, 1.0)
        on_current = max(on_current, raw_current / max(duty, 0.05))
        avg_current = max(raw_current, on_current * duty)
        throttle_level = duty
        spike_multiplier = 1.0 + _clamp(max(accel_m_s2, 0.0) * 1.2, 0.0, 1.25)
    else:
        avg_current = max(raw_current, cruise_current)
        on_current = max(on_current, avg_current * 1.35)
        duty = _clamp(avg_current / max(on_current, 1.0), 0.04, 0.85)
        avg_current = on_current * duty
        throttle_level = duty
        spike_multiplier = 1.08
    avg_current = max(avg_current, physics_current_mA)
    avg_power = max(raw_power, avg_current / 1000.0 * voltage_v, physics_power_w)
    peak_current = max(on_current * spike_multiplier, avg_current)
    return {
        "avg_current_mA": float(avg_current),
        "avg_power_w": float(avg_power),
        "peak_current_mA": float(peak_current),
        "on_current_mA": float(on_current),
        "throttle_duty": float(duty),
        "throttle_level": float(throttle_level),
        "fuse_risk_duration_s": 0.0,
    }


def optimize_speed_profile(
    segments_df: pd.DataFrame,
    model: dict[str, object],
    time_budget_sec: float,
    speed_min_kph: float,
    speed_max_kph: float,
    max_delta_kph_per_segment: float,
    speed_step_kph: float = 1.0,
    hold_delta_kph: float = 1.0,
    fuse_current_ma: float = 20000.0,
    fuse_max_duration_sec: float = 1.0,
    current_penalty_weight: float = 5.0,
    motor_config: dict[str, float | str] | None = None,
    start_speed_kph: float = 0.0,
    min_time_sec: float | None = None,
) -> pd.DataFrame:
    if time_budget_sec <= 0:
        raise ValueError("time_budget_sec must be positive.")
    if speed_min_kph <= 0 or speed_max_kph <= speed_min_kph:
        raise ValueError("speed bounds are invalid.")

    if motor_config is not None:
        speed_max_kph = min(speed_max_kph, float(motor_config.get("top_speed_kph", speed_max_kph)))

    candidate_speeds = np.arange(
        speed_min_kph,
        speed_max_kph + speed_step_kph * 0.5,
        speed_step_kph,
    )
    initial_speed = float(np.clip(start_speed_kph, 0.0, speed_max_kph))

    def solve_for_lambda(lambda_time: float) -> pd.DataFrame:
        dp: list[dict[tuple[float, float], tuple[float, float, float, tuple[float, float] | None, str]]] = []
        for idx, row in segments_df.iterrows():
            length_m = float(row["length_m"])
            grade_pct = float(row["baseline_grade_pct"])
            position_frac = float(row["center_frac"])
            state_costs: dict[tuple[float, float], tuple[float, float, float, tuple[float, float] | None, str]] = {}
            for speed in candidate_speeds:
                speed = float(speed)
                if idx == 0:
                    prev_states = {(initial_speed, 0.0): (0.0, 0.0, 0.0, None, ACTION_HOLD)}
                else:
                    prev_states = dp[idx - 1]

                for (prev_speed, prev_over_s), (prev_cost, prev_time, _prev_current, _prev_key, _prev_action) in prev_states.items():
                    delta_kph = speed - float(prev_speed)
                    launch_from_stop = idx == 0 and float(prev_speed) < speed_min_kph
                    if abs(delta_kph) > max_delta_kph_per_segment and not launch_from_stop:
                        continue
                    action = classify_strategy_action(delta_kph, hold_delta_kph=hold_delta_kph)
                    if action == ACTION_COAST:
                        coast_exit = _coast_exit_speed_kph(float(prev_speed), length_m, grade_pct, model)
                        coast_exit = max(coast_exit, speed_min_kph)
                        coast_tolerance = max(speed_step_kph * 1.5, 1.0)
                        if abs(speed - coast_exit) > coast_tolerance:
                            continue
                    accel_m_s2 = _signed_accel_from_speed_change(float(prev_speed), speed, length_m)
                    time_s = _segment_transition_time_s(length_m, float(prev_speed), speed)
                    electrical = predict_strategy_electrical(
                        model,
                        speed_kph=speed,
                        accel_m_s2=accel_m_s2,
                        grade_pct=grade_pct,
                        position_frac=position_frac,
                        action=action,
                        motor_config=motor_config,
                    )
                    pred_current_mA = electrical["avg_current_mA"]
                    pred_power_w = electrical["avg_power_w"]
                    pred_peak_current_mA = electrical["peak_current_mA"]
                    throttle_duty = electrical["throttle_duty"]
                    risk_time_s = _fuse_burst_duration_s(time_s, throttle_duty) if pred_peak_current_mA > fuse_current_ma else 0.0
                    over_fuse = risk_time_s > 0.0
                    over_s = prev_over_s + risk_time_s if over_fuse else 0.0
                    if over_s > fuse_max_duration_sec:
                        continue
                    pred_energy_j = pred_power_w * time_s
                    current_penalty = _soft_current_penalty(
                        pred_peak_current_mA,
                        time_s,
                        fuse_current_ma=fuse_current_ma,
                        weight=current_penalty_weight,
                    )
                    total_cost = prev_cost + pred_energy_j + current_penalty + lambda_time * time_s
                    total_time = prev_time + time_s
                    state_key = (speed, round(over_s, 3))
                    best = state_costs.get(state_key)
                    payload = (
                        total_cost,
                        total_time,
                        pred_current_mA,
                        (float(prev_speed), float(prev_over_s)),
                        action,
                    )
                    if best is None or total_cost < best[0]:
                        state_costs[state_key] = payload
            if not state_costs:
                raise ValueError("No feasible strategy found under the speed/current constraints.")
            dp.append(state_costs)

        final_key, (final_cost, _final_time, _final_current, _prev_key, _final_action) = min(
            dp[-1].items(),
            key=lambda item: item[1][0],
        )
        chosen_keys: list[tuple[float, float]] = [final_key]
        chosen_actions: list[str] = [dp[-1][final_key][4]]
        for idx in range(len(dp) - 1, 0, -1):
            prev_key = dp[idx][chosen_keys[-1]][3]
            if prev_key is None:
                break
            chosen_keys.append(prev_key)
            chosen_actions.append(dp[idx - 1][prev_key][4])
        chosen_keys.reverse()
        chosen_actions.reverse()

        chosen_speeds = [float(key[0]) for key in chosen_keys]
        prev_speeds = [initial_speed] + chosen_speeds[:-1]
        deltas = [speed - prev for prev, speed in zip(prev_speeds, chosen_speeds)]
        actions = [classify_strategy_action(delta, hold_delta_kph=hold_delta_kph) for delta in deltas]
        accels = [
            _signed_accel_from_speed_change(prev_speed, speed, length)
            for prev_speed, speed, length in zip(prev_speeds, chosen_speeds, segments_df["length_m"])
        ]

        out = segments_df.copy().reset_index(drop=True)
        out["target_speed_kph"] = chosen_speeds
        out["entry_speed_kph"] = prev_speeds
        out["speed_delta_kph"] = deltas
        out["action"] = actions
        out["accel_demand_m_s2"] = accels
        out["segment_time_s"] = [
            _segment_transition_time_s(length, prev_speed, speed)
            for length, prev_speed, speed in zip(
                out["length_m"],
                out["entry_speed_kph"],
                out["target_speed_kph"],
            )
        ]
        electrical_rows = [
            predict_strategy_electrical(
                model,
                speed_kph=speed,
                accel_m_s2=accel,
                grade_pct=grade,
                position_frac=pos,
                action=action,
                motor_config=motor_config,
            )
            for speed, accel, grade, pos, action in zip(
                out["target_speed_kph"],
                out["accel_demand_m_s2"],
                out["baseline_grade_pct"],
                out["center_frac"],
                out["action"],
            )
        ]
        out["pred_current_mA"] = [row["avg_current_mA"] for row in electrical_rows]
        out["pred_avg_current_mA"] = out["pred_current_mA"]
        out["pred_power_w"] = [row["avg_power_w"] for row in electrical_rows]
        out["pred_peak_current_mA"] = [row["peak_current_mA"] for row in electrical_rows]
        out["pred_on_current_mA"] = [row["on_current_mA"] for row in electrical_rows]
        out["throttle_duty"] = [row["throttle_duty"] for row in electrical_rows]
        out["throttle_level"] = [row["throttle_level"] for row in electrical_rows]
        out["pulse_duration_s"] = [
            0.0 if action == ACTION_COAST else time_s * duty
            for action, time_s, duty in zip(out["action"], out["segment_time_s"], out["throttle_duty"])
        ]
        out["coast_duration_s"] = [
            time_s if action == ACTION_COAST else (time_s - pulse_s if action == ACTION_HOLD else 0.0)
            for action, time_s, pulse_s in zip(out["action"], out["segment_time_s"], out["pulse_duration_s"])
        ]
        out["coast_speed_drop_kph"] = [
            max(entry - target, 0.0) if coast_s > 0 else 0.0
            for entry, target, coast_s in zip(
                out["entry_speed_kph"],
                out["target_speed_kph"],
                out["coast_duration_s"],
            )
        ]
        out["pred_energy_j"] = out["pred_power_w"] * out["segment_time_s"]
        out["cum_pred_energy_j"] = out["pred_energy_j"].cumsum()
        out["cum_pred_time_s"] = out["segment_time_s"].cumsum()
        out["over_fuse_limit"] = out["pred_peak_current_mA"] > fuse_current_ma
        out["fuse_limit_mA"] = float(fuse_current_ma)
        out["current_penalty"] = [
            _soft_current_penalty(
                current,
                time_s,
                fuse_current_ma=fuse_current_ma,
                weight=current_penalty_weight,
            )
            for current, time_s in zip(out["pred_peak_current_mA"], out["segment_time_s"])
        ]
        out["fuse_over_duration_s"] = [
            _fuse_burst_duration_s(time_s, duty) if over else 0.0
            for time_s, duty, over in zip(out["segment_time_s"], out["throttle_duty"], out["over_fuse_limit"])
        ]
        out.attrs["objective"] = float(final_cost)
        out.attrs["total_time_s"] = float(out["segment_time_s"].sum())
        out.attrs["lambda_time"] = float(lambda_time)
        out.attrs["fuse_current_ma"] = float(fuse_current_ma)
        out.attrs["fuse_max_duration_sec"] = float(fuse_max_duration_sec)
        out.attrs["time_budget_sec"] = float(time_budget_sec)
        out.attrs["min_time_sec"] = float(target_min_time) if target_min_time is not None else None
        out.attrs["motor_config"] = motor_config or {}
        out.attrs["start_speed_kph"] = float(start_speed_kph)
        out.attrs["coast_sample_count"] = int(model.get("coast_sample_count", 0))
        return out

    target_min_time = float(min_time_sec) if min_time_sec is not None else None
    if target_min_time is not None and target_min_time <= 0:
        target_min_time = None

    lambda_grid = [
        -2000.0,
        -1000.0,
        -500.0,
        -250.0,
        -100.0,
        -50.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        -1.0,
        0.0,
    ]
    candidates = []
    for lambda_time in lambda_grid:
        try:
            candidates.append(solve_for_lambda(lambda_time))
        except ValueError:
            continue
    feasible = [
        candidate
        for candidate in candidates
        if candidate.attrs["total_time_s"] <= time_budget_sec
        and (target_min_time is None or candidate.attrs["total_time_s"] >= target_min_time)
    ]
    if feasible:
        return min(feasible, key=lambda item: float(item["pred_energy_j"].sum()))

    low = solve_for_lambda(0.0)
    if low.attrs["total_time_s"] <= time_budget_sec and target_min_time is None:
        return low

    hi_lambda = 1.0
    hi = solve_for_lambda(hi_lambda)
    while hi.attrs["total_time_s"] > time_budget_sec and hi_lambda < 1e6:
        hi_lambda *= 2.0
        hi = solve_for_lambda(hi_lambda)

    best = hi
    lo_lambda = 0.0
    for _ in range(20):
        mid_lambda = (lo_lambda + hi_lambda) / 2.0
        mid = solve_for_lambda(mid_lambda)
        if mid.attrs["total_time_s"] > time_budget_sec:
            lo_lambda = mid_lambda
        else:
            hi_lambda = mid_lambda
            best = mid
    if target_min_time is not None and best.attrs["total_time_s"] < target_min_time:
        under_budget = [candidate for candidate in candidates if candidate.attrs["total_time_s"] <= time_budget_sec]
        if under_budget:
            return min(
                under_budget,
                key=lambda item: abs(item.attrs["total_time_s"] - target_min_time),
            )
    return best


def build_strategy_samples(df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    df = build_full_run_distance(df)
    samples = df.copy().reset_index(drop=True)
    segment_edges = profile_df["dist_end_m"].to_numpy(dtype=float)
    segment_index = np.searchsorted(
        segment_edges,
        samples["run_cumdist_m"].to_numpy(dtype=float),
        side="left",
    )
    segment_index = np.clip(segment_index, 0, len(profile_df) - 1)
    mapped = profile_df.iloc[segment_index].reset_index(drop=True)
    samples["segment"] = mapped["segment"]
    samples["target_speed_kph"] = mapped["target_speed_kph"]
    samples["strategy_action"] = mapped["action"]
    samples["pred_current_mA"] = mapped["pred_current_mA"]
    samples["pred_avg_current_mA"] = mapped.get("pred_avg_current_mA", mapped["pred_current_mA"])
    samples["pred_peak_current_mA"] = mapped.get("pred_peak_current_mA", mapped["pred_current_mA"])
    samples["pred_on_current_mA"] = mapped.get("pred_on_current_mA", mapped["pred_current_mA"])
    samples["throttle_duty"] = mapped.get("throttle_duty", 0.0)
    samples["throttle_level"] = mapped.get("throttle_level", samples["throttle_duty"])
    samples["pred_power_w"] = mapped["pred_power_w"]
    samples["pulse_duration_s"] = mapped.get("pulse_duration_s", 0.0)
    samples["coast_duration_s"] = mapped.get("coast_duration_s", 0.0)
    samples["coast_speed_drop_kph"] = mapped.get("coast_speed_drop_kph", 0.0)
    samples["pred_energy_j"] = pd.to_numeric(samples["dt_s"], errors="coerce").fillna(0.0) * mapped["pred_power_w"]
    samples["pred_cum_energy_j"] = samples["pred_energy_j"].cumsum()
    samples["pred_over_fuse_limit"] = mapped["over_fuse_limit"]
    return samples


def evaluate_baseline_prediction(
    segments_df: pd.DataFrame,
    model: dict[str, object],
    motor_config: dict[str, float | str] | None = None,
    hold_delta_kph: float = 1.0,
    start_speed_kph: float | None = None,
) -> dict[str, float]:
    if start_speed_kph is None:
        prev_speed = float(segments_df["baseline_speed_kph"].iloc[0])
    else:
        prev_speed = float(start_speed_kph)
    pred_energy_j = 0.0
    pred_power = []
    actual_power = []
    pred_current = []
    actual_current = []
    for row in segments_df.itertuples(index=False):
        speed = float(row.baseline_speed_kph)
        accel = _signed_accel_from_speed_change(prev_speed, speed, float(row.length_m))
        action = classify_strategy_action(speed - prev_speed, hold_delta_kph=hold_delta_kph)
        electrical = predict_strategy_electrical(
            model,
            speed_kph=speed,
            accel_m_s2=accel,
            grade_pct=float(row.baseline_grade_pct),
            position_frac=float(row.center_frac),
            action=action,
            motor_config=motor_config,
        )
        pred_energy_j += electrical["avg_power_w"] * float(row.baseline_time_s)
        pred_power.append(electrical["avg_power_w"])
        actual_power.append(float(row.baseline_power_w))
        pred_current.append(electrical["avg_current_mA"])
        actual_current.append(float(row.baseline_current_mA))
        prev_speed = speed
    actual_energy_j = float(pd.to_numeric(segments_df["baseline_energy_j"], errors="coerce").fillna(0.0).sum())
    return {
        "actual_energy_j": actual_energy_j,
        "pred_energy_j": float(pred_energy_j),
        "energy_error_pct": _pct_error(pred_energy_j, actual_energy_j),
        "power_mae_w": _mae(pred_power, actual_power),
        "current_mae_mA": _mae(pred_current, actual_current),
    }


def build_strategy_report(
    baseline_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    time_budget_sec: float,
    calibration: dict[str, float] | None = None,
) -> str:
    baseline_df = build_full_run_distance(baseline_df)
    profile_df = profile_df.copy()
    if "pred_peak_current_mA" not in profile_df.columns:
        profile_df["pred_peak_current_mA"] = profile_df["pred_current_mA"]
    if "throttle_duty" not in profile_df.columns:
        profile_df["throttle_duty"] = 0.0
    if "throttle_level" not in profile_df.columns:
        profile_df["throttle_level"] = profile_df["throttle_duty"]
    if "pulse_duration_s" not in profile_df.columns:
        profile_df["pulse_duration_s"] = 0.0
    if "coast_duration_s" not in profile_df.columns:
        profile_df["coast_duration_s"] = 0.0
    baseline_energy_j = float(pd.to_numeric(baseline_df["energy_j"], errors="coerce").fillna(0.0).sum())
    baseline_time_s = float(pd.to_numeric(baseline_df["dt_s"], errors="coerce").fillna(0.0).sum())
    pred_energy_j = float(profile_df["pred_energy_j"].sum())
    pred_time_s = float(profile_df["segment_time_s"].sum())
    delta_j = pred_energy_j - baseline_energy_j
    delta_pct = (delta_j / baseline_energy_j * 100.0) if baseline_energy_j > 0 else 0.0

    fuse_current_ma = float(profile_df.attrs.get("fuse_current_ma", 20000.0))
    motor_config = profile_df.attrs.get("motor_config", {}) or {}
    start_speed_kph = float(profile_df.attrs.get("start_speed_kph", 0.0))
    total_over_fuse_s = float(profile_df["fuse_over_duration_s"].sum())
    longest_over_fuse_s = _longest_true_duration(
        profile_df["over_fuse_limit"].tolist(),
        profile_df["segment_time_s"].tolist(),
    )
    near_fuse = profile_df[profile_df["pred_peak_current_mA"] >= 0.9 * fuse_current_ma]
    total_coast_time = float(pd.to_numeric(profile_df["coast_duration_s"], errors="coerce").fillna(0.0).sum())
    total_coast_dist = float(
        profile_df.loc[profile_df["coast_duration_s"] > 0, "length_m"].sum()
    )
    pulse_count = int((pd.to_numeric(profile_df["pulse_duration_s"], errors="coerce").fillna(0.0) > 0).sum())
    avg_pulse_duration = float(
        pd.to_numeric(profile_df.loc[profile_df["pulse_duration_s"] > 0, "pulse_duration_s"], errors="coerce")
        .dropna()
        .mean()
    )
    if not math.isfinite(avg_pulse_duration):
        avg_pulse_duration = 0.0

    lines = [
        "=== Speed Strategy Report ===",
        "",
        f"Time budget: {time_budget_sec:.1f}s",
        f"Baseline run: {baseline_time_s:.1f}s, {baseline_energy_j:.1f} J",
        f"Predicted strategy: {pred_time_s:.1f}s, {pred_energy_j:.1f} J",
        f"Predicted delta: {delta_j:+.1f} J ({delta_pct:+.2f}%)",
        f"Fuse threshold: {fuse_current_ma/1000.0:.1f} A",
        f"Predicted time above fuse threshold: {total_over_fuse_s:.2f}s",
        f"Longest predicted over-fuse burst: {longest_over_fuse_s:.2f}s",
        f"Motor profile: {motor_config.get('name', 'not configured')}",
        f"Simulated start speed: {start_speed_kph:.1f} km/h",
        f"Top speed cap: {float(motor_config.get('top_speed_kph', 0.0)):.1f} km/h",
        f"Assumed wheel diameter: {float(motor_config.get('wheel_diameter_m', 0.0)):.2f} m",
        f"Inferred gear ratio: {float(motor_config.get('inferred_gear_ratio', 0.0)):.2f}:1",
        f"Coast policy: 0 mA propulsion current",
        f"Coast model samples: {int(profile_df.attrs.get('coast_sample_count', 0))}",
        f"Pulse/coast summary: {pulse_count} pulse segment(s), {avg_pulse_duration:.2f}s average pulse",
        f"Predicted no-throttle coast: {total_coast_time:.1f}s over {total_coast_dist:.1f} m",
    ]
    if calibration:
        lines.extend([
            f"Baseline backtest predicted energy: {calibration['pred_energy_j']:.1f} J",
            f"Baseline backtest energy error: {calibration['energy_error_pct']:+.2f}%",
            f"Baseline backtest power MAE: {calibration['power_mae_w']:.1f} W",
            f"Baseline backtest current MAE: {calibration['current_mae_mA']:.0f} mA",
        ])
    elev_diff = pd.to_numeric(baseline_df.get("elev_diff_m", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    lines.extend([
        f"Elevation gain/loss: {elev_diff.clip(lower=0).sum():.1f} m gain, {abs(elev_diff.clip(upper=0).sum()):.1f} m loss",
        "",
    ])
    if not near_fuse.empty:
        lines.append(
            "Warning: best feasible strategy runs near the fuse threshold "
            f"in {len(near_fuse)} segment(s)."
        )
        lines.append("")

    for action, heading in (
        (ACTION_ACCELERATE, "Top accelerate regions:"),
        (ACTION_HOLD, "Top hold regions:"),
        (ACTION_COAST, "Top coast regions:"),
    ):
        lines.append(heading)
        subset = profile_df[profile_df["action"] == action]
        if subset.empty:
            lines.append("None")
        else:
            if action == ACTION_HOLD:
                order = subset.sort_values("pred_energy_j", ascending=False).head(3)
            else:
                order = subset.sort_values("speed_delta_kph", ascending=(action == ACTION_COAST)).head(3)
            for row in order.itertuples(index=False):
                lines.append(
                    f"Segment {int(row.segment)} -> {row.action} at {row.target_speed_kph:.1f} km/h, "
                    f"avg current {row.pred_current_mA:.0f} mA, peak current {row.pred_peak_current_mA:.0f} mA, "
                    f"throttle {row.throttle_level:.2f}, pred energy {row.pred_energy_j:.1f} J"
                )
        lines.append("")
    return "\n".join(lines).strip()


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    speed = df["speed_kph"].to_numpy(dtype=float)
    pos_accel = df["pos_accel_m_s2"].to_numpy(dtype=float)
    neg_accel = df["neg_accel_m_s2"].to_numpy(dtype=float)
    uphill = df["uphill_grade_pct"].to_numpy(dtype=float)
    downhill = df["downhill_grade_pct"].to_numpy(dtype=float)
    position = df["position_frac"].to_numpy(dtype=float)
    is_accel = df["is_accelerate"].to_numpy(dtype=float)
    is_hold = df["is_hold"].to_numpy(dtype=float)
    is_coast = df["is_coast"].to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(df)),
        speed,
        speed ** 2,
        pos_accel,
        neg_accel,
        uphill,
        downhill,
        position,
        is_accel,
        is_hold,
        is_coast,
        pos_accel * speed,
        neg_accel * speed,
    ])


def _coast_design_matrix(df: pd.DataFrame) -> np.ndarray:
    speed = df["speed_kph"].to_numpy(dtype=float)
    grade = df["grade_pct"].to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(df)),
        speed,
        speed ** 2,
        grade,
    ])


def _solve_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _predict_linear(
    coeffs: object,
    speed_kph: float,
    accel_m_s2: float,
    grade_pct: float,
    position_frac: float,
    action: str,
) -> float:
    coeff_array = np.asarray(coeffs, dtype=float)
    pos_accel = max(accel_m_s2, 0.0)
    neg_accel = max(-accel_m_s2, 0.0)
    uphill = max(grade_pct, 0.0)
    downhill = max(-grade_pct, 0.0)
    is_accel = 1.0 if action == ACTION_ACCELERATE else 0.0
    is_hold = 1.0 if action == ACTION_HOLD else 0.0
    is_coast = 1.0 if action == ACTION_COAST else 0.0
    features = np.array([
        1.0,
        speed_kph,
        speed_kph ** 2,
        pos_accel,
        neg_accel,
        uphill,
        downhill,
        position_frac,
        is_accel,
        is_hold,
        is_coast,
        pos_accel * speed_kph,
        neg_accel * speed_kph,
    ])
    return float(features @ coeff_array)


def _segment_time_s(length_m: float, speed_kph: float) -> float:
    speed_m_s = max(speed_kph / 3.6, 0.1)
    return float(length_m / speed_m_s)


def _segment_transition_time_s(length_m: float, prev_speed_kph: float, speed_kph: float) -> float:
    v0 = max(prev_speed_kph / 3.6, 0.0)
    v1 = max(speed_kph / 3.6, 0.0)
    avg_speed_m_s = max((v0 + v1) * 0.5, 0.1)
    return float(length_m / avg_speed_m_s)


def _predict_coast_accel_m_s2(model: dict[str, object], speed_kph: float, grade_pct: float) -> float:
    coeffs = np.asarray(model.get("coast_coeffs", np.array([-0.25, 0.0, 0.0, -0.098])), dtype=float)
    features = np.array([1.0, speed_kph, speed_kph ** 2, grade_pct])
    accel = min(float(features @ coeffs), -0.06)
    return float(np.clip(accel, -4.0, -0.06))


def _coast_exit_speed_kph(
    prev_speed_kph: float,
    length_m: float,
    grade_pct: float,
    model: dict[str, object],
) -> float:
    v0 = max(prev_speed_kph / 3.6, 0.0)
    if length_m <= 0 or v0 <= 0:
        return 0.0
    accel = _predict_coast_accel_m_s2(model, prev_speed_kph, grade_pct)
    v1_sq = max(v0 * v0 + 2.0 * accel * length_m, 0.0)
    return float(math.sqrt(v1_sq) * 3.6)


def _signed_accel_from_speed_change(prev_speed_kph: float, speed_kph: float, length_m: float) -> float:
    if length_m <= 0:
        return 0.0
    v0 = max(prev_speed_kph / 3.6, 0.0)
    v1 = max(speed_kph / 3.6, 0.0)
    return float((v1 * v1 - v0 * v0) / (2.0 * length_m))


def _soft_current_penalty(
    pred_current_mA: float,
    time_s: float,
    fuse_current_ma: float,
    weight: float,
) -> float:
    soft_start = 0.75 * fuse_current_ma
    over = max(pred_current_mA - soft_start, 0.0)
    if over <= 0:
        return 0.0
    ratio = over / max(fuse_current_ma - soft_start, 1.0)
    return float(weight * (ratio ** 2) * time_s * 100.0)


def _physics_propulsion_power_w(
    speed_kph: float,
    accel_m_s2: float,
    grade_pct: float,
    motor_config: dict[str, float | str] | None,
) -> float:
    if motor_config is None:
        return 0.0
    speed_m_s = max(speed_kph / 3.6, 0.0)
    mass_kg = float(motor_config.get("vehicle_mass_kg", 100.0))
    crr = float(motor_config.get("rolling_resistance_coeff", 0.008))
    efficiency = max(float(motor_config.get("drivetrain_efficiency", 0.82)), 0.05)
    grade_force = mass_kg * 9.80665 * (grade_pct / 100.0)
    accel_force = mass_kg * max(accel_m_s2, 0.0)
    rolling_force = mass_kg * 9.80665 * crr
    propulsive_force = max(accel_force + grade_force + rolling_force, 0.0)
    return float(propulsive_force * speed_m_s / efficiency)


def _fuse_burst_duration_s(time_s: float, throttle_duty: float) -> float:
    # On/off driving is modeled as short repeated throttle pulses. The fuse
    # cares about the longest continuous high-current burst, not total segment
    # time spent with some duty cycle.
    return float(min(max(time_s * throttle_duty, 0.0), 0.8))


def _mae(predicted: list[float], actual: list[float]) -> float:
    if not predicted:
        return 0.0
    return float(np.mean(np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float))))


def _pct_error(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0
    return float((predicted - actual) / actual * 100.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _longest_true_duration(flags: list[bool], durations: list[float]) -> float:
    best = 0.0
    run = 0.0
    for flag, duration in zip(flags, durations):
        if flag:
            run += float(duration)
            best = max(best, run)
        else:
            run = 0.0
    return best
