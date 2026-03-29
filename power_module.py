import pandas as pd
import numpy as np
from scipy import stats


# ── Constants ──────────────────────────────────────────────────────────────────
RHO   = 1000.0   # kg/m³  — density of wastewater (approximated as water)
G     = 9.81     # m/s²   — gravitational acceleration
ETA   = 0.85     # dimensionless — turbine efficiency (conservative estimate)
TARIFF_USD_KWH = 0.11  # USD/kWh — Harare electricity tariff, JICA 2018


# ── 1. Single point power calculation ─────────────────────────────────────────

def compute_power(Q: float, H: float,
                  eta: float = ETA) -> float:
    """
    Compute hydraulic power at a single point.
    P = eta * rho * g * Q * H / 1000

    Parameters
    ----------
    Q   : flow rate (m³/s)
    H   : hydraulic head (m)
    eta : turbine efficiency (0–1)

    Returns
    -------
    Power in kilowatts (kW)
    """
    return (eta * RHO * G * Q * H) / 1000.0


# ── 2. Power surface P(x, t) ───────────────────────────────────────────────────

def compute_power_surface(head_profile: pd.DataFrame,
                           flow_series: pd.DataFrame,
                           eta: float = ETA) -> pd.DataFrame:
    """
    Compute P(x, t) — power at every spatial point for every time step.

    Parameters
    ----------
    head_profile : DataFrame with columns distance_m, head_m
                   (output of spatial_module)
    flow_series  : DataFrame with column flow_m3s, DatetimeIndex
                   (output of temporal_module)

    Returns
    -------
    DataFrame with columns:
        date, distance_m, head_m, flow_m3s, power_kw, energy_kwh_day
    """
    records = []
    for _, spatial_row in head_profile.iterrows():
        H = spatial_row["head_m"]
        if H <= 0:
            continue
        for ts, temporal_row in flow_series.iterrows():
            Q = temporal_row["flow_m3s"]
            P = compute_power(Q, H, eta)
            records.append({
                "date":          ts,
                "distance_m":    round(spatial_row["distance_m"], 2),
                "head_m":        round(H, 2),
                "flow_m3s":      round(Q, 6),
                "power_kw":      round(P, 4),
                "energy_kwh_day": round(P * 24, 4),
            })

    return pd.DataFrame(records)


# ── 3. Summary energy metrics ──────────────────────────────────────────────────

def compute_energy_summary(power_df: pd.DataFrame,
                            tariff: float = TARIFF_USD_KWH) -> dict:
    """
    Compute annual energy yield and economic value from the power surface.

    Parameters
    ----------
    power_df : output of compute_power_surface
    tariff   : electricity tariff in USD/kWh

    Returns
    -------
    Dictionary of summary metrics for Streamlit dashboard display
    """
    P = power_df["power_kw"]
    return {
        "P_mean_kw":          round(float(P.mean()), 2),
        "P_max_kw":           round(float(P.max()), 2),
        "P_min_kw":           round(float(P.min()), 2),
        "P_std_kw":           round(float(P.std()), 2),
        "annual_energy_mwh":  round(float(P.mean() * 8760 / 1000), 2),
        "annual_value_usd":   round(float(P.mean() * 8760 * tariff), 2),
        "firle_elec_budget":  900000,
        "offset_pct":         round(
            float(P.mean() * 8760 * tariff) / 900000 * 100, 1
        ),
    }


# ── 4. Monte Carlo uncertainty analysis ───────────────────────────────────────

def monte_carlo_uncertainty(Q_mean: float, Q_std: float,
                             H_mean: float, H_std: float,
                             n_samples: int = 10000,
                             eta: float = ETA) -> dict:
    """
    Propagate uncertainty in Q and H through the power equation
    using Monte Carlo simulation.

    Samples Q and H from normal distributions, computes P for each,
    returns statistics on the resulting P distribution.

    This quantifies how sensitive your power estimate is to
    measurement uncertainty in flow and head — required for PhD rigour.
    """
    rng = np.random.default_rng(seed=42)

    Q_samples = rng.normal(Q_mean, Q_std, n_samples)
    H_samples = rng.normal(H_mean, H_std, n_samples)

    # Clip negatives — physical constraint
    Q_samples = np.clip(Q_samples, 0, None)
    H_samples = np.clip(H_samples, 0, None)

    P_samples = (eta * RHO * G * Q_samples * H_samples) / 1000.0

    ci_low, ci_high = np.percentile(P_samples, [2.5, 97.5])

    return {
        "P_mean_kw":    round(float(P_samples.mean()), 2),
        "P_std_kw":     round(float(P_samples.std()), 2),
        "P_ci_low_kw":  round(float(ci_low), 2),
        "P_ci_high_kw": round(float(ci_high), 2),
        "n_samples":    n_samples,
    }


# ── 5. Sensitivity analysis ────────────────────────────────────────────────────

def sensitivity_analysis(Q_base: float, H_base: float,
                          eta_base: float = ETA) -> pd.DataFrame:
    """
    Vary each parameter ±10%, ±20%, ±30% while holding others constant.
    Shows which parameter most influences P — key thesis result.
    """
    P_base = compute_power(Q_base, H_base, eta_base)
    records = []

    for param, base_val in [("Q", Q_base), ("H", H_base), ("eta", eta_base)]:
        for pct in [-30, -20, -10, 0, 10, 20, 30]:
            varied = base_val * (1 + pct / 100)
            if param == "Q":
                P = compute_power(varied, H_base, eta_base)
            elif param == "H":
                P = compute_power(Q_base, varied, eta_base)
            else:
                P = compute_power(Q_base, H_base, varied)

            records.append({
                "parameter":  param,
                "change_pct": pct,
                "value":      round(varied, 4),
                "power_kw":   round(P, 4),
                "delta_pct":  round((P - P_base) / P_base * 100, 2),
            })

    return pd.DataFrame(records)


# ── 6. Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use JICA baseline values directly
    Q = 0.952   # m³/s — actual effluent 2012, JICA Table 3.4.7
    H = 97.0    # m    — Harare plateau minus Lake Chivero

    P = compute_power(Q, H)
    print(f"Single point power:  {P:.2f} kW")
    print(f"Annual energy:       {P * 8760 / 1000:.2f} MWh/year")
    print(f"Annual value:        ${P * 8760 * TARIFF_USD_KWH:,.2f} USD/year")
    print()

    # Monte Carlo
    mc = monte_carlo_uncertainty(
        Q_mean=0.952, Q_std=0.05,
        H_mean=97.0,  H_std=5.0
    )
    print("Monte Carlo uncertainty (10,000 samples):")
    for k, v in mc.items():
        print(f"  {k}: {v}")
    print()

    # Sensitivity
    sa = sensitivity_analysis(Q, H)
    print("Sensitivity analysis:")
    print(sa.to_string(index=False))