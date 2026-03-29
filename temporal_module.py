import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
import warnings


# ── JICA 2018 baseline data ────────────────────────────────────────────────────
# Source: JICA Data Collection Survey, Harare, April 2018
# These are the anchor points for regression until HCC logs are integrated

JICA_BASELINE = pd.DataFrame({
    "year": [2012, 2014, 2020, 2030],
    "flow_m3_day": [82258, 182354, 191270, 242468],
})


def load_jica_baseline() -> pd.DataFrame:
    """
    Return JICA flow baseline as a clean DataFrame with flow in m3/s.
    Used as training data until HCC real-time logs are available.
    """
    df = JICA_BASELINE.copy()
    df["flow_m3s"] = df["flow_m3_day"] / 86400
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df = df.set_index("date").sort_index()
    return df


def load_hcc_logs(csv_path: str) -> pd.DataFrame:
    """
    Load HCC flow logs once available.
    Expected CSV columns: timestamp, flow_m3s
    timestamp format: YYYY-MM-DD HH:MM:SS
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    assert "flow_m3s" in df.columns, "CSV must contain a 'flow_m3s' column"
    return df


# ── 1. Descriptive statistics ──────────────────────────────────────────────────

def characterise_flow(df: pd.DataFrame, q_col: str = "flow_m3s") -> dict:
    """
    Compute full descriptive statistics on Q(t).
    CV (coefficient of variation) quantifies how much flow
    fluctuates relative to the mean — directly sets variance in P(t).
    """
    q = df[q_col].dropna()
    return {
        "mean_m3s":   round(float(q.mean()), 4),
        "median_m3s": round(float(q.median()), 4),
        "std":        round(float(q.std()), 4),
        "variance":   round(float(q.var()), 6),
        "cv":         round(float(q.std() / q.mean()), 4),
        "skewness":   round(float(stats.skew(q)), 4),
        "kurtosis":   round(float(stats.kurtosis(q)), 4),
        "q10_m3s":    round(float(q.quantile(0.10)), 4),
        "q50_m3s":    round(float(q.quantile(0.50)), 4),
        "q90_m3s":    round(float(q.quantile(0.90)), 4),
        "iqr":        round(float(q.quantile(0.75) - q.quantile(0.25)), 4),
        "n_records":  int(len(q)),
    }


# ── 2. Flow Duration Curve ─────────────────────────────────────────────────────

def flow_duration_curve(df: pd.DataFrame,
                         q_col: str = "flow_m3s") -> pd.DataFrame:
    """
    Compute the Flow Duration Curve (FDC).
    Shows percentage of time flow equals or exceeds a given value.
    Standard hydrology tool — required thesis figure.
    """
    q = df[q_col].dropna()
    sorted_q = np.sort(q.values)[::-1]
    exceedance = np.arange(1, len(sorted_q) + 1) / len(sorted_q) * 100
    return pd.DataFrame({
        "exceedance_pct": exceedance,
        "flow_m3s":       sorted_q,
    })


# ── 3. STL decomposition ───────────────────────────────────────────────────────

def decompose_flow(df: pd.DataFrame,
                   q_col: str = "flow_m3s",
                   period: int = 12) -> dict:
    """
    Decompose Q(t) into trend, seasonal, and residual components using STL.

    period=12  for monthly data (annual cycle)
    period=24  for hourly data  (diurnal cycle)
    period=7   for daily data   (weekly pattern)

    Returns components + strength metrics (0–1 scale).
    Strength > 0.6 means the component explains most of the variance.
    """
    series = df[q_col].dropna()

    if len(series) < period * 2:
        warnings.warn(
            f"Only {len(series)} observations — need at least {period*2} "
            f"for reliable STL decomposition. Using JICA baseline projections."
        )
        return _synthetic_decomposition(series)

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    trend_strength = max(0, 1 - (
        result.resid.var() /
        (result.trend + result.resid).var()
    ))
    seasonal_strength = max(0, 1 - (
        result.resid.var() /
        (result.seasonal + result.resid).var()
    ))

    return {
        "trend":             pd.Series(result.trend,    index=series.index),
        "seasonal":          pd.Series(result.seasonal, index=series.index),
        "residual":          pd.Series(result.resid,    index=series.index),
        "trend_strength":    round(trend_strength, 4),
        "seasonal_strength": round(seasonal_strength, 4),
    }


def _synthetic_decomposition(series: pd.Series) -> dict:
    """
    Fallback when data is too sparse for STL.
    Fits a linear trend and returns zero seasonal component.
    Used with JICA baseline (4 data points).
    """
    x = np.arange(len(series)).reshape(-1, 1)
    model = LinearRegression().fit(x, series.values)
    trend = pd.Series(model.predict(x), index=series.index)
    residual = series - trend
    return {
        "trend":             trend,
        "seasonal":          pd.Series(np.zeros(len(series)), index=series.index),
        "residual":          residual,
        "trend_strength":    1.0,
        "seasonal_strength": 0.0,
    }


# ── 4. Regression forecast ─────────────────────────────────────────────────────

def forecast_flow(df: pd.DataFrame,
                  q_col: str = "flow_m3s",
                  forecast_year_end: int = 2035) -> dict:
    """
    Fit OLS linear regression to Q(t) and forecast to forecast_year_end.
    Returns model, R², slope, and forecast DataFrame.

    This captures the population-growth-driven trend component.
    Seasonal component is added back from STL for full Q(t) reconstruction.
    """
    df_clean = df[[q_col]].dropna().copy()
    df_clean["t"] = np.arange(len(df_clean))

    X = df_clean[["t"]].values
    y = df_clean[q_col].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    slope = model.coef_[0]

    # Build future index
    last_date = df_clean.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        end=pd.Timestamp(f"{forecast_year_end}-12-31"),
        freq="YS"
    )
    future_t = np.arange(
        len(df_clean),
        len(df_clean) + len(future_dates)
    ).reshape(-1, 1)
    future_q = model.predict(future_t)

    forecast_df = pd.DataFrame({
        "date":       future_dates,
        "flow_m3s":   future_q,
        "lower_ci":   future_q - 1.96 * df_clean[q_col].std(),
        "upper_ci":   future_q + 1.96 * df_clean[q_col].std(),
    }).set_index("date")

    return {
        "model":              model,
        "r_squared":          round(r2, 4),
        "slope_m3s_per_step": round(float(slope), 6),
        "annual_growth_m3s":  round(float(slope * 365), 4),
        "forecast_df":        forecast_df,
    }


# ── 5. Correlation analysis ────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame,
                          q_col: str = "flow_m3s") -> dict:
    """
    Compute Pearson and Spearman correlations between flow and time.
    Pearson R measures linear relationship.
    Spearman rho measures monotonic relationship (robust to outliers).
    Both reported — PhD thesis requires both.
    """
    q = df[q_col].dropna()
    t = np.arange(len(q))

    pearson_r,  pearson_p  = stats.pearsonr(t, q)
    spearman_r, spearman_p = stats.spearmanr(t, q)

    return {
        "pearson_r":   round(float(pearson_r), 4),
        "pearson_p":   round(float(pearson_p), 6),
        "spearman_r":  round(float(spearman_r), 4),
        "spearman_p":  round(float(spearman_p), 6),
        "significant": pearson_p < 0.05,
    }


# ── 6. Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_jica_baseline()
    print("JICA baseline loaded:")
    print(df[["flow_m3_day", "flow_m3s"]])
    print()

    stats_out = characterise_flow(df)
    print("Flow statistics:")
    for k, v in stats_out.items():
        print(f"  {k}: {v}")
    print()

    forecast = forecast_flow(df)
    print(f"Regression R²: {forecast['r_squared']}")
    print(f"Annual growth: {forecast['annual_growth_m3s']} m³/s per year")
    print()
    print("Forecast to 2035:")
    print(forecast["forecast_df"])
