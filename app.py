import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from temporal_module import (
    load_jica_baseline,
    characterise_flow,
    flow_duration_curve,
    decompose_flow,
    forecast_flow,
    correlation_analysis,
)
from power_module import (
    compute_power,
    compute_energy_summary,
    monte_carlo_uncertainty,
    sensitivity_analysis,
    TARIFF_USD_KWH,
    ETA,
)

st.set_page_config(
    page_title="Firle WWTP — Hydroelectric Potential Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Spatio-Temporal Hydroelectric Potential Analysis")
st.caption("Firle Wastewater Treatment Works, Harare, Zimbabwe — Topic 1 Baseline")

st.sidebar.header("Model Parameters")
eta_input    = st.sidebar.slider("Turbine efficiency η", 0.60, 0.92, ETA, 0.01)
H_input      = st.sidebar.slider("Hydraulic head H (m)", 60.0, 150.0, 93.3, 1.0)
tariff_input = st.sidebar.slider("Electricity tariff (USD/kWh)", 0.08, 0.20, TARIFF_USD_KWH, 0.01)
st.sidebar.markdown("---")
st.sidebar.caption("JICA 2018 baseline. HCC flow logs integrated when available.")

@st.cache_data
def load_all():
    df       = load_jica_baseline()
    stats    = characterise_flow(df)
    fdc      = flow_duration_curve(df)
    decomp   = decompose_flow(df)
    forecast = forecast_flow(df)
    corr     = correlation_analysis(df)
    return df, stats, fdc, decomp, forecast, corr

df, flow_stats, fdc, decomp, forecast, corr = load_all()

tab1, tab2, tab3 = st.tabs([
    "Temporal Analysis — Q(t)",
    "Power Integration — P(x,t)",
    "Uncertainty & Sensitivity",
])

# ── TAB 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Flow Rate Characterisation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean flow",    f"{flow_stats['mean_m3s']} m³/s")
    c2.metric("Std dev",      f"{flow_stats['std']} m³/s")
    c3.metric("CV",           f"{flow_stats['cv']}")
    c4.metric("Skewness",     f"{flow_stats['skewness']}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Q10",          f"{flow_stats['q10_m3s']} m³/s")
    c6.metric("Q50 (median)", f"{flow_stats['q50_m3s']} m³/s")
    c7.metric("Q90",          f"{flow_stats['q90_m3s']} m³/s")
    c8.metric("Records",      str(flow_stats['n_records']))

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Hydrograph — Q(t) with trend")
        fig_hydro = go.Figure()
        fig_hydro.add_trace(go.Scatter(
            x=df.index, y=df["flow_m3s"],
            mode="lines+markers", name="Observed Q",
            line=dict(color="#378ADD", width=2), marker=dict(size=8),
        ))
        fig_hydro.add_trace(go.Scatter(
            x=decomp["trend"].index, y=decomp["trend"].values,
            mode="lines", name="Trend",
            line=dict(color="#D85A30", width=2, dash="dash"),
        ))
        fig_hydro.update_layout(
            xaxis_title="Date", yaxis_title="Flow rate (m³/s)",
            template="simple_white", height=320,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_hydro, use_container_width=True)

    with col_right:
        st.markdown("#### Regression forecast — Q(t) to 2035")
        fc_df = forecast["forecast_df"]
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df.index, y=df["flow_m3s"],
            mode="markers", name="Observed",
            marker=dict(color="#378ADD", size=10),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_df.index, y=fc_df["flow_m3s"],
            mode="lines", name="Forecast",
            line=dict(color="#1D9E75", width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_df.index.to_series(),
                         fc_df.index.to_series()[::-1]]),
            y=pd.concat([fc_df["upper_ci"], fc_df["lower_ci"][::-1]]),
            fill="toself", fillcolor="rgba(29,158,117,0.12)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI",
        ))
        fig_fc.add_annotation(
            x=0.05, y=0.92, xref="paper", yref="paper",
            text=f"R² = {forecast['r_squared']}",
            showarrow=False, font=dict(size=13),
            bgcolor="rgba(255,255,255,0.7)",
        )
        fig_fc.update_layout(
            xaxis_title="Year", yaxis_title="Flow rate (m³/s)",
            template="simple_white", height=320,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")
    col_fdc, col_corr = st.columns(2)

    with col_fdc:
        st.markdown("#### Flow Duration Curve")
        fig_fdc = go.Figure()
        fig_fdc.add_trace(go.Scatter(
            x=fdc["exceedance_pct"], y=fdc["flow_m3s"],
            mode="lines", line=dict(color="#534AB7", width=2),
            fill="tozeroy", fillcolor="rgba(83,74,183,0.08)",
        ))
        fig_fdc.update_layout(
            xaxis_title="Exceedance probability (%)",
            yaxis_title="Flow rate (m³/s)",
            template="simple_white", height=300,
        )
        st.plotly_chart(fig_fdc, use_container_width=True)

    with col_corr:
        st.markdown("#### Correlation analysis")
        st.markdown(f"**Pearson R:** {corr['pearson_r']} (p = {corr['pearson_p']})")
        st.markdown(f"**Spearman ρ:** {corr['spearman_r']} (p = {corr['spearman_p']})")
        sig = "statistically significant" if corr["significant"] else "not significant"
        st.markdown(f"**Result:** Flow trend is {sig} at α = 0.05")
        st.markdown("---")
        st.markdown("**STL decomposition strength**")
        s1, s2 = st.columns(2)
        s1.metric("Trend strength",    str(decomp["trend_strength"]))
        s2.metric("Seasonal strength", str(decomp["seasonal_strength"]))

# ── TAB 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Hydroelectric Power Potential — P = ηρgQ(t)H(s) / 1000")

    Q_vals = df["flow_m3s"].values
    P_vals = np.array([compute_power(Q, H_input, eta_input) for Q in Q_vals])

    P_mean  = float(P_vals.mean())
    ann_mwh = round(P_mean * 8760 / 1000, 2)
    ann_usd = round(P_mean * 8760 * tariff_input, 2)
    offset  = round(ann_usd / 900000 * 100, 1)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean power",        f"{P_mean:.1f} kW")
    k2.metric("Max power",         f"{float(P_vals.max()):.1f} kW")
    k3.metric("Annual energy",     f"{ann_mwh} MWh")
    k4.metric("Annual value",      f"${ann_usd:,.0f}")
    k5.metric("Electricity offset",f"{offset}%", delta=f"of $900,000 budget")

    st.markdown("---")
    col_p, col_h = st.columns(2)

    with col_p:
        st.markdown("#### Power output across flow scenarios")
        Q_range = np.linspace(0.5, 3.0, 100)
        P_range = [compute_power(Q, H_input, eta_input) for Q in Q_range]
        fig_pq = go.Figure()
        fig_pq.add_trace(go.Scatter(
            x=Q_range, y=P_range, mode="lines",
            line=dict(color="#D85A30", width=2),
        ))
        for label, val, col in [
            ("Q10", flow_stats["q10_m3s"], "#888780"),
            ("Q50", flow_stats["q50_m3s"], "#378ADD"),
            ("Q90", flow_stats["q90_m3s"], "#1D9E75"),
        ]:
            fig_pq.add_vline(x=val, line_dash="dot", line_color=col,
                             annotation_text=label, annotation_position="top")
        fig_pq.update_layout(
            xaxis_title="Flow rate Q (m³/s)", yaxis_title="Power (kW)",
            template="simple_white", height=320,
        )
        st.plotly_chart(fig_pq, use_container_width=True)

    with col_h:
        st.markdown("#### Power vs hydraulic head")
        H_range = np.linspace(60, 150, 100)
        P_H = [compute_power(flow_stats["mean_m3s"], H, eta_input) for H in H_range]
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=H_range, y=P_H, mode="lines",
            line=dict(color="#534AB7", width=2),
        ))
        fig_ph.add_vline(x=H_input, line_dash="dash", line_color="#D85A30",
                         annotation_text=f"H = {H_input} m")
        fig_ph.update_layout(
            xaxis_title="Hydraulic head H (m)", yaxis_title="Power (kW)",
            template="simple_white", height=320,
        )
        st.plotly_chart(fig_ph, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Uncertainty Quantification and Sensitivity Analysis")

    col_mc, col_sa = st.columns(2)

    with col_mc:
        st.markdown("#### Monte Carlo simulation — 10,000 samples")
        mc = monte_carlo_uncertainty(
            Q_mean=flow_stats["mean_m3s"], Q_std=flow_stats["std"],
            H_mean=H_input, H_std=5.0, eta=eta_input,
        )
        m1, m2 = st.columns(2)
        m1.metric("Mean P",          f"{mc['P_mean_kw']} kW")
        m2.metric("Std dev P",       f"{mc['P_std_kw']} kW")
        m3, m4 = st.columns(2)
        m3.metric("CI low  (2.5%)",  f"{mc['P_ci_low_kw']} kW")
        m4.metric("CI high (97.5%)", f"{mc['P_ci_high_kw']} kW")

        rng = np.random.default_rng(seed=42)
        Q_s = np.clip(rng.normal(flow_stats["mean_m3s"], flow_stats["std"], 10000), 0, None)
        H_s = np.clip(rng.normal(H_input, 5.0, 10000), 0, None)
        P_s = (eta_input * 1000 * 9.81 * Q_s * H_s) / 1000

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=P_s, nbinsx=60, marker_color="#378ADD", opacity=0.8,
        ))
        fig_mc.add_vline(x=mc["P_mean_kw"],   line_color="#D85A30",
                         annotation_text="Mean")
        fig_mc.add_vline(x=mc["P_ci_low_kw"],  line_dash="dot",
                         line_color="#888780")
        fig_mc.add_vline(x=mc["P_ci_high_kw"], line_dash="dot",
                         line_color="#888780", annotation_text="95% CI")
        fig_mc.update_layout(
            xaxis_title="Power (kW)", yaxis_title="Count",
            template="simple_white", height=320, showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_sa:
        st.markdown("#### Sensitivity analysis — ±30% parameter variation")
        sa_df = sensitivity_analysis(flow_stats["mean_m3s"], H_input, eta_input)
        fig_sa = go.Figure()
        colours = {"Q": "#378ADD", "H": "#1D9E75", "eta": "#D85A30"}
        for param in ["Q", "H", "eta"]:
            sub = sa_df[sa_df["parameter"] == param]
            fig_sa.add_trace(go.Scatter(
                x=sub["change_pct"], y=sub["delta_pct"],
                mode="lines+markers", name=param,
                line=dict(color=colours[param], width=2),
                marker=dict(size=6),
            ))
        fig_sa.add_hline(y=0, line_dash="dot", line_color="#888780")
        fig_sa.update_layout(
            xaxis_title="Parameter change (%)",
            yaxis_title="Power change (%)",
            template="simple_white", height=320,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_sa, use_container_width=True)
        st.caption(
            "All three parameters show equal proportional influence — "
            "physically consistent with the linear power equation."
        )