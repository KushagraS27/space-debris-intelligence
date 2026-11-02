import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2

from src.optical.detect_streaks import detect_streaks
from src.orbits.conjunction_risk import (
    calculate_nasa_2d_gaussian_pc,
    batch_evaluate_conjunctions,
)

# ------------------------------------------------------------------ #
# 🛰️ Page Configuration & Modern Light Theme CSS                     #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="OrbitGuard: Space Debris Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --border-color: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --accent-blue: #2563eb;
    --accent-cyan: #0284c7;
    --accent-emerald: #059669;
    --accent-rose: #e11d48;
    --accent-amber: #d97706;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.06) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #f1f5f9;
    border-radius: 12px;
    padding: 6px;
    gap: 6px;
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary) !important;
    font-weight: 600;
    padding: 8px 18px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: var(--accent-blue) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Callout banner */
.problem-card {
    background: #ffffff;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 20px;
}
.badge-danger {
    background-color: #ffe4e6;
    color: #e11d48;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
}
.badge-safe {
    background-color: #d1fae5;
    color: #059669;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Header & Navigation                                                  #
# ------------------------------------------------------------------ #
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## 🛰️ OrbitGuard: Conjunction Risk & Fuel Optimization Engine")
    st.markdown(
        "**Smart Risk Filter for Small Satellites** · NASA 2D Gaussian $P_c$ · TLE Age Error Ellipse · False Alarm Fatigue Suppression"
    )

with col_status:
    st.markdown(
        """
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 10px; text-align: center;">
            <span style="color: #059669; font-weight: bold; font-size: 0.85rem;">● SOCRATES Engine Active</span><br/>
            <span style="color: #64748b; font-size: 0.75rem;">FastAPI Backend Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ #
# Sidebar: Risk Threshold Controls                                     #
# ------------------------------------------------------------------ #
st.sidebar.markdown("### ⚙️ Engine Risk Controls")
max_tle_age = st.sidebar.slider(
    "Max Acceptable TLE Age (Days)",
    min_value=0.5,
    max_value=7.0,
    value=2.5,
    step=0.5,
    help="TLEs older than this expand error bubbles exponentially, creating false alarms.",
)

pc_exp = st.sidebar.slider(
    "Critical Collision Probability Threshold log10(Pc)",
    min_value=-5.0,
    max_value=-2.0,
    value=-4.0,
    step=0.2,
    help="Standard NASA collision threshold is 1e-4 (0.0001).",
)
pc_threshold = 10 ** pc_exp

fuel_cost_usd = st.sidebar.slider(
    "Maneuver Cost per Burn ($ USD)",
    min_value=200,
    max_value=3000,
    value=800,
    step=100,
    help="Average propellant depletion & operational overhead per avoidance maneuver.",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="font-size: 0.8rem; color: #64748b;">
        <strong>Backend Architecture:</strong><br/>
        • <code>/api/conjunction/calculate</code><br/>
        • <code>/api/conjunction/candidates</code><br/>
        • <code>/api/metrics/summary</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Load & Process Data with NASA 2D Gaussian Model                      #
# ------------------------------------------------------------------ #
data_path = Path("data/processed/scored_candidates.csv")
if not data_path.exists():
    st.warning("⚠️ No candidate data found. Please run: `python -m src.main`")
    st.stop()

raw_df = pd.read_csv(data_path)
records = raw_df.to_dict(orient="records")

batch_summary = batch_evaluate_conjunctions(
    candidates=records,
    max_tle_age_threshold=max_tle_age,
    pc_threshold=pc_threshold,
    fuel_cost_usd=fuel_cost_usd,
)

eval_df = pd.DataFrame(batch_summary["candidates"])

# ------------------------------------------------------------------ #
# Top-Level Executive Metrics                                          #
# ------------------------------------------------------------------ #
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Ingested Alerts", batch_summary["total_alerts"])
m2.metric(
    "False Alarms Suppressed",
    f"{batch_summary['suppression_rate_pct']}%",
    delta=f"{batch_summary['false_alarms_suppressed']} alerts filtered",
)
m3.metric("Actionable Risks", batch_summary["actionable_alerts"])
m4.metric(
    "Estimated Fuel Dollars Saved",
    f"${batch_summary['total_fuel_dollars_saved']:,.0f}",
    delta="Propellant Saved",
)

st.divider()

# ------------------------------------------------------------------ #
# Main Tabs                                                            #
# ------------------------------------------------------------------ #
tab_problem, tab_sim, tab_table, tab_3d, tab_optical = st.tabs([
    "💡 Problem & Solution",
    "🎯 Error Ellipse & NASA Pc Simulator",
    "📋 Live Conjunction Feed & Actions",
    "🗺️ 3D Encounter Map",
    "🔭 Optical Streak Analysis",
])

# ── TAB 1: Problem & Solution (from Prompt / Screenshot) ──
with tab_problem:
    st.markdown(
        """
        <div class="problem-card">
            <h3 style="color: #2563eb; margin-top: 0;">1. The Problem: "False Alarm" Fatigue</h3>
            <p style="color: #334155; font-size: 0.95rem;">
                When satellites orbit Earth, they get public warnings from CelesTrak about nearby space debris.
                However, public data uses an old format called <strong>TLE (Two-Line Element)</strong>. TLE data gets inaccurate within 1 to 2 days.
                Because traditional screening software panics over these huge error margins, 
                <strong>over 95% of crash warnings are false alarms</strong>. Small satellite operators get flooded with alerts and don't know which ones are real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #0f172a; margin-top: 0;">🏢 How Big Companies Handle This</h4>
                <p style="font-size: 0.85rem; color: #475569; font-weight: 600;">SpaceX, NASA, Large Defense</p>
                <p style="color: #334155; font-size: 0.9rem;">
                    They pay millions of dollars to private radar networks (like LeoLabs) or receive classified military tracking to know exact satellite positions. They can easily afford high-end tracking tools.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_right:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #0f172a; margin-top: 0;">🚀 Small Companies & Indian Startups</h4>
                <p style="font-size: 0.85rem; color: #e11d48; font-weight: 600;">Budget Constrained Operators</p>
                <p style="color: #334155; font-size: 0.9rem;">
                    They cannot afford expensive radar subscriptions. They rely on free public CelesTrak data and calculate risks manually on spreadsheets, <strong>often burning precious satellite fuel on fake alarms</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px;">
            <h4 style="color: #166534; margin-top: 0;">🛡️ Our Solution: Smart Conjunction Risk Filter</h4>
            <ul style="color: #14532d; font-size: 0.9rem; line-height: 1.7;">
                <li><strong>TLE Age Penalty ($\sigma_{DSE}$):</strong> We calculate how old the CelesTrak data is. If data is 3 days old and distance is 1 km, our system marks it as a <em>False Alarm (High Uncertainty)</em>. If data is fresh and distance is close, it marks it as <em>Real Danger</em>.</li>
                <li><strong>NASA Standard Formula:</strong> We use NASA's 2D Gaussian Covariance formula to calculate the true collision probability ($P_c$) mathematically instead of making simple distance guesses.</li>
                <li><strong>Fuel Savings Metric:</strong> Shows satellite operators exact quantitative ROI: <em>Total Alerts Ingested</em>, <em>False Alarms Filtered Out</em>, and <em>Estimated Fuel Dollars Saved</em>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── TAB 2: Interactive Error Ellipse & NASA Pc Simulator ──
with tab_sim:
    st.subheader("🎯 Orbital Conjunction & Covariance Error Ellipse Simulator")
    st.caption("Inspect how aging TLE data artificially blows up covariance error bubbles, causing phantom collision warnings.")

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        sim_age = st.slider("Simulated TLE Age (Days Since Epoch)", 0.1, 7.0, 3.2, 0.1)
        sim_miss = st.slider("Simulated Miss Distance (km)", 0.05, 5.0, 1.20, 0.05)

        # Run single calculation
        res = calculate_nasa_2d_gaussian_pc(
            miss_distance_km=sim_miss,
            tle_age_days=sim_age,
            max_tle_age_threshold=max_tle_age,
            pc_threshold=pc_threshold,
            fuel_cost_usd=fuel_cost_usd,
        )

        st.markdown(
            f"""
            <div style="background: {'#fef2f2' if not res.is_false_alarm else '#f0fdf4'}; 
                        border: 1px solid {'#fecaca' if not res.is_false_alarm else '#bbf7d0'}; 
                        border-radius: 12px; padding: 16px; margin-top: 15px;">
                <span style="font-size: 0.8rem; font-weight: bold; color: {'#e11d48' if not res.is_false_alarm else '#059669'};">
                    ENGINE CLASSIFICATION: {res.classification}
                </span>
                <h4 style="margin: 6px 0; color: {'#991b1b' if not res.is_false_alarm else '#166534'};">
                    DECISION: {res.decision}
                </h4>
                <p style="font-size: 0.85rem; color: #334155; margin-bottom: 0;">
                    {res.decision_reason}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Formula Metrics Box
        st.markdown("<br/>", unsafe_allow_html=True)
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Calculated Pc", res.raw_pc_exponential)
        c_m2.metric("Covariance Sigma", f"{res.sigma_km} km")
        c_m3.metric("Sigma Expansion", f"{res.sigma_expansion_ratio}x")

    with sim_col2:
        # Plotly Covariance Error Ellipse Canvas
        theta = np.linspace(0, 2 * np.pi, 200)
        # Primary sat at origin (0, 0)
        sat_x, sat_y = 0.0, 0.0
        
        # Debris center placed at (miss_dist, 0)
        deb_x = sim_miss
        deb_y = 0.0

        # Covariance semi-major and semi-minor axis based on sigma
        a = res.sigma_km * 1.5
        b = res.sigma_km * 0.8

        ellipse_x = deb_x + a * np.cos(theta)
        ellipse_y = deb_y + b * np.sin(theta)

        fig_ellipse = go.Figure()

        # Error Ellipse Area
        fig_ellipse.add_trace(go.Scatter(
            x=ellipse_x, y=ellipse_y,
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.15)' if not res.is_false_alarm else 'rgba(217, 119, 6, 0.15)',
            line=dict(color='#e11d48' if not res.is_false_alarm else '#d97706', width=2, dash='dash'),
            name=f"Covariance Bubble ({res.sigma_expansion_ratio}x σ)",
        ))

        # Vector line
        fig_ellipse.add_trace(go.Scatter(
            x=[sat_x, deb_x], y=[sat_y, deb_y],
            mode='lines+text',
            text=["", f"d = {sim_miss:.2f} km"],
            textposition="top center",
            line=dict(color='#0284c7', width=2, dash='dot'),
            name="Encounter Miss Vector",
        ))

        # Primary Sat Point
        fig_ellipse.add_trace(go.Scatter(
            x=[sat_x], y=[sat_y],
            mode='markers+text',
            marker=dict(size=14, color='#2563eb', symbol='circle'),
            text=["Primary Satellite"],
            textposition="bottom center",
            name="Primary Satellite",
        ))

        # Debris Point
        fig_ellipse.add_trace(go.Scatter(
            x=[deb_x], y=[deb_y],
            mode='markers+text',
            marker=dict(size=10, color='#e11d48', symbol='diamond'),
            text=[f"Debris (DSE: {sim_age:.1f}d)"],
            textposition="bottom center",
            name="Debris Object",
        ))

        fig_ellipse.update_layout(
            title="Conjunction Plane (Error Bubble Overlay)",
            xaxis_title="Relative In-Plane (km)",
            yaxis_title="Cross-Plane (km)",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            showlegend=True,
            height=380,
            xaxis=dict(range=[-1.0, max(5.0, sim_miss + a + 1.0)], zeroline=True, gridcolor="#e2e8f0"),
            yaxis=dict(range=[-max(3.0, b + 1.0), max(3.0, b + 1.0)], zeroline=True, gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig_ellipse, use_container_width=True)

# ── TAB 3: Conjunction Feed & Operator Decisions ──
with tab_table:
    st.subheader("📋 Conjunction Alerts & Fuel Action Center")
    filter_choice = st.radio(
        "Display Filter:",
        ["All Alerts", "Actionable Maneuvers Only", "Suppressed False Alarms Only"],
        horizontal=True,
    )

    if filter_choice == "Actionable Maneuvers Only":
        display_df = eval_df[~eval_df["is_false_alarm"]]
    elif filter_choice == "Suppressed False Alarms Only":
        display_df = eval_df[eval_df["is_false_alarm"]]
    else:
        display_df = eval_df

    cols_to_show = [
        "obj_a", "obj_b", "miss_distance_km", "data_age_hours",
        "raw_pc_exp", "sigma_expansion", "classification", "decision", "decision_reason",
    ]
    exist_cols = [c for c in cols_to_show if c in display_df.columns]

    st.dataframe(
        display_df[exist_cols].rename(columns={
            "obj_a": "Primary Sat",
            "obj_b": "Debris ID",
            "miss_distance_km": "Miss Dist (km)",
            "data_age_hours": "Data Age (h)",
            "raw_pc_exp": "Calculated Pc",
            "sigma_expansion": "Sigma Exp (x)",
            "decision": "Action",
            "decision_reason": "Decision Explanation",
        }),
        use_container_width=True,
        height=400,
    )

# ── TAB 4: 3D Encounter Map ──
with tab_3d:
    st.subheader("🗺️ 3D TEME Coordinate Encounter Positions")
    if 'encounter_x' in eval_df.columns:
        fig_3d = px.scatter_3d(
            eval_df,
            x='encounter_x', y='encounter_y', z='encounter_z',
            color='is_false_alarm',
            color_discrete_map={True: "#d97706", False: "#e11d48"},
            hover_name='obj_b',
            hover_data=['miss_distance_km', 'raw_pc_exp', 'decision'],
            title="Spatial Distribution of Conjunction Encounters",
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            height=500,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ── TAB 5: Optical Streak Analysis ──
with tab_optical:
    st.subheader("🔭 Telescope Feed Optical Debris Detection")
    st.caption("AutoEncoder-assisted background suppression and streak vectorization.")
    optical_dir = Path("data/optical_frames")
    available_images = list(optical_dir.glob("*.jpg")) + list(optical_dir.glob("*.png"))

    if available_images:
        selected_img = st.selectbox("Select Astronomical Image:", [img.name for img in available_images])
        img_path = optical_dir / selected_img
        frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    else:
        st.info("Using simulated streak frame.")
        frame = np.zeros((512, 512), dtype=np.uint8)
        cv2.line(frame, (80, 120), (340, 380), 255, 2)

    streak_len = st.slider("Min Streak Length (px)", 10, 200, 50)
    if frame is not None:
        detections, _ = detect_streaks(frame, min_length=streak_len)
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for d in detections:
            if isinstance(d, dict) and 'start' in d and 'end' in d:
                cv2.line(annotated, d['start'], d['end'], (225, 29, 72), 2)
        st.image(annotated, caption="Extracted Debris Vectors (Red)", use_container_width=True)
