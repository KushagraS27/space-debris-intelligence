import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
import requests
from datetime import datetime, timezone

from src.optical.detect_streaks import detect_streaks
from src.orbits.conjunction_risk import (
    calculate_nasa_2d_gaussian_pc,
    batch_evaluate_conjunctions,
)
from src.data.fetch_gp import fetch_gp_snapshot

# ------------------------------------------------------------------ #
# 🌌 Page Configuration & Deep Space Dark Styling                    #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Space Debris Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPACE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-primary:   #080b14;
    --bg-secondary: #0e1322;
    --bg-card:      #141b2f;
    --accent-cyan:  #00f0ff;
    --accent-blue:  #3b82f6;
    --accent-purple:#8b5cf6;
    --accent-rose:  #f43f5e;
    --accent-amber: #f59e0b;
    --accent-green: #10b981;
    --text-primary: #e2e8f0;
    --text-muted:   #94a3b8;
    --border-dim:   rgba(0, 240, 255, 0.15);
}

.stApp {
    background: radial-gradient(circle at 50% 10%, #111a33 0%, #080b14 75%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0b0f1d !important;
    border-right: 1px solid var(--border-dim) !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.08) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-weight: 700 !important;
    font-family: 'Orbitron', sans-serif !important;
}

/* Tab Headers */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-secondary);
    border-radius: 12px;
    padding: 5px;
    gap: 6px;
    border: 1px solid var(--border-dim);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-muted) !important;
    font-weight: 600;
    padding: 8px 18px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(0, 240, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

.problem-box {
    background: linear-gradient(135deg, rgba(20, 27, 47, 0.8) 0%, rgba(14, 19, 34, 0.9) 100%);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 18px 22px;
    border-top: 1px solid var(--border-dim);
    border-right: 1px solid var(--border-dim);
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 20px;
}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Title Header                                                         #
# ------------------------------------------------------------------ #
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("<h1 style='font-family: Orbitron; letter-spacing: 1.5px; margin-bottom: 0;'>🛰️ SPACE DEBRIS INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-top: 4px;'><b>Conjunction Risk & Fuel Optimization Engine</b> · NASA 2D Gaussian $P_c$ · TLE Age Uncertainty Filter · Optical Streak Detection</p>", unsafe_allow_html=True)

with col_badge:
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 10px; text-align: center;">
            <span style="color: #10b981; font-weight: bold; font-family: JetBrains Mono; font-size: 0.85rem;">● SYSTEM OPERATIONAL</span><br/>
            <span style="color: #94a3b8; font-size: 0.75rem;">FastAPI :8000 | Client Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ #
# Sidebar: Threshold Controls                                          #
# ------------------------------------------------------------------ #
st.sidebar.markdown("### ⚙️ Engine Risk Parameters")
max_tle_age = st.sidebar.slider(
    "Max Acceptable TLE Age (Days)",
    min_value=0.5,
    max_value=7.0,
    value=2.5,
    step=0.5,
    help="TLEs older than this expand error bubbles exponentially, creating false alarms.",
)

pc_exp = st.sidebar.slider(
    "Critical Collision Threshold log10(Pc)",
    min_value=-5.0,
    max_value=-2.0,
    value=-4.0,
    step=0.2,
    help="Standard NASA collision avoidance threshold is 1e-4.",
)
pc_threshold = 10 ** pc_exp

fuel_cost_usd = st.sidebar.slider(
    "Maneuver Cost per Burn ($ USD)",
    min_value=200,
    max_value=3000,
    value=800,
    step=100,
    help="Propellant & operational cost per orbital avoidance maneuver.",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="font-size: 0.8rem; color: #94a3b8; font-family: JetBrains Mono;">
        <strong>Endpoints & Clients:</strong><br/>
        • <code>/api/conjunction/calculate</code><br/>
        • <code>/api/conjunction/candidates</code><br/>
        • <code>web_client/index.html</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Load & Process Screening Dataset                                     #
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
# Executive Summary Metrics                                            #
# ------------------------------------------------------------------ #
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Ingested Encounters", batch_summary["total_alerts"])
m2.metric(
    "False Alarms Suppressed",
    f"{batch_summary['suppression_rate_pct']}%",
    delta=f"{batch_summary['false_alarms_suppressed']} alerts filtered",
)
m3.metric("Actionable Risks (Real Danger)", batch_summary["actionable_alerts"])
m4.metric(
    "Est. Thruster Fuel Saved",
    f"${batch_summary['total_fuel_dollars_saved']:,.0f}",
    delta="Propellant Saved",
)

st.divider()

# ------------------------------------------------------------------ #
# Main Tabs (Including Enhanced 3D, High-Fidelity Sim & Live Feed)    #
# ------------------------------------------------------------------ #
tab_problem, tab_sim, tab_table, tab_3d, tab_live, tab_optical = st.tabs([
    "💡 Problem & Solution",
    "🎯 NASA Pc & Error Ellipse Simulator",
    "📋 Conjunction Feed & Actions",
    "🌍 3D Earth Encounter Sphere",
    "📡 Live Data Pipeline",
    "🔭 Optical Streak Analysis",
])

# ── TAB 1: Problem & Solution ──
with tab_problem:
    st.markdown(
        """
        <div class="problem-box">
            <h3 style="color: #00f0ff; margin-top: 0; font-family: Orbitron;">1. The Problem: "False Alarm" Fatigue</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                When satellites orbit Earth, they get public warnings from CelesTrak about nearby space debris.
                However, public data uses an old format called <strong>TLE (Two-Line Element)</strong>. TLE data gets inaccurate within 1 to 2 days.
                Because traditional screening software panics over these huge error margins, 
                <strong style="color: #f43f5e;">over 95% of crash warnings are false alarms</strong>. Satellite operators get flooded with alerts and don't know which ones are real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ind_l, col_ind_r = st.columns(2)
    with col_ind_l:
        st.markdown(
            """
            <div style="background: rgba(20, 27, 47, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #3b82f6; margin-top: 0; font-family: Orbitron;">🏢 Big Companies (SpaceX, NASA)</h4>
                <p style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">High-Budget Defense & Mega-Constellations</p>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    They pay millions of dollars to private radar networks (like LeoLabs) or get custom military tracking to know exact satellite positions. They can easily afford high-end tracking tools.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ind_r:
        st.markdown(
            """
            <div style="background: rgba(20, 27, 47, 0.7); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #f43f5e; margin-top: 0; font-family: Orbitron;">🚀 Small Companies & Indian Startups</h4>
                <p style="font-size: 0.85rem; color: #f43f5e; font-weight: 600;">Budget-Constrained Commercial Fleets</p>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    They cannot afford expensive radar subscriptions. They rely on free public CelesTrak data and calculate risks manually on spreadsheets, <strong style="color: #f59e0b;">often burning precious satellite fuel on fake alarms</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 20px;">
            <h4 style="color: #10b981; margin-top: 0; font-family: Orbitron;">🛡️ Our Solution & Advantages</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8;">
                <li><strong style="color: #00f0ff;">TLE Age Penalty (σ_DSE):</strong> We compute how old the tracking data is. If data is 3 days old and distance is 1 km, our system marks it as a <em>False Alarm (High Uncertainty)</em>. If data is fresh and distance is close, it flags an actionable <em>Real Danger</em>.</li>
                <li><strong style="color: #00f0ff;">NASA Standard Formula:</strong> We apply NASA's 2D Gaussian Covariance formulation to calculate the true collision probability ($P_c$) mathematically instead of naive distance guessing.</li>
                <li><strong style="color: #00f0ff;">Fuel Savings Metric:</strong> Directly quantifies operator ROI: <em>Total Alerts Ingested</em>, <em>False Alarms Filtered Out</em>, and <em>Estimated Fuel Dollars Saved</em>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── TAB 2: Corrected & High-Fidelity Simulator ──
with tab_sim:
    st.subheader("🎯 High-Fidelity Conjunction & Covariance Simulator")
    st.caption("Live mathematical assessment comparing true encounter miss distance against covariance bubble expansion.")

    sim_c1, sim_c2 = st.columns([1, 1.25])
    with sim_c1:
        sim_age = st.slider("TLE Age (Days Since Epoch)", 0.1, 7.0, 3.2, 0.1)
        sim_miss = st.slider("Encounter Miss Distance (km)", 0.05, 5.0, 1.20, 0.05)
        sim_angle = st.slider("Relative Trajectory Angle (°)", 0, 180, 50, 5)

        sim_res = calculate_nasa_2d_gaussian_pc(
            miss_distance_km=sim_miss,
            tle_age_days=sim_age,
            max_tle_age_threshold=max_tle_age,
            pc_threshold=pc_threshold,
            fuel_cost_usd=fuel_cost_usd,
        )

        badge_bg = "rgba(244, 63, 94, 0.15)" if not sim_res.is_false_alarm else "rgba(16, 185, 129, 0.15)"
        badge_border = "rgba(244, 63, 94, 0.4)" if not sim_res.is_false_alarm else "rgba(16, 185, 129, 0.4)"
        badge_text = "#f43f5e" if not sim_res.is_false_alarm else "#10b981"

        st.markdown(
            f"""
            <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 12px; padding: 18px; margin-top: 15px;">
                <span style="font-family: Orbitron; font-size: 0.82rem; font-weight: bold; color: {badge_text};">
                    CLASSIFICATION: {sim_res.classification}
                </span>
                <h3 style="margin: 6px 0; color: #ffffff; font-family: Orbitron;">
                    ACTION: {sim_res.decision}
                </h3>
                <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0;">
                    {sim_res.decision_reason}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        m_s1, m_s2, m_s3 = st.columns(3)
        m_s1.metric("Calculated Pc", sim_res.raw_pc_exponential)
        m_s2.metric("Positional Sigma", f"{sim_res.sigma_km} km")
        m_s3.metric("Sigma Expansion", f"{sim_res.sigma_expansion_ratio}x")

    with sim_c2:
        # High precision 2D encounter geometry
        rad = np.radians(sim_angle)
        sat_x, sat_y = 0.0, 0.0
        deb_x = sim_miss * np.cos(rad)
        deb_y = sim_miss * np.sin(rad)

        theta = np.linspace(0, 2 * np.pi, 250)
        a_axis = sim_res.sigma_km * 2.2
        b_axis = sim_res.sigma_km * 0.9

        ellipse_x_unrot = a_axis * np.cos(theta)
        ellipse_y_unrot = b_axis * np.sin(theta)
        ellipse_x = deb_x + ellipse_x_unrot * np.cos(rad) - ellipse_y_unrot * np.sin(rad)
        ellipse_y = deb_y + ellipse_x_unrot * np.sin(rad) + ellipse_y_unrot * np.cos(rad)

        fig_sim = go.Figure()

        # 1. Covariance Uncertainty Bubble
        fig_sim.add_trace(go.Scatter(
            x=ellipse_x, y=ellipse_y,
            fill='toself',
            fillcolor='rgba(244, 63, 94, 0.22)' if not sim_res.is_false_alarm else 'rgba(245, 158, 11, 0.16)',
            line=dict(color='#f43f5e' if not sim_res.is_false_alarm else '#f59e0b', width=2, dash='dash'),
            name=f"Covariance Error Bubble ({sim_res.sigma_expansion_ratio}x σ)",
        ))

        # 2. Satellite 0.5 km Keep-Out Zone
        fig_sim.add_trace(go.Scatter(
            x=0.5 * np.cos(theta), y=0.5 * np.sin(theta),
            fill='toself',
            fillcolor='rgba(0, 240, 255, 0.10)',
            line=dict(color='#00f0ff', width=1.5, dash='dot'),
            name="Satellite Safety Sphere (0.5 km)",
        ))

        # 3. Trajectory Lines
        traj_len = max(2.5, sim_miss * 1.5)
        fig_sim.add_trace(go.Scatter(
            x=[-traj_len * np.sin(rad), traj_len * np.sin(rad)],
            y=[traj_len * np.cos(rad), -traj_len * np.cos(rad)],
            mode='lines',
            line=dict(color='rgba(59, 130, 246, 0.4)', width=2),
            name="Primary Orbit Path",
        ))

        # 4. Miss Vector
        fig_sim.add_trace(go.Scatter(
            x=[sat_x, deb_x], y=[sat_y, deb_y],
            mode='lines+text',
            text=["", f"d = {sim_miss:.2f} km"],
            textposition="top center",
            line=dict(color='#00f0ff', width=2),
            name="Miss Distance Vector",
        ))

        # 5. Satellite & Debris Markers
        fig_sim.add_trace(go.Scatter(
            x=[sat_x], y=[sat_y],
            mode='markers+text',
            marker=dict(size=14, color='#00f0ff', symbol='circle', line=dict(color='#ffffff', width=1)),
            text=["Primary Satellite"],
            textposition="bottom center",
            name="Primary Satellite",
        ))

        fig_sim.add_trace(go.Scatter(
            x=[deb_x], y=[deb_y],
            mode='markers+text',
            marker=dict(size=12, color='#f43f5e', symbol='diamond', line=dict(color='#ffffff', width=1)),
            text=[f"Debris ({sim_age:.1f}d)"],
            textposition="top right",
            name="Debris Object",
        ))

        axis_limit = max(4.0, float(sim_miss + a_axis + 0.8))
        fig_sim.update_layout(
            title="Conjunction Encounter Plane (True Scale)",
            height=440,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0c1122",
            font=dict(family="Inter", color="#cbd5e1"),
            xaxis=dict(
                title="Relative In-Track (km)",
                range=[-axis_limit * 0.7, axis_limit],
                gridcolor="#1b2440",
                zerolinecolor="#334155",
            ),
            yaxis=dict(
                title="Relative Cross-Track (km)",
                range=[-axis_limit * 0.7, axis_limit * 0.7],
                gridcolor="#1b2440",
                zerolinecolor="#334155",
            ),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

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
            "obj_b": "Debris NORAD",
            "miss_distance_km": "Miss Dist (km)",
            "data_age_hours": "Data Age (h)",
            "raw_pc_exp": "Calculated Pc",
            "sigma_expansion": "Sigma Exp (x)",
            "decision": "Recommended Action",
            "decision_reason": "Operational Rationale",
        }),
        use_container_width=True,
        height=400,
    )

# ── TAB 4: Enhanced 3D Earth Encounter Sphere ──
with tab_3d:
    st.subheader("🌍 3D Earth-Centered Encounter Sphere (TEME Frame)")
    st.caption("Crystal-clear visualization with true 3D Earth reference sphere (R=6371 km) and altitude orbits.")

    if 'encounter_x' in eval_df.columns:
        fig_earth = go.Figure()

        # 1. 3D Earth Wireframe Sphere (Mean Earth Radius = 6371 km)
        r_earth = 6371.0
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 25)
        x_e = r_earth * np.outer(np.cos(u), np.sin(v))
        y_e = r_earth * np.outer(np.sin(u), np.sin(v))
        z_e = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))

        fig_earth.add_trace(go.Surface(
            x=x_e, y=y_e, z=z_e,
            colorscale=[[0, '#0f172a'], [0.5, '#1e293b'], [1, '#0284c7']],
            showscale=False,
            opacity=0.6,
            name="Earth (R=6,371 km)",
            hoverinfo="name",
        ))

        # 2. Add LEO Reference Equatorial Ring (Altitude ~500 km)
        r_leo = r_earth + 500.0
        theta_ring = np.linspace(0, 2 * np.pi, 100)
        fig_earth.add_trace(go.Scatter3d(
            x=r_leo * np.cos(theta_ring),
            y=r_leo * np.sin(theta_ring),
            z=np.zeros_like(theta_ring),
            mode='lines',
            line=dict(color='rgba(0, 240, 255, 0.25)', width=2, dash='dot'),
            name="LEO 500km Reference Ring",
        ))

        # 3. Actionable Real Threats (Red Glowing Orbs)
        actionable_df = eval_df[~eval_df["is_false_alarm"]]
        if not actionable_df.empty:
            fig_earth.add_trace(go.Scatter3d(
                x=actionable_df['encounter_x'],
                y=actionable_df['encounter_y'],
                z=actionable_df['encounter_z'],
                mode='markers',
                marker=dict(size=6, color='#f43f5e', symbol='diamond', opacity=0.95),
                text=actionable_df.apply(lambda r: f"NORAD {r['obj_b']}<br>Miss: {r['miss_distance_km']:.1f} km<br>Pc: {r['raw_pc_exp']}", axis=1),
                name="Actionable Real Threat",
            ))

        # 4. Suppressed False Alarms (Amber / Cyan Dots)
        suppressed_df = eval_df[eval_df["is_false_alarm"]]
        if not suppressed_df.empty:
            fig_earth.add_trace(go.Scatter3d(
                x=suppressed_df['encounter_x'],
                y=suppressed_df['encounter_y'],
                z=suppressed_df['encounter_z'],
                mode='markers',
                marker=dict(size=3.5, color='#f59e0b', opacity=0.6),
                text=suppressed_df.apply(lambda r: f"NORAD {r['obj_b']}<br>Miss: {r['miss_distance_km']:.1f} km<br>Suppressed (Aged TLE)", axis=1),
                name="Suppressed False Alarm",
            ))

        fig_earth.update_layout(
            scene=dict(
                xaxis=dict(title="X TEME (km)", backgroundcolor="#080b14", gridcolor="#1b2440"),
                yaxis=dict(title="Y TEME (km)", backgroundcolor="#080b14", gridcolor="#1b2440"),
                zaxis=dict(title="Z TEME (km)", backgroundcolor="#080b14", gridcolor="#1b2440"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=580,
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0, y=1, bgcolor="rgba(14, 19, 34, 0.8)"),
        )
        st.plotly_chart(fig_earth, use_container_width=True)

# ── TAB 5: Live Data Pipeline Feed ──
with tab_live:
    st.subheader("📡 Live CelesTrak Ingestion Pipeline")
    st.caption("Trigger live General Perturbation (GP/OMM) catalog synchronization and monitor snapshot health.")

    live_c1, live_c2 = st.columns([1, 2])
    with live_c1:
        st.markdown("##### Pipeline Controls")
        catalog_group = st.selectbox("CelesTrak Catalog Group:", ["stations", "active", "starlink", "visual"])
        
        if st.button("🚀 Fetch Fresh CelesTrak Snapshot", use_container_width=True):
            with st.spinner(f"Ingesting latest {catalog_group.upper()} snapshot from CelesTrak..."):
                try:
                    snapshot_dir = fetch_gp_snapshot(catalog_group)
                    st.success(f"Snapshot saved to `{snapshot_dir.name}` with SHA-256 validation.")
                except Exception as e:
                    st.error(f"Live fetch notice: {e}")

        st.markdown(
            """
            <div style="background: rgba(20, 27, 47, 0.7); border: 1px solid var(--border-dim); border-radius: 10px; padding: 14px; margin-top: 15px; font-size: 0.82rem;">
                <strong>Automated Synchronization:</strong><br/>
                • Frequency: 4-hour polling interval<br/>
                • Format: OMM v3.0 JSON Standard<br/>
                • Integrity: SHA-256 Merkle Provenance
            </div>
            """,
            unsafe_allow_html=True,
        )

    with live_c2:
        st.markdown("##### Snapshot Provenance & Integrity Ledger")
        snapshots_dir = Path("data/raw/snapshots")
        if snapshots_dir.exists():
            snaps = sorted([s for s in snapshots_dir.iterdir() if s.is_dir()], reverse=True)
            snap_data = []
            for s in snaps[:8]:
                manifest_file = s / "manifest.json"
                if manifest_file.exists():
                    import json
                    m = json.loads(manifest_file.read_text(encoding="utf-8"))
                    snap_data.append({
                        "Snapshot ID": s.name,
                        "Source": m.get("source", "CelesTrak"),
                        "Group": m.get("group", "STATIONS"),
                        "Records": m.get("record_count", 0),
                        "SHA-256 Hash": (m.get("sha256", "")[:12] + "..."),
                    })
            if snap_data:
                st.dataframe(pd.DataFrame(snap_data), use_container_width=True)
            else:
                st.info("No snapshots found in data/raw/snapshots.")

# ── TAB 6: Optical Streak Detection ──
with tab_optical:
    st.subheader("🔭 Telescope Feed Optical Debris Detection")
    st.caption("Directional linear filtering & vectorization detecting real satellite trains, tumbling fragments, and crossing tracks.")

    optical_dir = Path("data/optical_frames")
    available_images = list(optical_dir.glob("*.jpg")) + list(optical_dir.glob("*.png"))

    col_opt_ctrl1, col_opt_ctrl2 = st.columns([2, 1])
    with col_opt_ctrl1:
        selected_img_name = st.selectbox(
            "Select Telescope Frame to Analyze (Real & Synthetic Library):",
            [img.name for img in available_images],
        )
    with col_opt_ctrl2:
        streak_len = st.slider("Min Streak Length (px)", 10, 150, 30)

    if selected_img_name:
        img_path = optical_dir / selected_img_name
        frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if frame is not None:
            detections, mask = detect_streaks(frame, min_length=streak_len)

            c_img1, c_img2 = st.columns(2)
            with c_img1:
                st.markdown("**Original Raw Frame**")
                st.image(frame, use_container_width=True, caption=f"Input: {selected_img_name}")

            with c_img2:
                st.markdown(f"**Identified Debris Vectors: {len(detections)} Detected**")
                annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for d in detections:
                    cv2.line(annotated, d['start'], d['end'], (0, 240, 255), 2)
                    cv2.circle(annotated, d['start'], 3, (244, 63, 94), -1)
                st.image(annotated, use_container_width=True, caption="Vector Overlay (Cyan = Streak, Red = Head)")

            if detections:
                st.markdown("##### Extracted Debris Streak Vectors")
                st.dataframe(pd.DataFrame(detections), use_container_width=True)
            else:
                st.info("Clean sky detected: Zero debris streaks found in this observation window.")
