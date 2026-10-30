import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from sklearn.metrics import precision_recall_curve, auc
from src.optical.detect_streaks import detect_streaks

# ------------------------------------------------------------------ #
# 🌌 Space-Themed Page Configuration & Custom CSS                     #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Space Debris Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPACE_CSS = """
<style>
/* ── Import a sci-fi font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Exo+2:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0b0d17;
    --bg-secondary: #111527;
    --bg-card:      #161a2e;
    --accent-blue:  #00d4ff;
    --accent-cyan:  #00f0ff;
    --accent-purple:#7b61ff;
    --text-primary: #e0e6ed;
    --text-muted:   #8892a4;
    --glow-blue:    0 0 15px rgba(0,212,255,0.3);
    --glow-purple:  0 0 15px rgba(123,97,255,0.25);
    --border-dim:   rgba(0,212,255,0.12);
}

/* ── Global background ── */
.stApp {
    background: linear-gradient(170deg, #0b0d17 0%, #0f1123 40%, #151933 100%);
    color: var(--text-primary);
}

/* Starfield overlay (subtle CSS stars) */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(1px 1px at 10% 20%,  rgba(255,255,255,0.5) 50%, transparent 50%),
        radial-gradient(1px 1px at 30% 65%,  rgba(255,255,255,0.4) 50%, transparent 50%),
        radial-gradient(1px 1px at 55% 15%,  rgba(255,255,255,0.3) 50%, transparent 50%),
        radial-gradient(1px 1px at 70% 80%,  rgba(255,255,255,0.35) 50%, transparent 50%),
        radial-gradient(1px 1px at 85% 40%,  rgba(255,255,255,0.45) 50%, transparent 50%),
        radial-gradient(2px 2px at 45% 50%,  rgba(0,212,255,0.25) 50%, transparent 50%),
        radial-gradient(1px 1px at 92% 10%,  rgba(255,255,255,0.4) 50%, transparent 50%),
        radial-gradient(1px 1px at 18% 90%,  rgba(255,255,255,0.3) 50%, transparent 50%),
        radial-gradient(1px 1px at 60% 35%,  rgba(123,97,255,0.2) 50%, transparent 50%),
        radial-gradient(1px 1px at 5% 55%,   rgba(255,255,255,0.3) 50%, transparent 50%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1e 0%, #141834 100%) !important;
    border-right: 1px solid var(--border-dim);
}
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary) !important;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h1 { letter-spacing: 2px; }

/* ── Body text ── */
p, li, span, label, .stMarkdown {
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: var(--glow-blue);
    transition: box-shadow 0.3s;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 0 25px rgba(0,212,255,0.45);
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    color: #4ade80 !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border-dim);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-muted) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600;
    padding: 8px 16px;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.08);
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* ── DataFrame / tables ── */
.stDataFrame, div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-dim);
    border-radius: 10px;
    overflow: hidden;
}

/* ── Slider ── */
.stSlider > div > div {
    color: var(--accent-blue) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-dim) !important;
}

/* ── Info / Warning banners ── */
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Selectbox / multiselect ── */
div[data-baseweb="select"] {
    background: var(--bg-card) !important;
    border-color: var(--border-dim) !important;
    border-radius: 8px !important;
}
</style>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Title                                                                #
# ------------------------------------------------------------------ #
st.markdown("# 🛰️ SPACE DEBRIS INTELLIGENCE")
st.markdown("##### Uncertainty-Aware ML Triage · Optical Tracking · Active Debris Removal")

# ------------------------------------------------------------------ #
# Load Data                                                            #
# ------------------------------------------------------------------ #
data_path = Path("data/processed/scored_candidates.csv")
if not data_path.exists():
    st.warning("No data found. Please run the pipeline first: `python -m src.main`")
    st.stop()

data = pd.read_csv(data_path)
class_map = {0: "Payload", 1: "Rocket Body", 2: "Known Debris", 3: "Foreign/Unknown"}
data['classification_label'] = data.get(
    'object_classification', pd.Series(["Unknown"] * len(data))
).map(class_map).fillna("Unknown")

# ------------------------------------------------------------------ #
# Sidebar Filters                                                      #
# ------------------------------------------------------------------ #
st.sidebar.markdown("## 🔍 Mission Filters")
selected_classes = st.sidebar.multiselect(
    "Debris Classification",
    options=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"],
    default=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"],
)

filtered_data = data[data['classification_label'].isin(selected_classes)]

# ------------------------------------------------------------------ #
# Top-level Metrics                                                    #
# ------------------------------------------------------------------ #
col1, col2, col3 = st.columns(3)
col1.metric("Total Encounters", len(filtered_data))

anomaly_col = (
    'anomaly_consensus' if 'anomaly_consensus' in data.columns
    else ('anomaly_score' if 'anomaly_score' in data.columns else None)
)
if anomaly_col:
    col2.metric("Critical Anomalies", len(filtered_data[filtered_data[anomaly_col] > 0]))
else:
    col2.metric("Critical Anomalies", 0)

if 'adr_priority_score' in filtered_data.columns:
    col3.metric("High Priority ADR Targets", len(filtered_data[filtered_data['adr_priority_score'] > 60.0]))
else:
    col3.metric("High Priority ADR Targets", 0)

st.divider()

# ------------------------------------------------------------------ #
# Plotly Dark Template                                                 #
# ------------------------------------------------------------------ #
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0e1225",
        font=dict(family="Exo 2", color="#c0c8d4"),
        xaxis=dict(gridcolor="#1c2240"),
        yaxis=dict(gridcolor="#1c2240"),
    )
)

# ------------------------------------------------------------------ #
# Tabs                                                                 #
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ 3D Encounter Map",
    "🤖 ML Model Comparison",
    "📊 Debris Classifications",
    "🔭 Optical Streak Analysis",
    "🎓 Academic Metrics",
    "🧹 Active Debris Removal",
])

# ── Tab 1: 3D Encounter Map ──
with tab1:
    if not filtered_data.empty and 'encounter_x' in filtered_data.columns:
        fig_3d = px.scatter_3d(
            filtered_data,
            x='encounter_x', y='encounter_y', z='encounter_z',
            color=anomaly_col,
            hover_name='obj_b',
            hover_data=['miss_distance_km', 'classification_label'],
            color_continuous_scale="Plasma",
            title="TEME Encounter Positions",
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(**PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_3d, use_container_width=True)

# ── Tab 2: ML Model Comparison ──
with tab2:
    cols = st.columns(2)
    with cols[0]:
        target_cols = [
            'obj_a', 'obj_b', 'iforest_anomaly', 'svm_anomaly',
            'lof_anomaly', 'anomaly_score', 'anomaly_consensus',
        ]
        exist_cols = [c for c in target_cols if c in filtered_data.columns]
        if exist_cols:
            sort_c = 'anomaly_consensus' if 'anomaly_consensus' in exist_cols else exist_cols[-1]
            st.dataframe(
                filtered_data[exist_cols].sort_values(sort_c, ascending=False).head(15),
                use_container_width=True,
            )
    with cols[1]:
        if (not filtered_data.empty and anomaly_col
                and 'relative_velocity_kms' in filtered_data.columns):
            fig_vel = px.scatter(
                filtered_data,
                x="time_to_approach_hours",
                y="relative_velocity_kms",
                color=anomaly_col,
                size="miss_distance_km",
                hover_data=["classification_label"],
                color_continuous_scale="Viridis",
                title="Velocity vs Time-to-Approach",
            )
            fig_vel.update_layout(**PLOT_TEMPLATE["layout"])
            st.plotly_chart(fig_vel, use_container_width=True)

# ── Tab 3: Debris Classifications ──
with tab3:
    pie_col, box_col = st.columns(2)
    with pie_col:
        class_counts = filtered_data['classification_label'].value_counts().reset_index()
        class_counts.columns = ['Classification', 'Count']
        if not class_counts.empty:
            fig_pie = px.pie(
                class_counts, names='Classification', values='Count',
                title="Distribution of Tracked Objects",
                hole=0.4, color='Classification',
                color_discrete_map={
                    "Payload": "#00d4ff",
                    "Rocket Body": "#7b61ff",
                    "Known Debris": "#f59e0b",
                    "Foreign/Unknown": "#ef4444",
                },
            )
            fig_pie.update_layout(**PLOT_TEMPLATE["layout"])
            st.plotly_chart(fig_pie, use_container_width=True)
    with box_col:
        if not filtered_data.empty and 'data_age_hours' in filtered_data.columns:
            fig_box = px.box(
                filtered_data,
                x="classification_label", y="data_age_hours",
                color="classification_label",
                title="Data Age Degradation by Object Type",
                color_discrete_map={
                    "Payload": "#00d4ff",
                    "Rocket Body": "#7b61ff",
                    "Known Debris": "#f59e0b",
                    "Foreign/Unknown": "#ef4444",
                },
            )
            fig_box.update_layout(**PLOT_TEMPLATE["layout"])
            st.plotly_chart(fig_box, use_container_width=True)

    base_cols = ['obj_a', 'obj_b', 'classification_label', 'miss_distance_km', 'time_to_approach_hours']
    if anomaly_col:
        base_cols.append(anomaly_col)
    exist_base = [c for c in base_cols if c in filtered_data.columns]
    st.dataframe(filtered_data[exist_base], use_container_width=True)

# ── Tab 4: Optical Streak Analysis ──
with tab4:
    st.subheader("Local Dataset Telescope Feed Analysis")
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        optical_dir = Path("data/optical_frames")
        available_images = list(optical_dir.glob("*.jpg")) + list(optical_dir.glob("*.png"))

        if available_images:
            selected_img = st.selectbox("Select Local Astronomical Image:", [img.name for img in available_images])
            img_path = optical_dir / selected_img
            frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            st.warning("No local images found in data/optical_frames. Using a synthetic frame.")
            frame = np.zeros((512, 512), dtype=np.uint8)
            cv2.line(frame, (80, 120), (340, 380), 255, 2)

        streak_len = st.slider("Min Streak Length (px)", 10, 200, 50)

        if frame is not None:
            detections, _ = detect_streaks(frame, min_length=streak_len)
            annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for d in detections:
                if isinstance(d, dict) and 'start' in d and 'end' in d:
                    cv2.line(annotated, d['start'], d['end'], (0, 212, 255), 2)
            st.image(annotated, caption="Deep Learning Extracted Debris Vectors (Cyan)", use_container_width=True)
        else:
            st.error("Failed to load image data.")
            detections = []

    with col_opt2:
        st.write(f"**Detected Candidate Streaks:** `{len(detections) if 'detections' in locals() else 0}`")
        if 'detections' in locals() and detections:
            st.dataframe(pd.DataFrame(detections), use_container_width=True)

# ── Tab 5: Academic Metrics ──
with tab5:
    st.subheader("Formal Research Evaluation")
    budget_hours = st.slider("Nightly Telescope Budget (Hours)", 1.0, 12.0, 4.0, 0.5)
    slots = int((budget_hours * 60) // 15)

    if 'object_classification' in data.columns and anomaly_col and 'relative_velocity_kms' in data.columns:
        b_queue = data.sort_values("miss_distance_km", ascending=True).head(slots)
        data["triage_rank"] = (
            data[anomaly_col] * 100.0
            + data["relative_velocity_kms"]
            - data["miss_distance_km"] / 1000.0
        )
        m_queue = data.sort_values("triage_rank", ascending=False).head(slots)

        b_crit = len(b_queue[(b_queue["object_classification"] == 3) | (b_queue[anomaly_col] > 0)])
        m_crit = len(m_queue[(m_queue["object_classification"] == 3) | (m_queue[anomaly_col] > 0)])

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Anomalies Caught (Distance-Only)", f"{b_crit} / {slots}")
        col_s2.metric(
            "Anomalies Caught (ML Ensemble)",
            f"{m_crit} / {slots}",
            delta=f"+{m_crit - b_crit} over baseline",
        )

# ── Tab 6: Active Debris Removal ──
with tab6:
    st.subheader("🧹 Active Debris Removal (ADR) Prioritisation")
    if 'adr_priority_score' in filtered_data.columns:
        adr_cols = [
            c for c in [
                'obj_a', 'obj_b', 'classification_label', 'miss_distance_km',
                'adr_rf_score', 'adr_gb_score', 'adr_priority_score',
            ] if c in filtered_data.columns
        ]
        adr_data = filtered_data[adr_cols].sort_values('adr_priority_score', ascending=False)
        st.dataframe(adr_data, use_container_width=True)
        st.info("Model Comparison: Random Forest and Gradient Boosting regressors independently weigh mass proxies and collision risk.")
    else:
        st.warning("No ADR scores found in data. Please run the main pipeline.")
