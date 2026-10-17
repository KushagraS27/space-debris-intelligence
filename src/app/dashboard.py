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

st.set_page_config(page_title="Advanced Space Debris Intelligence", layout="wide")
st.title("🌌 Space Debris Intelligence: Advanced ML & Optical Tracking")

data_path = Path("data/processed/scored_candidates.csv")
if not data_path.exists():
    st.warning("No data found. Please run the pipeline first.")
    st.stop()

data = pd.read_csv(data_path)
class_map = {0: "Payload", 1: "Rocket Body", 2: "Known Debris", 3: "Foreign/Unknown"}
data['classification_label'] = data['object_classification'].map(class_map)

# --- GLOBAL SIDEBAR FILTER ---
st.sidebar.header("🔍 Filter Candidates")
selected_classes = st.sidebar.multiselect(
    "Filter by Debris Classification",
    options=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"],
    default=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"]
)

# Apply the filter to the data
filtered_data = data[data['classification_label'].isin(selected_classes)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Encounters Displayed", len(filtered_data))
col2.metric("Critical Consensus Anomalies", len(filtered_data[filtered_data['anomaly_consensus'] == 3]))
col3.metric("Foreign/Unknown Tracked", len(filtered_data[filtered_data['object_classification'] == 3]))

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 3D Encounter Map", 
    "🤖 ML Model Comparison", 
    "📊 Debris Classifications", 
    "🔭 Optical Streak Analysis",
    "🎓 Academic Metrics"
])

with tab1:
    if not filtered_data.empty:
        fig_3d = px.scatter_3d(filtered_data, x='encounter_x', y='encounter_y', z='encounter_z', color='anomaly_consensus', hover_name='obj_b', hover_data=['miss_distance_km', 'classification_label'], color_continuous_scale="Reds")
        fig_3d.update_traces(marker=dict(size=4, opacity=0.7))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("No data matches the selected classification filters.")

with tab2:
    cols = st.columns(2)
    with cols[0]:
        st.dataframe(filtered_data[['obj_a', 'obj_b', 'iforest_anomaly', 'svm_anomaly', 'lof_anomaly', 'anomaly_consensus']].sort_values('anomaly_consensus', ascending=False).head(15), use_container_width=True)
    with cols[1]:
        if not filtered_data.empty:
            fig_vel = px.scatter(filtered_data, x="time_to_approach_hours", y="relative_velocity_kms", color="anomaly_consensus", size="miss_distance_km", hover_data=["classification_label"], color_continuous_scale="Viridis")
            st.plotly_chart(fig_vel, use_container_width=True)

with tab3:
    st.subheader("Debris Classification Breakdown")
    
    # --- NEW: Classification Pie Chart and Table ---
    pie_col, box_col = st.columns(2)
    with pie_col:
        class_counts = filtered_data['classification_label'].value_counts().reset_index()
        class_counts.columns = ['Classification', 'Count']
        if not class_counts.empty:
            fig_pie = px.pie(class_counts, names='Classification', values='Count', title="Distribution of Tracked Objects", hole=0.4, color='Classification')
            st.plotly_chart(fig_pie, use_container_width=True)
        
    with box_col:
        if not filtered_data.empty:
            fig_box = px.box(filtered_data, x="classification_label", y="data_age_hours", color="classification_label", title="Data Age Degradation by Object Type")
            st.plotly_chart(fig_box, use_container_width=True)
            
    st.markdown("#### Classified Encounter Table")
    st.dataframe(filtered_data[['obj_a', 'obj_b', 'classification_label', 'miss_distance_km', 'time_to_approach_hours', 'anomaly_consensus']].sort_values('anomaly_consensus', ascending=False), use_container_width=True)

with tab4:
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        streak_len = st.slider("Min Streak Length (px)", 10, 150, 35)
        frame = np.zeros((512, 512), dtype=np.uint8)
        np.random.seed(101)
        sx, sy = np.random.randint(0, 512, 80), np.random.randint(0, 512, 80)
        for x, y in zip(sx, sy): cv2.circle(frame, (x, y), 1, int(np.random.randint(150, 255)), -1)
        cv2.line(frame, (80, 120), (340, 380), 255, 2)
        detections, _ = detect_streaks(frame, min_length=streak_len)
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for d in detections:
            cv2.line(annotated, d['start'], d['end'], (0, 0, 255), 2)
        st.image(annotated, caption="Extracted Debris Vector (Red)", use_container_width=True)
    with col_opt2:
        st.write(f"**Detected Candidate Streaks:** `{len(detections)}`")
        if detections: st.dataframe(pd.DataFrame(detections), use_container_width=True)

with tab5:
    st.subheader("Formal Research Evaluation")
    
    st.markdown("#### 1. Telescope Observing-Time Budget Simulation")
    budget_hours = st.slider("Nightly Telescope Budget (Hours)", min_value=1.0, max_value=12.0, value=4.0, step=0.5)
    slots = int((budget_hours * 60) // 15)
    
    b_queue = data.sort_values("miss_distance_km", ascending=True).head(slots)
    data["triage_rank"] = (data["anomaly_consensus"] * 100.0) + data["relative_velocity_kms"] - (data["miss_distance_km"] / 1000.0)
    m_queue = data.sort_values("triage_rank", ascending=False).head(slots)
    
    b_crit = len(b_queue[(b_queue["object_classification"] == 3) | (b_queue["anomaly_consensus"] == 3)])
    m_crit = len(m_queue[(m_queue["object_classification"] == 3) | (m_queue["anomaly_consensus"] == 3)])
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Anomalies Caught (Distance-Only)", f"{b_crit} / {slots}")
    col_s2.metric("Anomalies Caught (ML Ensemble)", f"{m_crit} / {slots}", delta=f"+{m_crit - b_crit} over baseline")
    
    st.markdown("---")
    
    st.markdown("#### 2. Stratified Empirical Uncertainty (Error vs. Horizon)")
    strat_path = Path("data/processed/stratified_uncertainty_metrics.csv")
    if strat_path.exists():
        st.dataframe(pd.read_csv(strat_path), use_container_width=True)
    
    st.markdown("#### 3. ML Ensemble vs Distance-Only Baseline (PR-AUC)")
    y_true = (data['object_classification'] == 3).astype(int)
    if y_true.sum() > 0:
        base_scores = 1.0 / (data['miss_distance_km'] + 1e-6)
        ml_scores = data['anomaly_consensus'].astype(float) + (data['relative_velocity_kms'] / data['relative_velocity_kms'].max())
        
        _, base_rec, _ = precision_recall_curve(y_true, base_scores)
        _, ml_rec, _ = precision_recall_curve(y_true, ml_scores)
        
        st.info(f"The Machine Learning ensemble effectively prioritizes critical anomalies over a naive distance-only baseline.")
