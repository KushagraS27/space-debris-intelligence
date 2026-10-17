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
if 'object_classification' in data.columns:
    data['classification_label'] = data['object_classification'].map(class_map)
else:
    data['classification_label'] = "Unknown"

st.sidebar.header("🔍 Filter Candidates")
selected_classes = st.sidebar.multiselect(
    "Filter by Debris Classification",
    options=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"],
    default=["Payload", "Rocket Body", "Known Debris", "Foreign/Unknown"]
)

filtered_data = data[data['classification_label'].isin(selected_classes)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Encounters Displayed", len(filtered_data))
if 'anomaly_consensus' in filtered_data.columns:
    col2.metric("Critical Consensus Anomalies", len(filtered_data[filtered_data['anomaly_consensus'] >= 2]))
else:
    col2.metric("Critical Anomalies", 0)

if 'adr_priority_score' in filtered_data.columns:
    col3.metric("High Priority ADR Targets", len(filtered_data[filtered_data['adr_priority_score'] > 75.0]))
else:
    col3.metric("High Priority ADR Targets", 0)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ 3D Encounter Map", 
    "🤖 ML Model Comparison", 
    "📊 Debris Classifications", 
    "🔭 Optical Streak Analysis",
    "🎓 Academic Metrics",
    "🧹 Active Debris Removal"
])

with tab1:
    if not filtered_data.empty and 'encounter_x' in filtered_data.columns:
        color_col = 'anomaly_consensus' if 'anomaly_consensus' in filtered_data.columns else None
        fig_3d = px.scatter_3d(filtered_data, x='encounter_x', y='encounter_y', z='encounter_z', color=color_col, hover_name='obj_b', hover_data=['miss_distance_km', 'classification_label'], color_continuous_scale="Reds")
        fig_3d.update_traces(marker=dict(size=4, opacity=0.7))
        st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    cols = st.columns(2)
    with cols[0]:
        # Dynamically select columns that actually exist to prevent KeyErrors
        available_ml_cols = [c for c in ['obj_a', 'obj_b', 'iforest_anomaly', 'svm_anomaly', 'lof_anomaly', 'anomaly_score', 'anomaly_consensus'] if c in filtered_data.columns]
        sort_target = 'anomaly_consensus' if 'anomaly_consensus' in available_ml_cols else available_ml_cols[-1]
        st.dataframe(filtered_data[available_ml_cols].sort_values(sort_target, ascending=False).head(15), use_container_width=True)
    
    with cols[1]:
        if not filtered_data.empty and 'anomaly_consensus' in filtered_data.columns:
            fig_vel = px.scatter(filtered_data, x="time_to_approach_hours", y="relative_velocity_kms", color="anomaly_consensus", size="miss_distance_km", hover_data=["classification_label"], color_continuous_scale="Viridis")
            st.plotly_chart(fig_vel, use_container_width=True)

with tab3:
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
            
    base_cols = ['obj_a', 'obj_b', 'classification_label', 'miss_distance_km', 'time_to_approach_hours']
    if 'anomaly_consensus' in filtered_data.columns: base_cols.append('anomaly_consensus')
    sort_val = 'anomaly_consensus' if 'anomaly_consensus' in filtered_data.columns else 'miss_distance_km'
    ascend_val = False if sort_val == 'anomaly_consensus' else True
    st.dataframe(filtered_data[base_cols].sort_values(sort_val, ascending=ascend_val), use_container_width=True)

with tab4:
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        streak_len = st.slider("Min Streak Length (px)", 10, 150, 35)
        frame = np.zeros((512, 512), dtype=np.uint8)
        cv2.line(frame, (80, 120), (340, 380), 255, 2)
        detections, _ = detect_streaks(frame, min_length=streak_len)
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for d in detections:
            cv2.line(annotated, d['start'], d['end'], (0, 0, 255), 2)
        st.image(annotated, caption="Extracted Debris Vector", use_container_width=True)
    with col_opt2:
        st.write(f"**Detected Candidate Streaks:** `{len(detections)}`")
        if detections: st.dataframe(pd.DataFrame(detections), use_container_width=True)

with tab5:
    st.subheader("Formal Research Evaluation")
    budget_hours = st.slider("Nightly Telescope Budget (Hours)", 1.0, 12.0, 4.0, 0.5)
    slots = int((budget_hours * 60) // 15)
    
    if 'anomaly_consensus' in data.columns and 'object_classification' in data.columns:
        b_queue = data.sort_values("miss_distance_km", ascending=True).head(slots)
        data["triage_rank"] = (data["anomaly_consensus"] * 100.0) + data["relative_velocity_kms"] - (data["miss_distance_km"] / 1000.0)
        m_queue = data.sort_values("triage_rank", ascending=False).head(slots)
        
        b_crit = len(b_queue[(b_queue["object_classification"] == 3) | (b_queue["anomaly_consensus"] == 3)])
        m_crit = len(m_queue[(m_queue["object_classification"] == 3) | (m_queue["anomaly_consensus"] == 3)])
        
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Anomalies Caught (Distance-Only)", f"{b_crit} / {slots}")
        col_s2.metric("Anomalies Caught (ML Ensemble)", f"{m_crit} / {slots}", delta=f"+{m_crit - b_crit} over baseline")

with tab6:
    st.subheader("🧹 Active Debris Removal (ADR) Prioritization")
    st.markdown("Identifies large, massive objects (Rocket Bodies and Defunct Payloads) exhibiting severe kinematic risk for physical de-orbiting missions.")
    
    if 'adr_priority_score' in filtered_data.columns:
        adr_cols = [c for c in ['obj_a', 'obj_b', 'classification_label', 'miss_distance_km', 'adr_rf_score', 'adr_gb_score', 'adr_priority_score'] if c in filtered_data.columns]
        adr_data = filtered_data[adr_cols].sort_values('adr_priority_score', ascending=False)
        st.dataframe(adr_data.style.background_gradient(subset=['adr_priority_score'], cmap='Reds'), use_container_width=True)
        st.info("Model Comparison: Random Forest and Gradient Boosting regressors independently weigh mass proxies and collision risk.")
    else:
        st.warning("No ADR scores found in data.")
