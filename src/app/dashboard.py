import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from pathlib import Path
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

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Encounters Screened", len(data))
col2.metric("Critical Consensus Anomalies", len(data[data['anomaly_consensus'] == 3]))
col3.metric("Foreign/Unknown Tracked", len(data[data['object_classification'] == 3]))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ 3D Encounter Map", 
    "🤖 ML Model Comparison", 
    "📊 Debris Classifications", 
    "🔭 Optical Streak Analysis"
])

with tab1:
    st.subheader("3D Earth-Centered Encounter Coordinates")
    fig_3d = px.scatter_3d(
        data, x='encounter_x', y='encounter_y', z='encounter_z',
        color='anomaly_consensus', hover_name='obj_b',
        hover_data=['miss_distance_km', 'classification_label'],
        color_continuous_scale="Reds", title="Spatial Distribution of Close Approaches"
    )
    fig_3d.update_traces(marker=dict(size=4, opacity=0.7))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Ensemble Model Agreement Matrix")
    st.markdown("Comparing **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor**.")
    
    cols = st.columns(2)
    with cols[0]:
        st.dataframe(
            data[['obj_a', 'obj_b', 'iforest_anomaly', 'svm_anomaly', 'lof_anomaly', 'anomaly_consensus']]
            .sort_values('anomaly_consensus', ascending=False).head(15), 
            use_container_width=True
        )
    with cols[1]:
        fig_vel = px.scatter(
            data, x="time_to_approach_hours", y="relative_velocity_kms", color="anomaly_consensus",
            size="miss_distance_km", hover_data=["classification_label"],
            color_continuous_scale="Viridis", title="Kinematics vs ML Consensus"
        )
        st.plotly_chart(fig_vel, use_container_width=True)

with tab3:
    st.subheader("Candidate Characteristics by Classification")
    fig_box = px.box(
        data, x="classification_label", y="data_age_hours", color="classification_label",
        title="Data Age Degradation by Object Type"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.subheader("Telescope Streak Ingestion & Extraction")
    st.markdown("Simulate or analyze optical frames to identify high-velocity non-catalogued debris signatures.")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        streak_len = st.slider("Min Streak Length (px)", min_value=10, max_value=150, value=35)
        num_stars = st.slider("Simulated Star Density", min_value=20, max_value=300, value=80)
        
        # Generate synthetic optical observation frame
        frame = np.zeros((512, 512), dtype=np.uint8)
        np.random.seed(101)
        sx = np.random.randint(0, 512, num_stars)
        sy = np.random.randint(0, 512, num_stars)
        for x, y in zip(sx, sy):
            cv2.circle(frame, (x, y), 1, int(np.random.randint(150, 255)), -1)
            
        # Draw linear debris streak
        cv2.line(frame, (80, 120), (340, 380), 255, 2)
        
        detections, thresh_mask = detect_streaks(frame, min_length=streak_len)
        
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for d in detections:
            cv2.line(annotated, d['start'], d['end'], (0, 0, 255), 2)
            cv2.circle(annotated, d['start'], 4, (0, 255, 0), -1)
            cv2.circle(annotated, d['end'], 4, (255, 0, 0), -1)

        st.image(annotated, caption="Extracted Debris Vector (Red) vs Background Stars", use_container_width=True)

    with col_opt2:
        st.write(f"**Detected Candidate Streaks:** `{len(detections)}`")
        if detections:
            st.dataframe(pd.DataFrame(detections), use_container_width=True)
            st.success("Association Status: High angular velocity streak detected. Candidate matched for tracklet prioritization.")
        else:
            st.warning("No streaks matched current length threshold.")
