import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Advanced Space Debris Intelligence", layout="wide")
st.title("🌌 Space Debris Intelligence: Advanced ML & 3D Tracking")

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

tab1, tab2, tab3 = st.tabs(["🗺️ 3D Encounter Map", "🤖 ML Model Comparison", "📊 Debris Classifications"])

with tab1:
    st.subheader("3D Earth-Centered Encounter Coordinates")
    fig_3d = px.scatter_3d(data, x='encounter_x', y='encounter_y', z='encounter_z',
                           color='anomaly_consensus', hover_name='obj_b',
                           hover_data=['miss_distance_km', 'classification_label'],
                           color_continuous_scale="Reds", title="Spatial Distribution of Close Approaches")
    fig_3d.update_traces(marker=dict(size=4, opacity=0.7))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Ensemble Model Agreement Matrix")
    st.markdown("Comparing **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor**.")
    
    cols = st.columns(2)
    with cols[0]:
        st.dataframe(data[['obj_a', 'obj_b', 'iforest_anomaly', 'svm_anomaly', 'lof_anomaly', 'anomaly_consensus']]
                     .sort_values('anomaly_consensus', ascending=False).head(15), use_container_width=True)
    with cols[1]:
        fig_vel = px.scatter(data, x="time_to_approach_hours", y="relative_velocity_kms", color="anomaly_consensus",
                             size="miss_distance_km", hover_data=["classification_label"],
                             color_continuous_scale="Viridis", title="Kinematics vs ML Consensus")
        st.plotly_chart(fig_vel, use_container_width=True)

with tab3:
    st.subheader("Candidate Characteristics by Classification")
    fig_box = px.box(data, x="classification_label", y="data_age_hours", color="classification_label",
                     title="Data Age Degradation by Object Type")
    st.plotly_chart(fig_box, use_container_width=True)
