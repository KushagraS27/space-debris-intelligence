import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Space Debris Intelligence", layout="wide")
st.title("🛰️ Space Debris Intelligence Dashboard")

st.markdown("### Close Approach Candidates")

# Dummy data representing the final pipeline output
data = pd.DataFrame([
    {"obj_a": "123", "obj_b": "456", "miss_distance_km": 5.0, "time_to_approach_hours": 12, "anomaly_score": 0.1},
    {"obj_a": "123", "obj_b": "789", "miss_distance_km": 0.5, "time_to_approach_hours": 24, "anomaly_score": -0.5}, 
    {"obj_a": "456", "obj_b": "789", "miss_distance_km": 15.0, "time_to_approach_hours": 4, "anomaly_score": 0.3},
])

st.dataframe(data.style.highlight_min(subset=['anomaly_score'], color='#ff4b4b'), use_container_width=True)

st.markdown("### Risk Analysis")
fig = px.scatter(data, x="time_to_approach_hours", y="miss_distance_km", color="anomaly_score", 
                 title="Approach Horizon vs Miss Distance (Color = Anomaly Score)", 
                 labels={"time_to_approach_hours": "Hours to Approach", "miss_distance_km": "Miss Distance (km)"},
                 color_continuous_scale="RdYlGn")
st.plotly_chart(fig, use_container_width=True)
