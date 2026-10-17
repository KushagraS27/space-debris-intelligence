# Low-Cost, Uncertainty-Aware Space Debris Intelligence

## Overview
A software-first research prototype designed to triage space debris close approaches. It upgrades naive distance-only collision screening by incorporating empirical data age, orbital kinematics, and machine learning novelty detection to prioritize high-risk, uncertain encounters.

## System Architecture
* **Phase 1 (Data Foundation):** Automated CelesTrak GP/OMM ingestion, validation, and immutable versioning.
* **Phase 2 (Physics Baseline):** SGP4 orbital propagation and coarse pair-screening.
* **Phase 3 (Empirical Uncertainty):** Chronological multi-snapshot comparison for true position degradation.
* **Phase 4 (ML Triage):** Multi-model ensemble (Isolation Forest, One-Class SVM, LOF) evaluating kinematic risk and data staleness.
* **Phase 5 (Product Dashboard):** Streamlit UI featuring 3D Earth-centered tracking and ensemble consensus matrices.
* **Phase 6 (Optical Module):** TensorFlow AutoEncoder for isolating debris streaks in telescope imagery.

## Quick Start
1. Run the main pipeline to ingest data, screen pairs, and score anomalies:
   `python -m src.main`
2. Launch the interactive dashboard:
   `python -m streamlit run src/app/dashboard.py`
3. Run the automated test suite:
   `python -m pytest`
