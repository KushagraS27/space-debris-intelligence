# 🛰️ Low-Cost, Uncertainty-Aware Space Debris Intelligence

## Overview
A software-first research prototype designed to triage space debris close approaches. It upgrades naive distance-only collision screening by incorporating empirical data age, orbital kinematics, and machine learning novelty detection to prioritize high-risk, uncertain encounters.

## System Architecture
* **Phase 1 (Data Foundation):** Automated CelesTrak GP/OMM ingestion, Pydantic validation, and immutable versioning with SHA-256 manifests.
* **Phase 2 (Physics Baseline):** SGP4 orbital propagation (TEME frame) and coarse pair-screening with full state output (position + velocity at closest approach).
* **Phase 3 (Empirical Uncertainty):** Chronological multi-snapshot comparison for true position degradation, stratified by orbital regime (LEO/MEO/GEO) and data age.
* **Phase 4 (ML Triage):** Isolation Forest anomaly detection with train/test validated RF + Gradient Boosting regressors for Active Debris Removal (ADR) prioritisation.
* **Phase 5 (Product Dashboard):** Streamlit UI with space-themed design, 3D Earth-centered tracking, and ensemble consensus matrices.
* **Phase 6 (Optical Module):** TensorFlow Convolutional AutoEncoder for isolating debris streaks in telescope imagery.

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the main pipeline to ingest data, screen pairs, and score anomalies:
   ```
   python -m src.main
   ```

3. Launch the interactive dashboard:
   ```
   streamlit run src/app/dashboard.py
   ```

4. Run the automated test suite:
   ```
   python -m pytest
   ```

## Training Scripts

Train the Isolation Forest anomaly detector:
```
python -m src.analysis.train_iforest
```

Train the optical AutoEncoder for streak detection:
```
python -m src.optical.train_autoencoder
```

## Project Structure
```
src/
├── data/          # CelesTrak ingestion, Pydantic schema, validation
├── orbits/        # SGP4 propagation, pair screening
├── analysis/      # ML triage, uncertainty analysis, evaluation, benchmarks
├── optical/       # AutoEncoder streak detection, image processing
├── app/           # Streamlit dashboard
└── models/        # Serialised ML models
data/
├── raw/           # Immutable CelesTrak snapshots
├── processed/     # Scored candidates, uncertainty metrics
└── optical_frames/# Telescope imagery
tests/             # pytest suite (schema, screening, optical, triage, integration)
```
