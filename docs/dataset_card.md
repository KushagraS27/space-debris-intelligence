# Dataset Card: CelesTrak GP/OMM Snapshots

## Dataset Overview
* **Source:** CelesTrak General Perturbation (GP) Orbital Mean Elements (OMM).
* **Format:** Chronologically versioned JSON snapshots.
* **Population:** Active satellites, rocket bodies, and catalogued fragmentation debris.

## Intended Use
* Research-grade SGP4 orbital propagation.
* Evaluating empirical position degradation over specific propagation horizons.
* Training semi-supervised novelty detection models (Isolation Forests, SVMs) to triage anomalous behavior.

## Limitations, Biases & Safety Boundaries
* **Not for Collision Avoidance:** This system is a research prototype. Output candidate distances are labeled strictly as "screening," and must never be used for operational satellite maneuvers or collision probability claims.
* **Data Staleness:** Subject to global sensor tracking gaps; older records inherently contain higher unmodeled uncertainty.
* **Synthetic Injection:** For the purpose of ML ensemble evaluation, synthetic classifications (Foreign/Unknown) and relative velocity kinematics were programmatically injected into the screening pipeline.
