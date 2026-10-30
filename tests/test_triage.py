"""Tests for the ML triage pipeline (anomaly detection + ADR scoring)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.triage import score_candidates


@pytest.fixture
def sample_candidates():
    """Create a minimal candidates DataFrame with realistic values."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "obj_a": [f"SAT-{i}" for i in range(n)],
        "obj_b": [f"DEB-{i}" for i in range(n)],
        "miss_distance_km": np.random.uniform(100, 10000, n),
        "time_to_approach_hours": np.random.uniform(0, 6, n),
        "data_age_hours": np.random.uniform(1, 100, n),
        "relative_velocity_kms": np.random.uniform(3, 16, n),
        "object_classification": np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.2, 0.45, 0.15]),
    })


def test_score_candidates_returns_dataframe(sample_candidates, tmp_path, monkeypatch):
    """score_candidates should return a DataFrame with all expected columns."""
    # Use a temporary model path so we don't pollute the repo
    monkeypatch.setattr(
        "src.analysis.triage.Path",
        lambda *a, **kw: tmp_path / "model.joblib" if str(a[0]).endswith(".joblib") else Path(*a, **kw),
    )
    result = score_candidates(sample_candidates)

    assert isinstance(result, pd.DataFrame)
    for col in ["anomaly_score", "is_anomaly", "anomaly_consensus",
                 "adr_rf_score", "adr_gb_score", "adr_priority_score"]:
        assert col in result.columns


def test_adr_scores_are_non_negative(sample_candidates):
    """ADR priority scores must never be negative after the formula fix."""
    result = score_candidates(sample_candidates)
    assert (result["adr_priority_score"] >= 0).all(), (
        f"Found negative ADR scores: {result['adr_priority_score'].min()}"
    )


def test_adr_scores_capped_at_100(sample_candidates):
    """ADR scores should be clipped at 100."""
    result = score_candidates(sample_candidates)
    assert (result["adr_priority_score"] <= 100).all()


def test_payloads_score_higher_than_debris(sample_candidates):
    """Payloads and rocket bodies should generally score higher ADR priority."""
    result = score_candidates(sample_candidates)

    heavy = result[result["object_classification"].isin([0, 1])]["adr_priority_score"].mean()
    light = result[result["object_classification"] == 2]["adr_priority_score"].mean()

    # Heavy objects (Payload/RocketBody) should have higher mean ADR score
    assert heavy > light, f"Heavy mean={heavy:.2f} should be > light mean={light:.2f}"


def test_result_is_sorted_by_priority(sample_candidates):
    """Output should be sorted descending by adr_priority_score."""
    result = score_candidates(sample_candidates)
    scores = result["adr_priority_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_anomaly_consensus_values(sample_candidates):
    """anomaly_consensus should only be 0 or 3."""
    result = score_candidates(sample_candidates)
    unique_vals = set(result["anomaly_consensus"].unique())
    assert unique_vals.issubset({0, 3})
