"""Integration tests for the full pipeline components."""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from src.analysis.stratified_uncertainty import classify_regime


def test_classify_regime_leo():
    assert classify_regime(15.0) == "LEO"
    assert classify_regime(11.25) == "LEO"


def test_classify_regime_meo():
    assert classify_regime(5.0) == "MEO"
    assert classify_regime(1.8) == "MEO"


def test_classify_regime_geo():
    assert classify_regime(1.0) == "GEO/HEO"
    assert classify_regime(0.5) == "GEO/HEO"


def test_scored_candidates_csv_schema():
    """If scored_candidates.csv exists, verify it has the expected columns."""
    path = Path("data/processed/scored_candidates.csv")
    if not path.exists():
        pytest.skip("scored_candidates.csv not available")

    df = pd.read_csv(path)
    required = {
        "obj_a", "obj_b", "miss_distance_km", "time_to_approach_hours",
        "data_age_hours", "relative_velocity_kms", "object_classification",
    }
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_stratified_uncertainty_csv_schema():
    """If the stratified CSV exists, check its structure."""
    path = Path("data/processed/stratified_uncertainty_metrics.csv")
    if not path.exists():
        pytest.skip("stratified_uncertainty_metrics.csv not available")

    df = pd.read_csv(path)
    assert "regime" in df.columns
    assert "Age_Bin" in df.columns
    assert "Median_Err_km" in df.columns
    assert len(df) > 0
