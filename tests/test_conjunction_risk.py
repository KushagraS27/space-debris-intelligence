import pytest
from src.orbits.conjunction_risk import (
    calculate_nasa_2d_gaussian_pc,
    batch_evaluate_conjunctions,
)


def test_nasa_pc_fresh_tle_danger():
    """Fresh TLE + close miss distance should result in actionable danger."""
    res = calculate_nasa_2d_gaussian_pc(
        miss_distance_km=0.5,
        tle_age_days=0.5,
        max_tle_age_threshold=2.5,
        pc_threshold=1e-4,
    )
    assert not res.is_false_alarm
    assert res.classification == "ACTIONABLE DANGER"
    assert res.decision == "EXECUTE MANEUVER"
    assert res.fuel_saved_usd == 0.0


def test_nasa_pc_aged_tle_suppressed():
    """Aged TLE (3.5 days) should be flagged as false alarm and suppressed."""
    res = calculate_nasa_2d_gaussian_pc(
        miss_distance_km=1.0,
        tle_age_days=3.5,
        max_tle_age_threshold=2.5,
        pc_threshold=1e-4,
        fuel_cost_usd=800.0,
    )
    assert res.is_false_alarm
    assert res.is_high_uncertainty
    assert res.decision == "STAND DOWN"
    assert res.fuel_saved_usd == 800.0
    assert res.sigma_expansion_ratio > 1.0


def test_batch_conjunction_evaluation():
    """Batch evaluation should tally false alarms and calculate fuel dollars saved."""
    sample_candidates = [
        {"miss_distance_km": 0.3, "data_age_hours": 12.0},  # Fresh (0.5d) -> Danger
        {"miss_distance_km": 1.2, "data_age_hours": 96.0},  # 4 days -> False alarm
        {"miss_distance_km": 8.0, "data_age_hours": 10.0},  # Distant -> False alarm
    ]
    summary = batch_evaluate_conjunctions(sample_candidates, fuel_cost_usd=800.0)

    assert summary["total_alerts"] == 3
    assert summary["false_alarms_suppressed"] == 2
    assert summary["actionable_alerts"] == 1
    assert summary["total_fuel_dollars_saved"] == 1600.0
