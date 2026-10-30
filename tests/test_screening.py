import pytest
from datetime import datetime, timedelta, timezone
from src.orbits.propagate import propagate_omm
from src.orbits.screen import screen_pair, separation_km, relative_velocity_km_s


@pytest.fixture
def mock_omm_records():
    return [
        {
            "NORAD_CAT_ID": 22675,
            "OBJECT_ID": "1993-061A",
            "EPOCH": "2023-10-26T12:00:00.000000",
            "MEAN_MOTION": 14.123,
            "ECCENTRICITY": 0.001,
            "INCLINATION": 98.2,
            "RA_OF_ASC_NODE": 120.0,
            "ARG_OF_PERICENTER": 45.0,
            "MEAN_ANOMALY": 90.0,
            "BSTAR": 0.0001,
            "MEAN_MOTION_DOT": 0.0,
            "MEAN_MOTION_DDOT": 0.0,
            "CLASSIFICATION_TYPE": "U",
            "EPHEMERIS_TYPE": 0,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 1000,
        },
        {
            "NORAD_CAT_ID": 33757,
            "OBJECT_ID": "2009-008A",
            "EPOCH": "2023-10-26T12:00:00.000000",
            "MEAN_MOTION": 14.125,
            "ECCENTRICITY": 0.0015,
            "INCLINATION": 98.1,
            "RA_OF_ASC_NODE": 120.1,
            "ARG_OF_PERICENTER": 45.1,
            "MEAN_ANOMALY": 90.1,
            "BSTAR": 0.00015,
            "MEAN_MOTION_DOT": 0.0,
            "MEAN_MOTION_DDOT": 0.0,
            "CLASSIFICATION_TYPE": "U",
            "EPHEMERIS_TYPE": 0,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 2000,
        },
    ]


def test_separation_zero():
    pos = (1000.0, 2000.0, 3000.0)
    assert separation_km(pos, pos) == 0.0


def test_separation_known_distance():
    """3-4-5 triangle → distance should be ~7.07 km along the diagonal."""
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 5.0)
    expected = (3**2 + 4**2 + 5**2) ** 0.5
    assert abs(separation_km(a, b) - expected) < 1e-9


def test_relative_velocity():
    v_a = (7.0, 0.0, 0.0)
    v_b = (0.0, 7.0, 0.0)
    result = relative_velocity_km_s(v_a, v_b)
    expected = (7**2 + 7**2) ** 0.5
    assert abs(result - expected) < 1e-9


def test_relative_velocity_same_vector_is_zero():
    v = (5.0, 3.0, 1.0)
    assert relative_velocity_km_s(v, v) == 0.0


def test_screen_pair_detects_closest(mock_omm_records):
    start = datetime.now(timezone.utc)
    rec_a, rec_b = mock_omm_records[0], mock_omm_records[1]
    res = screen_pair(rec_a, rec_b, start, start + timedelta(hours=1), coarse_step_seconds=300)
    assert res.coarse_min_separation_km >= 0.0
    assert res.samples_checked == 13
    assert res.object_a_id == str(rec_a["NORAD_CAT_ID"])
    assert res.object_b_id == str(rec_b["NORAD_CAT_ID"])


def test_screen_pair_returns_positions(mock_omm_records):
    """The updated screen_pair must return positions and velocities."""
    start = datetime.now(timezone.utc)
    rec_a, rec_b = mock_omm_records
    res = screen_pair(rec_a, rec_b, start, start + timedelta(hours=1))

    assert len(res.position_a_km) == 3
    assert len(res.position_b_km) == 3
    assert len(res.velocity_a_km_s) == 3
    assert len(res.velocity_b_km_s) == 3
    # Positions should not all be zero (objects are in orbit)
    assert any(v != 0.0 for v in res.position_a_km)


def test_screen_pair_rejects_invalid_window(mock_omm_records):
    start = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="end_time must be after start_time"):
        screen_pair(mock_omm_records[0], mock_omm_records[1], start, start - timedelta(hours=1))


def test_screen_pair_rejects_zero_step(mock_omm_records):
    start = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="coarse_step_seconds must be positive"):
        screen_pair(mock_omm_records[0], mock_omm_records[1], start, start + timedelta(hours=1), coarse_step_seconds=0)


def test_propagation_requires_tz_aware_timestamp(mock_omm_records):
    naive_dt = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        propagate_omm(mock_omm_records[0], naive_dt)


def test_propagation_returns_valid_state(mock_omm_records):
    ts = datetime.now(timezone.utc)
    state = propagate_omm(mock_omm_records[0], ts)
    assert state.object_id == "22675"
    assert state.frame == "TEME"
    assert len(state.position_km) == 3
    assert len(state.velocity_km_s) == 3
