import pytest
from datetime import datetime, timedelta, timezone
from src.orbits.propagate import propagate_omm
from src.orbits.screen import screen_pair, separation_km

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
            "REV_AT_EPOCH": 1000
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
            "REV_AT_EPOCH": 2000
        }
    ]

def test_separation_zero():
    pos = (1000.0, 2000.0, 3000.0)
    assert separation_km(pos, pos) == 0.0

def test_screen_pair_detects_closest(mock_omm_records):
    start = datetime.now(timezone.utc)
    rec_a, rec_b = mock_omm_records[0], mock_omm_records[1]
    res = screen_pair(rec_a, rec_b, start, start + timedelta(hours=1), coarse_step_seconds=300)
    assert res.coarse_min_separation_km >= 0.0
    assert res.samples_checked == 13
    assert res.object_a_id == str(rec_a["NORAD_CAT_ID"])
    assert res.object_b_id == str(rec_b["NORAD_CAT_ID"])
