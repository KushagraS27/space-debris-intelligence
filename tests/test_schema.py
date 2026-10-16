import pytest

from src.data.schema import OrbitalRecord


def sample_record() -> dict:
    return {
        "NORAD_CAT_ID": "25544",
        "OBJECT_NAME": "ISS (ZARYA)",
        "EPOCH": "2026-10-17T00:00:00Z",
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 45.0,
        "ECCENTRICITY": 0.0003,
        "ARG_OF_PERICENTER": 120.0,
        "MEAN_ANOMALY": 240.0,
        "MEAN_MOTION": 15.5,
        "source": "CelesTrak",
        "retrieved_at": "2026-10-17T01:00:00Z",
    }


def test_valid_record_is_accepted() -> None:
    record = OrbitalRecord(**sample_record())

    assert record.object_id == "25544"
    assert record.eccentricity == 0.0003
    assert record.source == "CelesTrak"


def test_invalid_eccentricity_is_rejected() -> None:
    data = sample_record()
    data["ECCENTRICITY"] = 1.2

    with pytest.raises(ValueError):
        OrbitalRecord(**data)