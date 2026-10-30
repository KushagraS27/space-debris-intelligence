import pytest
from datetime import datetime, timezone

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


def test_negative_eccentricity_is_rejected() -> None:
    data = sample_record()
    data["ECCENTRICITY"] = -0.01

    with pytest.raises(ValueError):
        OrbitalRecord(**data)


def test_zero_eccentricity_is_accepted() -> None:
    data = sample_record()
    data["ECCENTRICITY"] = 0.0

    record = OrbitalRecord(**data)
    assert record.eccentricity == 0.0


def test_retrieved_at_naive_gets_utc() -> None:
    data = sample_record()
    data["retrieved_at"] = "2026-10-17T01:00:00"  # no timezone

    record = OrbitalRecord(**data)
    assert record.retrieved_at.tzinfo is not None


def test_object_name_is_optional() -> None:
    data = sample_record()
    del data["OBJECT_NAME"]

    record = OrbitalRecord(**data)
    assert record.object_name is None


def test_quality_flags_default_empty() -> None:
    record = OrbitalRecord(**sample_record())
    assert record.quality_flags == []
