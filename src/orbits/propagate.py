from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, jday


@dataclass(frozen=True)
class PropagatedState:
    object_id: str
    timestamp_utc: datetime
    frame: str
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]


def propagate_omm(
    omm_record: Mapping[str, Any],
    timestamp: datetime,
) -> PropagatedState:
    """Propagate one OMM record to a UTC time using SGP4.

    Output coordinates use the TEME frame. They must only be compared with
    states propagated in the same frame and at the same timestamp.
    """
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware.")
    timestamp_utc = timestamp.astimezone(timezone.utc)

    satellite = Satrec()
    omm.initialize(satellite, dict(omm_record))

    second = timestamp_utc.second + timestamp_utc.microsecond / 1_000_000
    jd, fraction = jday(
        timestamp_utc.year,
        timestamp_utc.month,
        timestamp_utc.day,
        timestamp_utc.hour,
        timestamp_utc.minute,
        second,
    )

    error_code, position_km, velocity_km_s = satellite.sgp4(jd, fraction)

    if error_code != 0:
        message = SGP4_ERRORS.get(error_code, "unknown SGP4 error")
        raise RuntimeError(f"SGP4 propagation failed: {message}")

    return PropagatedState(
        object_id=str(omm_record["NORAD_CAT_ID"]),
        timestamp_utc=timestamp_utc,
        frame="TEME",
        position_km=tuple(position_km),
        velocity_km_s=tuple(velocity_km_s),
    )