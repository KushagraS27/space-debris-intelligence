from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Any, Mapping

from src.orbits.propagate import propagate_omm


@dataclass(frozen=True)
class CandidateApproach:
    object_a_id: str
    object_b_id: str
    closest_time_utc: datetime
    coarse_min_separation_km: float
    coarse_step_seconds: int
    samples_checked: int


def separation_km(
    position_a: tuple[float, float, float],
    position_b: tuple[float, float, float],
) -> float:
    return sqrt(
        sum((value_a - value_b) ** 2 for value_a, value_b in zip(position_a, position_b))
    )


def screen_pair(
    object_a: Mapping[str, Any],
    object_b: Mapping[str, Any],
    start_time: datetime,
    end_time: datetime,
    coarse_step_seconds: int = 300,
) -> CandidateApproach:
    """Find the smallest sampled TEME separation for one pair of objects.

    This is an initial screening method, not collision-probability estimation.
    A later step will refine the time near the coarse minimum.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be after start_time.")
    if coarse_step_seconds <= 0:
        raise ValueError("coarse_step_seconds must be positive.")

    current_time = start_time
    step = timedelta(seconds=coarse_step_seconds)
    minimum_distance = float("inf")
    closest_time = start_time
    samples_checked = 0

    while current_time <= end_time:
        state_a = propagate_omm(object_a, current_time)
        state_b = propagate_omm(object_b, current_time)

        distance = separation_km(state_a.position_km, state_b.position_km)
        samples_checked += 1

        if distance < minimum_distance:
            minimum_distance = distance
            closest_time = current_time

        current_time += step

    return CandidateApproach(
        object_a_id=str(object_a["NORAD_CAT_ID"]),
        object_b_id=str(object_b["NORAD_CAT_ID"]),
        closest_time_utc=closest_time,
        coarse_min_separation_km=minimum_distance,
        coarse_step_seconds=coarse_step_seconds,
        samples_checked=samples_checked,
    )