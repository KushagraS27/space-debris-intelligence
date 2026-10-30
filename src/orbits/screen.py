from dataclasses import dataclass, field
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
    position_a_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_b_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_a_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_b_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


def separation_km(
    position_a: tuple[float, float, float],
    position_b: tuple[float, float, float],
) -> float:
    return sqrt(
        sum((value_a - value_b) ** 2 for value_a, value_b in zip(position_a, position_b))
    )


def relative_velocity_km_s(
    velocity_a: tuple[float, float, float],
    velocity_b: tuple[float, float, float],
) -> float:
    """Compute the scalar relative velocity between two objects."""
    return sqrt(
        sum((va - vb) ** 2 for va, vb in zip(velocity_a, velocity_b))
    )


def screen_pair(
    object_a: Mapping[str, Any],
    object_b: Mapping[str, Any],
    start_time: datetime,
    end_time: datetime,
    coarse_step_seconds: int = 300,
) -> CandidateApproach:
    """Find the smallest sampled TEME separation for one pair of objects.

    Returns the full state (position + velocity) at the closest approach
    time so downstream consumers can derive physical quantities instead
    of injecting synthetic features.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be after start_time.")
    if coarse_step_seconds <= 0:
        raise ValueError("coarse_step_seconds must be positive.")

    current_time = start_time
    step = timedelta(seconds=coarse_step_seconds)
    minimum_distance = float("inf")
    closest_time = start_time

    best_state_a = None
    best_state_b = None

    samples_checked = 0

    while current_time <= end_time:
        state_a = propagate_omm(object_a, current_time)
        state_b = propagate_omm(object_b, current_time)

        distance = separation_km(state_a.position_km, state_b.position_km)
        samples_checked += 1

        if distance < minimum_distance:
            minimum_distance = distance
            closest_time = current_time
            best_state_a = state_a
            best_state_b = state_b

        current_time += step

    pos_a = best_state_a.position_km if best_state_a else (0.0, 0.0, 0.0)
    pos_b = best_state_b.position_km if best_state_b else (0.0, 0.0, 0.0)
    vel_a = best_state_a.velocity_km_s if best_state_a else (0.0, 0.0, 0.0)
    vel_b = best_state_b.velocity_km_s if best_state_b else (0.0, 0.0, 0.0)

    return CandidateApproach(
        object_a_id=str(object_a["NORAD_CAT_ID"]),
        object_b_id=str(object_b["NORAD_CAT_ID"]),
        closest_time_utc=closest_time,
        coarse_min_separation_km=minimum_distance,
        coarse_step_seconds=coarse_step_seconds,
        samples_checked=samples_checked,
        position_a_km=tuple(pos_a),
        position_b_km=tuple(pos_b),
        velocity_a_km_s=tuple(vel_a),
        velocity_b_km_s=tuple(vel_b),
    )
