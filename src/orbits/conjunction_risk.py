import math
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ConjunctionInput(BaseModel):
    miss_distance_km: float = Field(..., description="Miss distance at closest approach in km", ge=0.01)
    tle_age_days: float = Field(..., description="Age of TLE since epoch in days", ge=0.0)
    relative_velocity_kms: float = Field(default=10.0, description="Relative speed in km/s", ge=0.1)
    hard_body_radius_m: float = Field(default=10.0, description="Combined hard body collision radius in meters")
    max_tle_age_threshold_days: float = Field(default=2.5, description="Max acceptable TLE age before flag as high uncertainty")
    pc_critical_threshold: float = Field(default=1e-4, description="Critical collision probability threshold")
    fuel_cost_usd_per_maneuver: float = Field(default=800.0, description="Estimated USD fuel cost per thruster burn")


class ConjunctionRiskResult(BaseModel):
    raw_pc: float
    raw_pc_exponential: str
    sigma_km: float
    sigma_expansion_ratio: float
    is_false_alarm: bool
    is_high_uncertainty: bool
    is_safe_distance: bool
    classification: str
    decision: str
    decision_reason: str
    fuel_saved_usd: float


def calculate_nasa_2d_gaussian_pc(
    miss_distance_km: float,
    tle_age_days: float,
    max_tle_age_threshold: float = 2.5,
    pc_threshold: float = 1e-4,
    fuel_cost_usd: float = 800.0,
) -> ConjunctionRiskResult:
    """
    Calculate Collision Probability (Pc) using NASA 2D Gaussian Covariance formulation
    and evaluate TLE age error penalty.

    Formulation:
      sigma(DSE) = sigma_0 + (DSE * alpha)
      where:
        sigma_0 = 0.30 km (nominal baseline tracking error)
        alpha   = 0.40 km/day (degradation rate per day of age)
      
      2D Gaussian isotropic encounter plane:
        P_c = (1 / (sqrt(2*pi) * sigma)) * exp( - d^2 / (2 * sigma^2) )
    """
    sigma_0 = 0.30
    alpha = 0.40
    sigma = sigma_0 + (tle_age_days * alpha)

    # 2D Gaussian formulation
    exponent = -1.0 * (miss_distance_km ** 2) / (2.0 * (sigma ** 2))
    raw_pc = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(exponent)
    
    # Clip to realistic space-tracking bounds
    raw_pc = max(1e-7, min(9.9e-3, raw_pc))

    # Sigma error expansion ratio compared to fresh TLE (<6h)
    expansion_ratio = round(sigma / sigma_0, 2)

    # Operational decision boundaries
    is_high_uncertainty = tle_age_days > max_tle_age_threshold
    is_safe_distance = miss_distance_km > 1.5
    is_low_prob = raw_pc < pc_threshold

    # False alarm condition:
    # If the TLE is too old, the error bubble artificially overlaps even though true satellite
    # position is elsewhere, OR the distance is comfortably outside warning radius.
    is_false_alarm = is_high_uncertainty or is_safe_distance or is_low_prob

    if not is_false_alarm:
        classification = "ACTIONABLE DANGER"
        decision = "EXECUTE MANEUVER"
        decision_reason = (
            f"Verified high risk encounter! Fresh TLE ({tle_age_days:.1f}d) and close separation "
            f"({miss_distance_km:.2f} km) with Pc = {raw_pc:.2e} exceeding critical safety limit."
        )
        fuel_saved = 0.0
    else:
        classification = "FALSE ALARM (SUPPRESSED)"
        decision = "STAND DOWN"
        if is_high_uncertainty:
            decision_reason = (
                f"Suppressed due to data staleness: TLE is {tle_age_days:.1f} days old (max allowed {max_tle_age_threshold}d). "
                f"Error bubble expanded {expansion_ratio}x sigma, creating a phantom alert."
            )
        elif is_safe_distance:
            decision_reason = (
                f"Safe separation: Miss distance ({miss_distance_km:.2f} km) exceeds operational safety sphere."
            )
        else:
            decision_reason = f"Negligible risk: Collision probability Pc ({raw_pc:.2e}) is below safety threshold ({pc_threshold:.1e})."
        fuel_saved = fuel_cost_usd

    return ConjunctionRiskResult(
        raw_pc=raw_pc,
        raw_pc_exponential=f"{raw_pc:.2e}",
        sigma_km=round(sigma, 3),
        sigma_expansion_ratio=expansion_ratio,
        is_false_alarm=is_false_alarm,
        is_high_uncertainty=is_high_uncertainty,
        is_safe_distance=is_safe_distance,
        classification=classification,
        decision=decision,
        decision_reason=decision_reason,
        fuel_saved_usd=fuel_saved,
    )


def batch_evaluate_conjunctions(
    candidates: List[Dict[str, Any]],
    max_tle_age_threshold: float = 2.5,
    pc_threshold: float = 1e-4,
    fuel_cost_usd: float = 800.0,
) -> Dict[str, Any]:
    """
    Evaluates a collection of conjunction candidates, categorizing false alarms vs actionable
    maneuvers and calculating operator financial ROI (thruster fuel saved).
    """
    results = []
    total_fuel_saved = 0.0
    false_alarm_count = 0
    actionable_count = 0

    for item in candidates:
        miss_km = float(item.get("miss_distance_km", 2.0))
        # Support both data_age_hours and tle_age_days
        if "data_age_hours" in item:
            age_days = float(item["data_age_hours"]) / 24.0
        else:
            age_days = float(item.get("tle_age_days", 1.0))

        eval_res = calculate_nasa_2d_gaussian_pc(
            miss_distance_km=miss_km,
            tle_age_days=age_days,
            max_tle_age_threshold=max_tle_age_threshold,
            pc_threshold=pc_threshold,
            fuel_cost_usd=fuel_cost_usd,
        )

        res_dict = {
            **item,
            "raw_pc": eval_res.raw_pc,
            "raw_pc_exp": eval_res.raw_pc_exponential,
            "sigma_km": eval_res.sigma_km,
            "sigma_expansion": eval_res.sigma_expansion_ratio,
            "is_false_alarm": eval_res.is_false_alarm,
            "classification": eval_res.classification,
            "decision": eval_res.decision,
            "decision_reason": eval_res.decision_reason,
            "fuel_saved_usd": eval_res.fuel_saved_usd,
        }
        results.append(res_dict)

        if eval_res.is_false_alarm:
            false_alarm_count += 1
            total_fuel_saved += eval_res.fuel_saved_usd
        else:
            actionable_count += 1

    total = len(candidates)
    suppressed_pct = round((false_alarm_count / total * 100.0), 1) if total > 0 else 0.0

    return {
        "total_alerts": total,
        "false_alarms_suppressed": false_alarm_count,
        "actionable_alerts": actionable_count,
        "suppression_rate_pct": suppressed_pct,
        "total_fuel_dollars_saved": total_fuel_saved,
        "candidates": results,
    }
