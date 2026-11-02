"""
FastAPI Backend API for Space Debris Intelligence & Conjunction Risk Engine.
Provides REST endpoints for orbital risk calculations, false alarm suppression,
and operator fuel savings analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from pathlib import Path

from src.orbits.conjunction_risk import (
    ConjunctionInput,
    ConjunctionRiskResult,
    calculate_nasa_2d_gaussian_pc,
    batch_evaluate_conjunctions,
)

app = FastAPI(
    title="Space Debris Intelligence & Conjunction Risk API",
    description="Smart Risk Filter for Small Satellite Operators (NASA 2D Gaussian + TLE Age Penalty)",
    version="1.0.0",
)

# Enable CORS for frontend clients / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Space Debris Intelligence Backend",
        "mission": "Smart Risk Filtering & False Alarm Suppression for Satellite Operators",
    }


@app.post("/api/conjunction/calculate", response_model=ConjunctionRiskResult)
def evaluate_single_conjunction(payload: ConjunctionInput):
    """
    Calculate 2D Gaussian Collision Probability (Pc) and evaluate whether
    the encounter is actionable danger or a false alarm triggered by aged TLE covariance.
    """
    return calculate_nasa_2d_gaussian_pc(
        miss_distance_km=payload.miss_distance_km,
        tle_age_days=payload.tle_age_days,
        max_tle_age_threshold=payload.max_tle_age_threshold_days,
        pc_threshold=payload.pc_critical_threshold,
        fuel_cost_usd=payload.fuel_cost_usd_per_maneuver,
    )


@app.get("/api/conjunction/candidates")
def get_processed_candidates(
    max_tle_age: float = 2.5,
    pc_threshold: float = 1e-4,
    fuel_cost_usd: float = 800.0,
):
    """
    Load current scored candidates dataset and evaluate false alarm suppression and fuel savings.
    """
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Scored candidates dataset not found. Please run the screening pipeline first.",
        )

    df = pd.read_csv(data_path)
    records = df.to_dict(orient="records")

    summary = batch_evaluate_conjunctions(
        candidates=records,
        max_tle_age_threshold=max_tle_age,
        pc_threshold=pc_threshold,
        fuel_cost_usd=fuel_cost_usd,
    )
    return summary


@app.get("/api/metrics/summary")
def get_executive_metrics(
    max_tle_age: float = 2.5,
    pc_threshold: float = 1e-4,
    fuel_cost_usd: float = 800.0,
):
    """
    Returns executive metrics: Total Alerts, Suppressed False Alarms, Actionable Alerts, Fuel Saved ($).
    """
    data_path = Path("data/processed/scored_candidates.csv")
    if not data_path.exists():
        return {
            "total_alerts": 0,
            "false_alarms_suppressed": 0,
            "actionable_alerts": 0,
            "suppression_rate_pct": 0.0,
            "total_fuel_dollars_saved": 0.0,
        }

    df = pd.read_csv(data_path)
    records = df.to_dict(orient="records")
    summary = batch_evaluate_conjunctions(
        candidates=records,
        max_tle_age_threshold=max_tle_age,
        pc_threshold=pc_threshold,
        fuel_cost_usd=fuel_cost_usd,
    )
    return {
        "total_alerts": summary["total_alerts"],
        "false_alarms_suppressed": summary["false_alarms_suppressed"],
        "actionable_alerts": summary["actionable_alerts"],
        "suppression_rate_pct": summary["suppression_rate_pct"],
        "total_fuel_dollars_saved": summary["total_fuel_dollars_saved"],
    }
