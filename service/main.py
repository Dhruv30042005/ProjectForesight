from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.risk import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    risk_level,
    safe_percentage,
)


app = FastAPI(
    title="FORESIGHT Scoring Service",
    description="Inventory risk scoring API for demand and inventory intelligence",
    version="1.0.0",
)


class ScoreRequest(BaseModel):
    current_inventory: float = Field(ge=0)
    units_ordered: float = Field(ge=0)
    current_price: float = Field(ge=0)
    forecast_8_weeks: list[float] = Field(min_length=1, max_length=8)
    lead_time_days: int = Field(default=14, ge=1)


def calculate_score(data: ScoreRequest):
    forecast = [max(float(x), 0) for x in data.forecast_8_weeks]

    forward_demand = sum(forecast)

    weeks_needed = max(1, (data.lead_time_days + 6) // 7)

    lead_time_demand = sum(forecast[:weeks_needed])

    available_supply = (
        data.current_inventory + data.units_ordered
    )

    stockout_gap = max(
        lead_time_demand - available_supply,
        0
    )

    if lead_time_demand > 0:
        stockout_score = (
            stockout_gap / lead_time_demand
        ) * 100
    else:
        stockout_score = 0

    stockout_score = round(
        safe_percentage(stockout_score), 2
    )

    excess_inventory = max(
        data.current_inventory - forward_demand,
        0
    )

    if data.current_inventory > 0:
        overstock_score = (
            excess_inventory / data.current_inventory
        ) * 100
    else:
        overstock_score = 0

    overstock_score = round(
        safe_percentage(overstock_score), 2
    )

    stockout_level = risk_level(stockout_score)
    overstock_level = risk_level(overstock_score)

    stockout_impact = (
        stockout_gap * data.current_price
    )

    overstock_impact = (
        excess_inventory * data.current_price
    )

    total_impact = (
        stockout_impact + overstock_impact
    )

    stockout_high = (
        stockout_score >= HIGH_RISK_THRESHOLD
    )

    overstock_high = (
        overstock_score >= HIGH_RISK_THRESHOLD
    )

    if stockout_high and not overstock_high:
        decision = "Reorder Now"
        action = (
            "Prioritize replenishment; available "
            "supply may not cover lead-time demand."
        )

    elif overstock_high and not stockout_high:
        decision = "Markdown / Clear"
        action = (
            "Review excess inventory and consider "
            "markdown or clearance action."
        )

    elif stockout_high and overstock_high:
        decision = "Watch / Volatile"
        action = (
            "Monitor closely; both stockout and "
            "overstock signals require review."
        )

    else:
        decision = "Healthy"
        action = (
            "Maintain current inventory position "
            "and continue monitoring."
        )

    return {
        "forward_8_week_demand": round(forward_demand, 2),
        "lead_time_demand": round(lead_time_demand, 2),
        "available_supply": round(available_supply, 2),
        "stockout_gap_units": round(stockout_gap, 2),
        "stockout_risk_score": stockout_score,
        "stockout_risk_level": stockout_level,
        "excess_inventory_units": round(excess_inventory, 2),
        "overstock_risk_score": overstock_score,
        "overstock_risk_level": overstock_level,
        "estimated_stockout_impact_inr": round(
            stockout_impact, 2
        ),
        "estimated_overstock_impact_inr": round(
            overstock_impact, 2
        ),
        "estimated_impact_inr": round(
            total_impact, 2
        ),
        "decision_quadrant": decision,
        "recommended_action": action,
        "assumed_lead_time_days": data.lead_time_days,
    }


@app.get("/")
def root():
    return {
        "service": "FORESIGHT Scoring Service",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/score")
def score_inventory(data: ScoreRequest):
    try:
        return calculate_score(data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )