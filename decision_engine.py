"""Rule-based and ML-assisted decision engine for inbound logistics demo."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

MODEL_FEATURES = [
    "origin_port", "destination_port", "carrier", "mode", "criticality",
    "sku_count", "container_value_usd", "delay_days",
    "port_arrival_to_discharge_days", "discharge_to_customs_days", "customs_to_gateout_days",
    "gateout_to_delivery_days", "delivery_to_empty_return_days", "arrival_to_empty_return_days",
]


def safe_load_model(name: str):
    path = MODEL_DIR / name
    return joblib.load(path) if path.exists() else None


def scenario_table(row: pd.Series, predicted_accessorial: float) -> pd.DataFrame:
    base_risk = "High" if row["delay_days"] >= 8 else "Medium" if row["delay_days"] >= 4 else "Low"
    scenarios = [
        {
            "scenario": "Wait and monitor",
            "estimated_incremental_cost_usd": round(predicted_accessorial, 2),
            "risk": base_risk,
            "notes": "Lowest immediate action cost, but exposure can rise if empty return is delayed.",
        },
        {
            "scenario": "Pre-pull container",
            "estimated_incremental_cost_usd": round(min(predicted_accessorial * 0.55 + 650, predicted_accessorial + 250), 2),
            "risk": "Medium",
            "notes": "Can reduce terminal dwell exposure; useful when demurrage/storage risk is increasing.",
        },
        {
            "scenario": "Priority drayage",
            "estimated_incremental_cost_usd": round(min(predicted_accessorial * 0.65 + 450, predicted_accessorial + 150), 2),
            "risk": "Low-Medium",
            "notes": "Useful after gate-out/customs release when delivery and empty return timing matter.",
        },
        {
            "scenario": "Upgrade to premium option",
            "estimated_incremental_cost_usd": round(2500 + row["container_value_usd"] * 0.002, 2),
            "risk": "Low",
            "notes": "More relevant earlier in the move; best for high-criticality containers.",
        },
    ]
    return pd.DataFrame(scenarios).sort_values(["risk", "estimated_incremental_cost_usd"])


def explain_recommendation(row: pd.Series, ml_outputs: Dict, scenarios: pd.DataFrame) -> str:
    recommended = ml_outputs.get("recommended_action", "WAIT_AND_MONITOR")
    readable = recommended.replace("_", " ").title()
    risk = "high" if ml_outputs.get("delay_risk") == 1 else "moderate/low"
    accessorial = ml_outputs.get("predicted_accessorial_cost", 0.0)
    latest_status = row.get("latest_milestone_code", "UNKNOWN")

    return (
        f"Container {row['container_id']} is currently at milestone {latest_status}. "
        f"It shows {risk} delay risk with an estimated accessorial exposure of "
        f"${accessorial:,.0f}. Recommended action: {readable}. "
        f"This recommendation considers current milestone gaps, delay days, criticality, "
        f"destination port, and historical synthetic cost patterns."
    )


def make_decision(feature_row: pd.Series) -> Dict:
    X = pd.DataFrame([feature_row[MODEL_FEATURES].to_dict()])
    delay_model = safe_load_model("delay_risk_classifier.joblib")
    acc_risk_model = safe_load_model("accessorial_risk_classifier.joblib")
    acc_cost_model = safe_load_model("accessorial_cost_regressor.joblib")
    action_model = safe_load_model("action_recommender_classifier.joblib")

    delay_risk = int(delay_model.predict(X)[0]) if delay_model else int(feature_row["delay_days"] >= 7)
    accessorial_risk = int(acc_risk_model.predict(X)[0]) if acc_risk_model else int(feature_row.get("total_accessorial_cost", 0) > 750)
    predicted_accessorial = float(acc_cost_model.predict(X)[0]) if acc_cost_model else float(feature_row.get("total_accessorial_cost", 0))
    recommended_action = str(action_model.predict(X)[0]) if action_model else "WAIT_AND_MONITOR"

    outputs = {
        "delay_risk": delay_risk,
        "accessorial_risk": accessorial_risk,
        "predicted_accessorial_cost": max(0, predicted_accessorial),
        "recommended_action": recommended_action,
    }
    scenarios = scenario_table(feature_row, outputs["predicted_accessorial_cost"])
    explanation = explain_recommendation(feature_row, outputs, scenarios)
    return {"ml_outputs": outputs, "scenarios": scenarios, "explanation": explanation}
