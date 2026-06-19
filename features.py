"""Feature engineering for the InboundOps AI demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_data(data_dir: Path = DATA_DIR):
    profiles = pd.read_csv(data_dir / "container_profiles.csv", parse_dates=["planned_departure_date", "planned_arrival_date", "actual_arrival_date"])
    milestones = pd.read_csv(data_dir / "shipment_milestones.csv", parse_dates=["milestone_date"])
    costs = pd.read_csv(data_dir / "shipment_costs.csv", parse_dates=["empty_return_date"])
    return profiles, milestones, costs


def build_features(profiles: pd.DataFrame, milestones: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    pivot = milestones.pivot_table(
        index=["container_id", "bol"],
        columns="milestone_code",
        values="milestone_date",
        aggfunc="max",
    ).reset_index()

    df = profiles.merge(pivot, on=["container_id", "bol"], how="left")

    # Milestone gap features. These are generic elapsed-time signals.
    df["port_arrival_to_discharge_days"] = (df["VESSEL_DISCHARGE"] - df["DESTINATION_PORT_ARRIVAL"]).dt.days
    df["discharge_to_customs_days"] = (df["CUSTOMS_RELEASED"] - df["VESSEL_DISCHARGE"]).dt.days
    df["customs_to_gateout_days"] = (df["DESTINATION_GATE_OUT"] - df["CUSTOMS_RELEASED"]).dt.days
    df["gateout_to_delivery_days"] = (df["DELIVERED_TO_NODE"] - df["DESTINATION_GATE_OUT"]).dt.days
    df["delivery_to_empty_return_days"] = (df["EMPTY_RETURNED"] - df["DELIVERED_TO_NODE"]).dt.days
    df["arrival_to_empty_return_days"] = (df["EMPTY_RETURNED"] - df["DESTINATION_PORT_ARRIVAL"]).dt.days

    cost_pivot = costs.pivot_table(
        index=["container_id", "bol"],
        columns="cost_attribute_code",
        values="cost_amount_usd",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    df = df.merge(cost_pivot, on=["container_id", "bol"], how="left").fillna(0)

    for col in ["BASE", "FUEL", "DETENTION", "DEMURRAGE", "STORAGE", "PRE_PULL", "CHASSIS", "ACCESSORIAL"]:
        if col not in df.columns:
            df[col] = 0.0

    df["total_accessorial_cost"] = df[["DETENTION", "DEMURRAGE", "STORAGE", "PRE_PULL", "CHASSIS", "ACCESSORIAL"]].sum(axis=1)
    df["total_cost"] = df[["BASE", "FUEL", "DETENTION", "DEMURRAGE", "STORAGE", "PRE_PULL", "CHASSIS", "ACCESSORIAL"]].sum(axis=1)
    df["has_accessorial_risk"] = (df["total_accessorial_cost"] > 750).astype(int)
    df["delay_risk_label"] = (df["delay_days"] >= 7).astype(int)

    def action_label(row):
        if row["delay_days"] >= 14 and row["criticality"] == "HIGH":
            return "ESCALATE_TO_OPS"
        if row["DEMURRAGE"] + row["DETENTION"] > 1200:
            return "PRE_PULL_OR_PRIORITY_DRAYAGE"
        if row["delay_days"] >= 8 and row["criticality"] in ["MEDIUM", "HIGH"]:
            return "UPGRADE_TO_PREMIUM_OPTION"
        return "WAIT_AND_MONITOR"

    df["recommended_action_label"] = df.apply(action_label, axis=1)
    return df


if __name__ == "__main__":
    profiles, milestones, costs = load_data()
    features = build_features(profiles, milestones, costs)
    features.to_csv(DATA_DIR / "model_features.csv", index=False)
    print(f"Saved features: {features.shape}")
