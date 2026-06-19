"""Train simple local ML models for the InboundOps AI demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import build_features, load_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["origin_port", "destination_port", "carrier", "mode", "criticality"]
NUMERIC = [
    "sku_count",
    "container_value_usd",
    "delay_days",
    "port_arrival_to_discharge_days",
    "discharge_to_customs_days",
    "customs_to_gateout_days",
    "gateout_to_delivery_days",
    "delivery_to_empty_return_days",
    "arrival_to_empty_return_days",
]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )


def train_classifier(df: pd.DataFrame, target: str, model_name: str):
    X = df[CATEGORICAL + NUMERIC]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y if y.nunique() < 8 else None)
    pipe = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, preds))
    joblib.dump(pipe, MODEL_DIR / f"{model_name}.joblib")


def train_regressor(df: pd.DataFrame):
    X = df[CATEGORICAL + NUMERIC]
    y = df["total_accessorial_cost"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
    pipe = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("\n=== accessorial_cost_regressor ===")
    print(f"MAE: ${mean_absolute_error(y_test, preds):,.2f}")
    joblib.dump(pipe, MODEL_DIR / "accessorial_cost_regressor.joblib")


def main():
    profiles, milestones, costs = load_data()
    df = build_features(profiles, milestones, costs)
    df.to_csv(DATA_DIR / "model_features.csv", index=False)
    train_classifier(df, "delay_risk_label", "delay_risk_classifier")
    train_classifier(df, "has_accessorial_risk", "accessorial_risk_classifier")
    train_classifier(df, "recommended_action_label", "action_recommender_classifier")
    train_regressor(df)


if __name__ == "__main__":
    main()
