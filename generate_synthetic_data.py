"""Generate synthetic inbound logistics data for demo purposes.

This data is fully synthetic and intentionally generic. It is not based on any
proprietary company data, operational process, or internal system.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTS_ORIGIN = ["Shanghai", "Ningbo", "Yantian", "Busan", "Ho Chi Minh", "Singapore"]
PORTS_DEST = ["Los Angeles", "Long Beach", "Seattle", "New York/New Jersey", "Savannah", "Houston"]
CARRIERS = ["BlueWave", "OceanBridge", "GlobalMar", "NorthStar", "PacificLine"]
MODES = ["STANDARD_OCEAN", "PREMIUM_OCEAN"]
CRITICALITY = ["LOW", "MEDIUM", "HIGH"]

MILESTONES = [
    "ORIGIN_GATE_IN",
    "LOADED_ON_VESSEL",
    "VESSEL_DEPARTED",
    "DESTINATION_PORT_ARRIVAL",
    "VESSEL_DISCHARGE",
    "CUSTOMS_RELEASED",
    "DESTINATION_GATE_OUT",
    "DELIVERED_TO_NODE",
    "EMPTY_RETURNED",
]

BASE_COST_BY_MODE = {
    "STANDARD_OCEAN": (2400, 5200),
    "PREMIUM_OCEAN": (4800, 9500),
}

ACCESSORIAL_CODES = ["DETENTION", "DEMURRAGE", "STORAGE", "PRE_PULL", "CHASSIS", "ACCESSORIAL"]


def random_container_id(i: int) -> str:
    return f"CONT{1000000 + i}"


def random_bol(i: int) -> str:
    return f"BOL{2024}{100000 + i}"


def create_container_profiles(n: int = 1400) -> pd.DataFrame:
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days

    rows = []
    for i in range(n):
        mode = np.random.choice(MODES, p=[0.82, 0.18])
        criticality = np.random.choice(CRITICALITY, p=[0.45, 0.38, 0.17])
        origin = random.choice(PORTS_ORIGIN)
        dest = random.choice(PORTS_DEST)
        carrier = random.choice(CARRIERS)
        planned_departure = start_date + timedelta(days=random.randint(0, total_days - 55))
        planned_transit_days = random.randint(18, 42) if mode == "STANDARD_OCEAN" else random.randint(14, 31)
        planned_arrival = planned_departure + timedelta(days=planned_transit_days)

        # Port/carrier driven synthetic delay patterns.
        port_delay_bias = {
            "Los Angeles": 2,
            "Long Beach": 3,
            "Seattle": 1,
            "New York/New Jersey": 2,
            "Savannah": 1,
            "Houston": 2,
        }[dest]
        carrier_bias = CARRIERS.index(carrier) % 3
        random_delay = max(0, int(np.random.normal(loc=port_delay_bias + carrier_bias, scale=4)))
        if np.random.random() < 0.12:
            random_delay += random.randint(6, 18)
        if mode == "PREMIUM_OCEAN":
            random_delay = max(0, random_delay - random.randint(1, 4))

        rows.append(
            {
                "container_id": random_container_id(i),
                "bol": random_bol(i),
                "origin_port": origin,
                "destination_port": dest,
                "carrier": carrier,
                "mode": mode,
                "sku_count": random.randint(8, 140),
                "container_value_usd": round(float(np.random.lognormal(mean=10.7, sigma=0.55)), 2),
                "criticality": criticality,
                "planned_departure_date": planned_departure.date().isoformat(),
                "planned_arrival_date": planned_arrival.date().isoformat(),
                "actual_arrival_date": (planned_arrival + timedelta(days=random_delay)).date().isoformat(),
                "delay_days": random_delay,
            }
        )
    return pd.DataFrame(rows)


def create_milestones(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in profiles.iterrows():
        planned_departure = pd.to_datetime(r["planned_departure_date"])
        actual_arrival = pd.to_datetime(r["actual_arrival_date"])
        origin_gate = planned_departure - timedelta(days=random.randint(3, 9))
        loaded = planned_departure - timedelta(days=random.randint(1, 3))
        departed = planned_departure
        port_arrival = actual_arrival
        discharge = port_arrival + timedelta(days=random.randint(0, 3))
        customs = discharge + timedelta(days=random.randint(0, 5))
        gate_out = customs + timedelta(days=random.randint(0, 5))
        delivered = gate_out + timedelta(days=random.randint(0, 4))
        empty_return = delivered + timedelta(days=random.randint(1, 10))

        dates = [origin_gate, loaded, departed, port_arrival, discharge, customs, gate_out, delivered, empty_return]
        ports = [r["origin_port"]] * 3 + [r["destination_port"]] * 6
        for code, date, port in zip(MILESTONES, dates, ports):
            rows.append(
                {
                    "container_id": r["container_id"],
                    "bol": r["bol"],
                    "milestone_code": code,
                    "milestone_date": date.date().isoformat(),
                    "port_name": port,
                }
            )
    return pd.DataFrame(rows)


def create_costs(profiles: pd.DataFrame, milestones: pd.DataFrame) -> pd.DataFrame:
    empty_dates = milestones[milestones["milestone_code"] == "EMPTY_RETURNED"][["container_id", "milestone_date"]]
    empty_dates = empty_dates.rename(columns={"milestone_date": "empty_return_date"})
    df = profiles.merge(empty_dates, on="container_id", how="left")
    rows = []
    for _, r in df.iterrows():
        base_low, base_high = BASE_COST_BY_MODE[r["mode"]]
        base = random.uniform(base_low, base_high)
        fuel = base * random.uniform(0.09, 0.22)
        delay_days = int(r["delay_days"])
        criticality_mult = {"LOW": 0.85, "MEDIUM": 1.0, "HIGH": 1.2}[r["criticality"]]
        free_time = random.randint(3, 6)
        chargeable_days = max(0, delay_days - free_time)

        cost_items = [("BASE", base), ("FUEL", fuel)]

        if chargeable_days > 0:
            if np.random.random() < min(0.85, 0.25 + chargeable_days * 0.08):
                cost_items.append(("DEMURRAGE", chargeable_days * random.uniform(95, 210) * criticality_mult))
            if np.random.random() < min(0.75, 0.20 + chargeable_days * 0.07):
                cost_items.append(("DETENTION", chargeable_days * random.uniform(80, 180) * criticality_mult))
            if np.random.random() < 0.35:
                cost_items.append(("STORAGE", chargeable_days * random.uniform(40, 120)))

        if np.random.random() < 0.28:
            cost_items.append(("PRE_PULL", random.uniform(350, 1250)))
        if np.random.random() < 0.62:
            cost_items.append(("CHASSIS", random.uniform(120, 650)))
        if np.random.random() < 0.18:
            cost_items.append(("ACCESSORIAL", random.uniform(90, 850)))

        for code, amount in cost_items:
            rows.append(
                {
                    "container_id": r["container_id"],
                    "bol": r["bol"],
                    "empty_return_date": r["empty_return_date"],
                    "cost_attribute_code": code,
                    "cost_amount_usd": round(float(amount), 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    profiles = create_container_profiles()
    milestones = create_milestones(profiles)
    costs = create_costs(profiles, milestones)

    profiles.to_csv(DATA_DIR / "container_profiles.csv", index=False)
    milestones.to_csv(DATA_DIR / "shipment_milestones.csv", index=False)
    costs.to_csv(DATA_DIR / "shipment_costs.csv", index=False)

    print(f"Generated {len(profiles):,} containers")
    print(f"Generated {len(milestones):,} milestone rows")
    print(f"Generated {len(costs):,} cost rows")


if __name__ == "__main__":
    main()
