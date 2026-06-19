# InboundOps AI

A GitHub-ready demo project for a **hybrid Machine Learning + Agentic AI Decision Intelligence system** for inbound logistics.

This project uses **fully synthetic data**. It is not based on any proprietary company data, internal workflow, or confidential operational system.

## What it demonstrates

- Synthetic inbound container milestone data across two years
- Synthetic cost data after empty container return
- Feature engineering from shipment milestone gaps and cost attributes
- ML models for delay risk, accessorial risk, accessorial cost prediction, and action recommendation
- Agent-style decision workflow for scenario comparison and next-best-action recommendation
- Streamlit app for an interactive local demo

## Project Structure

```text
inboundops_ai/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── container_profiles.csv
│   ├── shipment_milestones.csv
│   ├── shipment_costs.csv
│   └── model_features.csv   # created after training
├── models/                  # created after training
├── src/
│   ├── generate_synthetic_data.py
│   ├── features.py
│   ├── train_models.py
│   └── decision_engine.py
├── requirements.txt
└── README.md
```

## Dataset Overview

### Shipment Milestones

Columns:

- `container_id`
- `bol`
- `milestone_code`
- `milestone_date`
- `port_name`

Generic milestone codes include:

- `ORIGIN_GATE_IN`
- `LOADED_ON_VESSEL`
- `VESSEL_DEPARTED`
- `DESTINATION_PORT_ARRIVAL`
- `VESSEL_DISCHARGE`
- `CUSTOMS_RELEASED`
- `DESTINATION_GATE_OUT`
- `DELIVERED_TO_NODE`
- `EMPTY_RETURNED`

### Shipment Costs

Columns:

- `container_id`
- `bol`
- `empty_return_date`
- `cost_attribute_code`
- `cost_amount_usd`

Generic cost attributes include:

- `BASE`
- `FUEL`
- `DETENTION`
- `DEMURRAGE`
- `STORAGE`
- `PRE_PULL`
- `CHASSIS`
- `ACCESSORIAL`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Synthetic Data

```bash
python src/generate_synthetic_data.py
```

## Train Models

```bash
python src/train_models.py
```

This will save model artifacts into the `models/` folder.

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```
