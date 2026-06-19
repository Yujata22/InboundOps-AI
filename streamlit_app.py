"""Streamlit app for InboundOps AI.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from decision_engine import make_decision  # noqa: E402
from features import build_features, load_data  # noqa: E402

st.set_page_config(page_title="InboundOps AI", layout="wide")

@st.cache_data
def get_features():
    profiles, milestones, costs = load_data(ROOT / "data")
    features = build_features(profiles, milestones, costs)
    latest = milestones.sort_values("milestone_date").groupby("container_id").tail(1)[["container_id", "milestone_code", "milestone_date", "port_name"]]
    latest = latest.rename(columns={"milestone_code": "latest_milestone_code", "milestone_date": "latest_milestone_date", "port_name": "latest_port_name"})
    return features.merge(latest, on="container_id", how="left"), milestones, costs

features, milestones, costs = get_features()

st.title("🚢 InboundOps AI")
st.caption("A synthetic-data demo for hybrid ML + Agentic AI decision intelligence in inbound logistics.")

with st.sidebar:
    st.header("Select Container")
    container_id = st.selectbox("Container ID", sorted(features["container_id"].unique()))
    st.markdown("---")
    st.write("This demo uses fully synthetic, generic logistics data.")

row = features[features["container_id"] == container_id].iloc[0]
decision = make_decision(row)
ml = decision["ml_outputs"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Delay Days", int(row["delay_days"]))
col2.metric("Criticality", row["criticality"])
col3.metric("Predicted Accessorial", f"${ml['predicted_accessorial_cost']:,.0f}")
col4.metric("Delay Risk", "High" if ml["delay_risk"] else "Low/Medium")

st.subheader("Agent Recommendation")
st.info(decision["explanation"])

st.subheader("Scenario Comparison")
st.dataframe(decision["scenarios"], use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Container Profile")
    display_cols = [
        "container_id", "bol", "origin_port", "destination_port", "carrier", "mode",
        "sku_count", "container_value_usd", "criticality", "planned_arrival_date", "actual_arrival_date",
        "latest_milestone_code", "latest_milestone_date", "latest_port_name",
    ]
    st.dataframe(pd.DataFrame(row[display_cols]).rename(columns={row.name: "value"}), use_container_width=True)

with right:
    st.subheader("Cost Summary")
    cost_summary = costs[costs["container_id"] == container_id].groupby("cost_attribute_code", as_index=False)["cost_amount_usd"].sum()
    st.dataframe(cost_summary.sort_values("cost_amount_usd", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Milestone Timeline")
st.dataframe(
    milestones[milestones["container_id"] == container_id].sort_values("milestone_date"),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Ask the Agent")
st.caption("Try: cost exposure, demurrage risk, scenario comparison, shipment status, or next best action.")


def answer_question(question: str, row, decision, costs, milestones) -> str:
    q = question.lower().strip()
    ml = decision["ml_outputs"]
    scenarios = decision["scenarios"]
    cid = row["container_id"]

    container_costs = costs[costs["container_id"] == cid]
    cost_by_attr = (
        container_costs.groupby("cost_attribute_code")["cost_amount_usd"].sum().to_dict()
        if not container_costs.empty else {}
    )
    detention = cost_by_attr.get("DETENTION", 0.0)
    demurrage = cost_by_attr.get("DEMURRAGE", 0.0)
    storage = cost_by_attr.get("STORAGE", 0.0)
    prepull = cost_by_attr.get("PRE_PULL", 0.0)
    chassis = cost_by_attr.get("CHASSIS", 0.0)
    accessorial_total = detention + demurrage + storage + prepull + chassis + cost_by_attr.get("ACCESSORIAL", 0.0)

    timeline = milestones[milestones["container_id"] == cid].sort_values("milestone_date")
    latest = timeline.iloc[-1] if not timeline.empty else None
    latest_text = (
        f"latest milestone is {latest['milestone_code']} on {latest['milestone_date'].date()} at {latest['port_name']}"
        if latest is not None else "latest milestone is unavailable"
    )

    if any(k in q for k in ["detention", "demurrage", "exposure", "accessorial", "cost"]):
        return (
            f"For container {cid}, the synthetic historical/finalized accessorial exposure is approximately "
            f"${accessorial_total:,.0f}. Breakdown: detention ${detention:,.0f}, demurrage ${demurrage:,.0f}, "
            f"storage ${storage:,.0f}, pre-pull ${prepull:,.0f}, and chassis ${chassis:,.0f}. "
            f"The ML model estimates total accessorial exposure at ${ml['predicted_accessorial_cost']:,.0f}. "
            f"Recommended action: {ml['recommended_action'].replace('_', ' ').title()}."
        )

    if any(k in q for k in ["status", "where", "milestone", "journey", "timeline"]):
        return (
            f"Container {cid} has {len(timeline)} recorded milestones. The {latest_text}. "
            f"Arrival-to-empty-return time is {int(row['arrival_to_empty_return_days'])} days, "
            f"with {int(row['delay_days'])} delay days. Current risk is "
            f"{'High' if ml['delay_risk'] else 'Low/Medium'}."
        )

    if any(k in q for k in ["scenario", "compare", "wait", "pre-pull", "prepull", "priority", "premium"]):
        top = scenarios.sort_values("estimated_incremental_cost_usd").head(3)
        lines = []
        for _, s in top.iterrows():
            lines.append(
                f"- {s['scenario']}: estimated incremental cost ${s['estimated_incremental_cost_usd']:,.0f}, "
                f"risk {s['risk']}. {s['notes']}"
            )
        return "Scenario comparison for " + cid + ":\n" + "\n".join(lines)

    if any(k in q for k in ["risk", "delay", "dwell"]):
        return (
            f"Container {cid} shows {'high' if ml['delay_risk'] else 'low/medium'} delay risk. "
            f"Key signals: {int(row['delay_days'])} delay days, "
            f"{int(row['port_arrival_to_discharge_days'])} days from destination arrival to discharge, "
            f"{int(row['discharge_to_customs_days'])} days from discharge to customs release, and "
            f"{int(row['delivery_to_empty_return_days'])} days from delivery to empty return."
        )

    if any(k in q for k in ["recommend", "next best", "what should", "action", "do next"]):
        return decision["explanation"]

    return (
        f"For container {cid}: {decision['explanation']} You can also ask about cost exposure, "
        f"detention/demurrage, shipment status, risk, or scenario comparison."
    )


user_question = st.text_input("Ask a question about the selected container")
if user_question:
    st.markdown(answer_question(user_question, row, decision, costs, milestones))
