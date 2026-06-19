# 🚢 InboundOPS-AI

## From Container Delays to Decisions: A Hybrid ML + Agentic AI Copilot for Inbound Logistics

InboundOPS-AI is a **Decision Intelligence framework** designed to support **global inbound logistics operations**. The project combines **Machine Learning, business rules, and agentic reasoning** to predict operational risk, estimate cost exposure, and recommend the next best action for inbound shipments.

The objective is simple:

> **Transform inbound logistics from dashboard-driven monitoring to predictive, explainable, and conversation-driven decision intelligence.**

---

## 📌 Business Problem

Global inbound logistics teams constantly manage uncertainty across the journey from **origin pickup, port movement, vessel transit, customs clearance, destination arrival, and empty container return**.

When a container gets delayed, logistics operations teams are often faced with questions such as:

* Should we continue with standard ocean movement, upgrade to premium movement, or expedite via air?
* Which shipments justify premium transportation costs?
* Is the shipment likely to incur detention or demurrage charges?
* Should the container be rerouted, prioritized, or escalated?
* What is the cost of action versus the cost of inaction?

Answering these questions typically requires navigating multiple dashboards, manually stitching together shipment events and cost information, and relying heavily on institutional knowledge.

InboundOPS-AI addresses these challenges by introducing an explainable decision intelligence layer capable of proactively assessing shipment risk and recommending next-best actions.

---

## 🏗️ Solution Overview

InboundOPS-AI leverages:

* Historical shipment milestone data
* Transportation and accessorial cost signals
* Machine Learning-based risk prediction
* Scenario simulation
* Agentic reasoning
* Natural language interaction

The platform provides:

✅ Shipment status summarization

✅ Delay and operational risk prediction

✅ Accessorial cost exposure estimation

✅ Scenario comparison

✅ Explainable next-best-action recommendations

---

## 🏛️ System Architecture

![Architecture](images/architecture.png)

---

## 🔄 End-to-End Workflow

```text
Synthetic Shipment & Cost Data
                │
                ▼
Feature Engineering Layer
(Container Dwell Time, Milestone Gaps,
Delay Computation, Cost Aggregation)
                │
                ▼
Machine Learning Layer
├── Delay Risk Prediction Model
├── Cost Exposure Prediction Model
└── Action Recommendation Model
                │
                ▼
Decision Intelligence Engine
├── Shipment Context Agent
├── Cost Exposure Agent
├── Risk Assessment Agent
├── Scenario Simulation Agent
└── Recommendation Agent
                │
                ▼
Streamlit Conversational Interface
                │
                ▼
Explainable Next-Best Actions
```

---

## 📂 Repository Structure

```text
InboundOPS-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── shipment_milestones.csv
│   ├── shipment_costs.csv
│   └── container_profile.csv
│
├── models/
│   ├── delay_model.pkl
│   └── cost_model.pkl
│
├── src/
│   ├── train_models.py
│   ├── decision_engine.py
│   └── utils.py
│
├── images/
│   ├── architecture.png
│   ├── demo.gif
│   └── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Machine Learning Components

### 1. Delay Risk Prediction

Predicts the likelihood of shipment delays based on historical milestone patterns.

**Example Features**

* Milestone gaps
* Port dwell time
* Carrier behavior
* Historical delay patterns
* Transit variability

---

### 2. Cost Exposure Prediction

Estimates potential cost exposure associated with:

* Detention
* Demurrage
* Storage
* Chassis
* Pre-pull
* Other accessorial charges

---

### 3. Recommendation Engine

Evaluates multiple operational scenarios including:

* Continue Standard Ocean Movement
* Upgrade to Premium Movement
* Pre-Pull Container
* Priority Drayage
* Escalate Shipment

The engine recommends the next best action by balancing:

* Cost
* Risk
* Operational constraints
* Business impact

---

## 💬 Example Questions

Users can interact with the system using natural language:

> What is the current status of container CONT10234?

> Container CONT10234 has been delayed at destination port. What should we do next?

> Estimate detention and demurrage exposure for container CONT10234.

> Compare waiting versus pre-pull for container CONT10234.

> Why was pre-pull recommended for container CONT10234?

> Which shipments should logistics operations teams prioritize today?

---

## ⚙️ Technology Stack

| Layer                | Technology    |
| -------------------- | ------------- |
| Programming Language | Python        |
| Frontend             | Streamlit     |
| Data Processing      | Pandas, NumPy |
| Machine Learning     | Scikit-learn  |
| Model Persistence    | Joblib        |
| Visualization        | Streamlit     |
| Version Control      | Git, GitHub   |

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/<your-username>/InboundOPS-AI.git
cd InboundOPS-AI
```

### Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train machine learning models

```bash
python3 src/train_models.py
```

### Launch Streamlit application

```bash
python3 -m streamlit run app/streamlit_app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 📊 Synthetic Data Notice

This repository uses **fully synthetic shipment and cost datasets** generated exclusively for demonstration purposes.

No proprietary, confidential, customer-specific, or employer-owned information has been used in the development of this project.

The synthetic datasets are intended solely to demonstrate the decision intelligence framework and its applicability to real-world inbound logistics operations.

---

## 🔮 Future Enhancements

* Integration with LLM-powered conversational agents
* Multi-agent orchestration using LangGraph or CrewAI
* Retrieval-Augmented Generation (RAG) over logistics SOP documentation
* Real-time shipment event ingestion
* External congestion and weather signal integration
* Cloud deployment on AWS
* Optimization-driven transportation mode selection
* Human-in-the-loop decision workflows

---
