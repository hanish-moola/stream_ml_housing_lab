# 🏠 Stream-ML-Housing-Lab
**A production-grade, on-demand ML pipeline that streams housing data, predicts prices in real time, and stress-tests itself with multi-agent chaos drills.**

---

## 🚀 Overview
`Stream-ML-Housing-Lab` turns the static [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) into a living, breathing **real-time ML system**.

It ingests streaming housing data, transforms it into features, and serves instant price predictions via an MLflow-registered model.  
The project then evolves into a **multi-agent simulation**, where autonomous agents stress test the system and even simulate **bidding behavior** on houses.

> 🧩 This repo doubles as both a *learning sandbox* for production ML pipelines and a *research lab* for agentic MLOps systems.

---

## 🧠 Architecture
Producer → Kafka (raw_housing)
            ↓
        Feature Processor (Faust/Consumer)
            ↓
        Inference Service (FastAPI + MLflow)
            ↓
        Predictions Topic / REST Response



| Layer | Description |
|-------|--------------|
| **Ingest** | Streams raw housing events into Kafka topics |
| **Feature Service** | Shared offline/online transforms ensure feature parity |
| **Model Serving** | FastAPI app serves predictions using latest MLflow model |
| **Observability** | Prometheus + Grafana dashboards track latency, drift, and SLOs |
| **CI/CD** | GitHub Actions retrain, validate, and canary new models automatically |

---

## 🧩 Key Components

### 🔹 Core ML Pipeline
- **Training:** XGBoost regressor wrapped in a scikit-learn pipeline  
- **Registry:** MLflow model tracking + versioned deployment  
- **Feature parity:** Shared transformation logic across offline and online flows

### 🔹 Streaming Infrastructure
- **Kafka** for event ingestion and pub/sub  
- **Redis** (optional) for feature caching  
- **FastAPI** endpoint `/predict` for on-demand inference

### 🔹 Observability & Quality
- **Prometheus / Grafana** dashboards  
- **Great Expectations** for data validation  
- **Drift detection** using PSI & KS statistics  
- **Latency SLOs** and automated guardrails

---

## 🧪 Multi-Agent Stress Framework
Once the core pipeline is live, **agents** simulate realistic production chaos:

| Agent | Role |
|--------|------|
| 🧮 **Traffic** | Generates variable RPS patterns (burst, soak, ramp) |
| 🌀 **Drift** | Alters feature distributions (income, rooms, population) |
| 💥 **Fault Injector** | Simulates latency, dropped events, cache failures |
| 📊 **Observer** | Monitors SLOs, PSI, and drift metrics |
| 🧭 **Guardrail** | Enforces rollback, alert, and canary rules |
| 🧠 **Commander** | Runs scenario YAMLs and generates post-run reports |

Each stress run produces a **structured run report** with metrics, drift charts, and guardrail decisions.

---

## 💰 Bid Simulation Engine (Phase 3)
In the final phase, autonomous bidder agents use the model’s price estimate and uncertainty to simulate **property auctions**.

| Persona | Behavior |
|----------|-----------|
| 💼 Investor | Conservative bids based on ROI & yield |
| 🛠️ Flipper | Aggressive, short-term ROI-driven bidding |
| 🏡 Owner-Occupier | Emotionally weighted bids above estimate |
| 📉 Opportunist | Lowball bids below predicted median |

This enables studying **market dynamics under predictive bias** — how model error influences pricing behavior.

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.11 |
| **Frameworks** | FastAPI · Faust · LangGraph (agents) |
| **ML** | scikit-learn · XGBoost · MLflow |
| **Data Validation** | Great Expectations |
| **Infra** | Docker · Kafka · Redis |
| **Observability** | Prometheus · Grafana · OpenTelemetry |
| **CI/CD** | GitHub Actions · pytest |

---

---

## 🧩 Example Inference Call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Avg_Area_Income": 65000,
    "Avg_Area_House_Age": 7.5,
    "Avg_Area_Number_of_Rooms": 6.0,
    "Avg_Area_Number_of_Bedrooms": 3.0,
    "Area_Population": 42000,
    "Address": "123 Main St, Austin, TX 78701"
  }'

Response
{"estimated_price": 456732.18}


🧠 Future Roadmap

 Deploy to AWS Lambda + API Gateway

 Integrate Feast feature store

 Add shadow deployments + canary rollback

 Reinforcement learning for bidder agents

 Streamlit dashboard for live simulation

🧾 License

Apache 2.0

👤 Author

Hanish Moola
Engineering Manager · ML & AI Systems Architect
💼 LinkedIn • 🧠 GitHub