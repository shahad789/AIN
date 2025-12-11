# AI Identity Security Engine  
A modern security analytics platform that detects risky behavior sessions by combining **rule-based signals**, **machine-learning anomaly detection**, and **AI-generated reasoning**.

This engine is designed to operate with **User & Entity Behavior Analytics (UEBA)** and **Identity Threat Detection** module. It identifies identity-based attacks such as:

- 🚨 Impossible travel  
- 🚨 Credential stuffing  
- 🚨 Account takeover  
- 🚨 Session hijacking  
- 🚨 Malicious device switching  
- 🚨 Behavioral anomalies (time, location, service access)

# ⭐ Key Features

### ✔ Hybrid scoring engine (Rules + ML)  

### ✔ IsolationForest anomaly detection  

### ✔ Real-time session risk scoring  

### ✔ API backend + Interactive frontend  
Backend: **FastAPI**  
Frontend: **React + Tailwind**  
Communication via `/score-session` endpoint.

# 📁 Project Architecture Overview

```
ain-security-master/
│
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── api/
│   │   └── routes.py       # Endpoints
│   ├── ml/
│   │   ├── model_loader.py # Loads IsolationForest
│   │   ├── anomaly.py      # ML scoring logic
│   │   └── features.py     # Feature engineering
│   ├── risk_engine/
│   │   ├── rules.py        # Rule-based scoring
│   │   └── combine.py      # Rule + AI combined score
│   └── utils/
│       └── geo.py          # Distance calculations, IP, etc.
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── models/                 # ML models (IsolationForest)
├── llm_openai.py           # AI explanation logic
├── requirements.txt
├── run.bat / run.sh
└── .env
```

The system is modular — every component is replaceable.


# 🧠 OpenAI Integration

The system generates a natural-language explanation for every session:

- What signals were detected  
- Why the ML flagged anomaly  
- Attack likelihood  
- SOC recommended response  

# 🛠 How to Run

install streamlit

## Backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## Frontend

```bash
streamlit run frontend/app_unified.py


# 📌 Conclusion

This system provides a full end-to-end identity threat detection solution with:

- **ML anomaly scoring**  
- **Rule-based detections**  
- **AI-powered incident explanations**  
- **Frontend UI for testing scenarios**  
- **Backend APIs for integration**  


