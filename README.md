# 🛡️ Network Intrusion Detection System

Real-time AI-powered network intrusion detection system that classifies 14 cyberattack types from network traffic flows.

## 🏆 Performance
| Metric | Score |
|--------|-------|
| Accuracy | 99.89% |
| Macro F1 | 0.9027 |
| Throughput | 28,531 flows/sec |
| Attack Classes | 14 |

## 🏗️ Architecture
```
Network Traffic → Feature Engineering → Ensemble Model → Threat Detection
                                              ↓
                                      Drift Detection
                                              ↓
                                    AI Threat Analyst (LLaMA3)
                                              ↓
                                    FastAPI + Streamlit Dashboard
```

## 📦 Tech Stack
- **ML Models:** Random Forest + XGBoost Ensemble
- **Explainability:** SHAP values
- **Drift Detection:** Custom ADWIN-style detector
- **AI Analyst:** Groq LLaMA3-70B
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly
- **Dataset:** CICIDS 2017 (2.8M flows, 14 attack types)

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/rishabhsingh7822/network-ids-project.git
cd network-ids-project
```

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download dataset
Download CICIDS 2017 from: https://www.kaggle.com/datasets/cicdataset/cicids2017
Place CSV files in `data/raw/`

### 4. Run pipeline
```bash
python src/ingestion/load_data.py
python src/preprocessing/split_and_balance.py
python src/preprocessing/feature_engineer.py
python src/preprocessing/pipeline.py
python src/models/train_random_forest.py
python src/models/train_xgboost.py
python src/models/ensemble.py
```

### 5. Start API
```bash
cd api
python -m uvicorn main:app --port 8080
```

### 6. Start Dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Docker
```bash
docker-compose up --build
```

## 📁 Project Structure
```
network-ids-project/
├── data/
│   ├── raw/          # CICIDS 2017 CSV files (not tracked)
│   └── processed/    # Cleaned parquet, models (not tracked)
├── src/
│   ├── ingestion/    # Data loading
│   ├── preprocessing/# Feature engineering, SMOTE, pipeline
│   ├── models/       # RF, XGBoost, Ensemble, SHAP
│   ├── detection/    # Stream simulator, drift detector
│   └── ai_analyst/   # Groq LLaMA3 threat briefs
├── api/              # FastAPI REST API
├── dashboard/        # Streamlit dashboard
├── tests/            # Unit tests
└── docs/             # Charts and visualizations
```

## 🎯 Key Design Decisions
1. **SMOTE after split** — prevents data leakage
2. **Macro F1 over accuracy** — handles class imbalance
3. **Drift detection** — production-ready, not just a notebook
4. **Rule-based fallback** — system works even if LLM API is down

## 👨‍💻 Author
Rajesh Singh