import numpy as np
# pyrefly: ignore [missing-import]
import joblib
import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sys
from contextlib import asynccontextmanager

BASE_DIR      = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from src.ai_analyst.threat_analyst import get_groq_brief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR    = BASE_DIR / 'src/models'

rf = xgb = le = pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rf, xgb, le, pipeline
    logger.info("Loading models...")
    rf       = joblib.load(MODELS_DIR / 'random_forest.pkl')
    xgb      = joblib.load(MODELS_DIR / 'xgboost.pkl')
    le       = joblib.load(PROCESSED_DIR / 'label_encoder.pkl')
    pipeline = joblib.load(PROCESSED_DIR / 'pipeline.pkl')
    logger.info("Models loaded successfully!")
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title="Network IDS API",
    description="Real-time Network Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response Models ───────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float]
    get_threat_brief: Optional[bool] = False

class PredictResponse(BaseModel):
    prediction:   str
    confidence:   float
    is_attack:    bool
    timestamp:    str
    threat_brief: Optional[str] = None

class FeedbackRequest(BaseModel):
    features:  list[float]
    predicted: str
    actual:    str

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "models":    ["random_forest", "xgboost", "ensemble"],
        "version":   "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        features = np.array(request.features).reshape(1, -1)

        # Scale features
        features_scaled = pipeline.transform(features)

        # Ensemble prediction
        rf_proba  = rf.predict_proba(features_scaled)
        xgb_proba = xgb.predict_proba(features_scaled)
        combined  = 0.4 * rf_proba + 0.6 * xgb_proba

        pred_idx    = int(np.argmax(combined))
        confidence  = float(combined[0][pred_idx])
        prediction  = le.inverse_transform([pred_idx])[0]
        is_attack   = prediction != 'BENIGN'

        # Get threat brief if requested
        threat_brief = None
        if request.get_threat_brief and is_attack:
            brief = get_groq_brief(prediction, {'count': 1, 'confidence': confidence,
                'flows_per_sec': 100, 'packet_rate': 50,
                'source_ips': '1 unique IP', 'duration': 1,
                'ports_scanned': 3, 'scan_rate': 1,
                'source_ip': '192.168.1.100'
            })
            threat_brief = brief['brief']

        return PredictResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            is_attack=is_attack,
            timestamp=datetime.now().isoformat(),
            threat_brief=threat_brief
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    import json
    entry = {
        'timestamp': datetime.now().isoformat(),
        'predicted': request.predicted,
        'actual':    request.actual,
        'features':  request.features[:10]
    }
    queue_path = PROCESSED_DIR / 'retraining_queue.jsonl'
    with open(queue_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    logger.info(f"Feedback logged: predicted={request.predicted}, actual={request.actual}")
    return {"status": "logged", "timestamp": entry['timestamp']}

@app.get("/stats")
async def stats():
    queue_path = PROCESSED_DIR / 'retraining_queue.jsonl'
    queue_size = 0
    if queue_path.exists():
        with open(queue_path) as f:
            queue_size = sum(1 for _ in f)
    return {
        "model":          "ensemble (RF + XGBoost)",
        "accuracy":       0.9989,
        "macro_f1":       0.9027,
        "attack_classes": 14,
        "throughput":     "28,531 flows/sec",
        "retraining_queue": queue_size,
        "timestamp":      datetime.now().isoformat()
    }

if __name__ == '__main__':
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0[IP_ADDRESS]", port=8080, reload=True)