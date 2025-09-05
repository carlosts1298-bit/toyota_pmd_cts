import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Paths (can be overridden with environment variables)
# -------------------------------------------------------------------
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "model.joblib"))
IMPUTER_PATH = os.getenv("IMPUTER_PATH", os.path.join(ARTIFACTS_DIR, "imputer.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
FEATURE_NAMES_PATH = os.getenv("FEATURE_NAMES_PATH", os.path.join(ARTIFACTS_DIR, "feature_names.json"))
SELECTED_FEATURES_PATH = os.getenv("SELECTED_FEATURES_PATH", os.path.join(ARTIFACTS_DIR, "selected_features.json"))

# -------------------------------------------------------------------
# Load artifacts on startup
# -------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(IMPUTER_PATH) and os.path.exists(SCALER_PATH)):
    raise RuntimeError("Missing artifacts. Please run the training pipeline first.")

with open(FEATURE_NAMES_PATH, "r") as f:
    FEATURE_ORDER: List[str] = json.load(f)

try:
    with open(SELECTED_FEATURES_PATH, "r") as f:
        SELECTED_FEATURES: List[str] = json.load(f)
except FileNotFoundError:
    SELECTED_FEATURES = FEATURE_ORDER

model = joblib.load(MODEL_PATH)     # sklearn LogisticRegression
imputer = joblib.load(IMPUTER_PATH) # sklearn SimpleImputer
scaler = joblib.load(SCALER_PATH)   # sklearn StandardScaler

DEFAULT_THRESHOLD = 0.393  # From KS statistic in the short report

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance — Minimal Deployment",
    version="1.0.0",
    description="FastAPI app serving a Logistic Regression model with imputer and scaler.",
)

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class InputRecord(BaseModel):
    # All input features are optional (imputer handles NaN)
    volt_mean_14d: Optional[float] = None
    volt_range_14d: Optional[float] = None
    volt_rstd_14d: Optional[float] = None
    rotate_mean_14d: Optional[float] = None
    rotate_std_14d: Optional[float] = None
    rotate_range_14d: Optional[float] = None
    pressure_mean_14d: Optional[float] = None
    pressure_range_14d: Optional[float] = None
    vibration_mean_14d: Optional[float] = None
    vibration_range_14d: Optional[float] = None
    vibration_rstd_14d: Optional[float] = None

class PredictRequest(BaseModel):
    records: List[InputRecord]
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description=f"Decision threshold. Defaults to {DEFAULT_THRESHOLD:.3f} if not provided."
    )

class PredictResponseItem(BaseModel):
    proba: float
    pred: int
    message: str

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Reindex input records to match FEATURE_ORDER used during training."""
    df = pd.DataFrame(records)

    # Add missing features as NaN, ignore unknown ones
    df = df.reindex(columns=FEATURE_ORDER, fill_value=np.nan)
    return df

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts": {
            "model": os.path.abspath(MODEL_PATH),
            "imputer": os.path.abspath(IMPUTER_PATH),
            "scaler": os.path.abspath(SCALER_PATH),
            "feature_names": os.path.abspath(FEATURE_NAMES_PATH),
            "selected_features": os.path.abspath(SELECTED_FEATURES_PATH),
        },
        "n_features_expected": len(FEATURE_ORDER),
    }

@app.get("/metadata")
def metadata():
    return {
        "model_type": str(type(model)),
        "features_ordered_for_model": FEATURE_ORDER,
        "selected_features": SELECTED_FEATURES,
        "notes": (
            "Inputs are reindexed to FEATURE_ORDER before transform. "
            "Missing values are imputed (median), then scaled, then scored. "
            "Output includes probability, class (0/1), and an interpretation message."
        ),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="Empty 'records' list.")

    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD

    # 1) DataFrame aligned to training order
    df = _records_to_dataframe([r.dict() for r in req.records])

    # 2) Imputation and scaling
    X_imp = imputer.transform(df)
    X_scl = scaler.transform(X_imp)

    # 3) Predict
    proba = model.predict_proba(X_scl)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = []
    for p, c in zip(proba, preds):
        msg = "It will probably fail in the next 14 days." if c == 1 else "It is unlikely to fail in the next 14 days."
        results.append(PredictResponseItem(proba=float(p), pred=int(c), message=msg))

    return PredictResponse(results=results)
