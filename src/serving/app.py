#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import json
import numpy as np
from joblib import load
from tensorflow import keras
import pandas as pd

app = FastAPI(title="Network Intrusion Detection API")

artifacts  = Path("artifacts")
data = Path("data/processed")
feat_names = []
feat_path = data / "feature_names.json"

if feat_names:
    expected_dim = len(feat_names)
elif (data / "X_train.npy").exists():
    expected_dim = int(np.load(data / "X_train.npy").shape[1])
else:
    expected_dim = None

def _to_infer_frame(x: np.ndarray):
    return pd.DataFrame(x, columns=feat_names) if feat_names else x

#lgbm + threshold
lgbm = None
for p in [artifacts / "lgbm_model.pkl", artifacts / "lgbm_model.pkl"]:
    if p.exists():
        lgbm = load(p)
        break

th_lgbm = 0.5
th_file = artifacts / "lgbm_threshold.json"
if th_file.exists():
    try:
        th_lgbm = float(json.loads(th_file.read_text(encoding="utf-8")).get("best_threshold", 0.5))
    except Exception:
        pass

# AE + threshold
ae = None
ae_thr = None
ae_h5 = artifacts / "ae_model.h5"
ae_thr_pkl = artifacts / "ae_threshold.pkl"
if ae_h5.exists() and ae_thr_pkl.exists():
    ae = keras.models.load_model(str(ae_h5))
    ae_thr = float(load(ae_thr_pkl))



class Flow(BaseModel):
    values: List[float] = Field(..., description="Feature vector đã preprocess (đúng chiều expected_dim)")

@app.get("/")
def root():
    return {"message": "OK. See /docs for API."}

@app.get("/health")
def health():
    return {
        "expected_dim": expected_dim,
        "lgbm_loaded": lgbm is not None,
        "lgbm_threshold": th_lgbm,
        "ae_loaded": ae is not None,
        "ae_threshold": ae_thr,
    }

@app.post("/predict")
def predict(flow: Flow):
    if expected_dim is None:
        raise HTTPException(status_code=500, detail="Server isn't familiar with enough features .")
    x = np.asarray(flow.values, dtype=np.float32).reshape(1, -1)
    if x.shape[1] != expected_dim:
        raise HTTPException(status_code=400, detail=f"Expected feature length {expected_dim}, got {x.shape[1]}")

    res = {"expected_dim": expected_dim}

    # LightGBM
    if lgbm is not None:
        proba = float(lgbm.predict_proba(_to_infer_frame(x))[:, 1][0])
        res["lgbm_attack_prob"] = proba
        res["is_attack_lgbm"] = bool(proba >= th_lgbm)

    # Autoencoder (TensorFlow)
    if ae is not None and ae_thr is not None:
        rec = ae.predict(x, verbose=0)
        err = float(((rec - x) ** 2).mean())
        res["ae_error"] = err
        res["is_attack_ae"] = bool(err >= ae_thr)

    # Fusion OR
    if ("is_attack_lgbm" in res) or ("is_attack_ae" in res):
        res["fusion_or"] = bool(res.get("is_attack_lgbm", False) or res.get("is_attack_ae", False))

    return res
