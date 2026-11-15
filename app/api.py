"""API de inferencia: pronóstico multisalida del precio de cierre diario de BTCUSDT.

Para cada horizonte H en {7, 14, 21, 28} se carga el MLP con la mejor arquitectura
encontrada (``model_h{H}.keras``) y sus escaladores (``scalers_h{H}.joblib``).
El modelo trabaja sobre log(precio) estandarizado; la API recibe y devuelve USD.
"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from app.schemas import VALID_HORIZONS, ModelInfo, PredictionResponse, TimeSeriesData

app = FastAPI(title="Pronóstico BTCUSDT con MLP multisalida", version="1.0.0")


def models_dir() -> Path:
    return Path(os.environ.get("T2_MODELS_DIR", Path(__file__).resolve().parent))


@lru_cache(maxsize=None)
def load(horizon: int):
    """Carga (una sola vez) el modelo y los escaladores del horizonte pedido."""
    import tensorflow as tf

    d = models_dir()
    model_path, scaler_path = d / f"model_h{horizon}.keras", d / f"scalers_h{horizon}.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(status_code=503, detail=f"No hay modelo para H={horizon} en {d}")
    return tf.keras.models.load_model(model_path, compile=False), joblib.load(scaler_path)


@app.get("/health")
def health():
    d = models_dir()
    disponibles = [H for H in VALID_HORIZONS
                   if (d / f"model_h{H}.keras").exists() and (d / f"scalers_h{H}.joblib").exists()]
    return {"status": "ok", "horizons": disponibles}


@app.get("/model/{horizon}", response_model=ModelInfo)
def model_info(horizon: int):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=422, detail=f"horizon debe ser uno de {VALID_HORIZONS}")
    _, sc = load(horizon)
    return ModelInfo(horizon=horizon, lag=int(sc["x"].n_features_in_),
                     architecture=list(sc.get("arch", [])), dropout_mask=list(sc.get("mask", [])))


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TimeSeriesData):
    H = data.horizon
    model, sc = load(H)
    scaler_x, scaler_y = sc["x"], sc["y"]
    features = np.asarray(data.lags, dtype="float64").reshape(1, -1)
    if features.shape[1] != scaler_x.n_features_in_:
        raise HTTPException(status_code=422,
                            detail=f"Se esperaban {scaler_x.n_features_in_} lags para H={H}, "
                                   f"se recibieron {features.shape[1]}")
    if sc.get("use_log", False):
        features = np.log(features)
    pred_scaled = model.predict(scaler_x.transform(features), verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    if sc.get("use_log", False):
        pred = np.exp(pred)
    return PredictionResponse(horizon=H, lag=int(scaler_x.n_features_in_),
                              prediction=[float(v) for v in pred])
