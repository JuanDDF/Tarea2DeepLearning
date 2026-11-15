"""Pruebas unitarias de la API."""
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import app

HORIZONS = (7, 14, 21, 28)
client = TestClient(app)


def _lag(models_dir, H):
    return int(joblib.load(models_dir / f"scalers_h{H}.joblib")["x"].n_features_in_)


def test_health_lista_los_cuatro_horizontes():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["horizons"] == list(HORIZONS)


@pytest.mark.parametrize("H", HORIZONS)
def test_predict_devuelve_H_valores_positivos(models_dir, H):
    lags = list(np.linspace(60000, 62000, _lag(models_dir, H)))
    r = client.post("/predict", json={"lags": lags, "horizon": H})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == H
    assert len(body["prediction"]) == H
    assert all(np.isfinite(body["prediction"])) and min(body["prediction"]) > 0


@pytest.mark.parametrize("H", HORIZONS)
def test_model_info(models_dir, H):
    r = client.get(f"/model/{H}")
    assert r.status_code == 200
    assert r.json()["lag"] == _lag(models_dir, H)


def test_numero_de_lags_incorrecto_da_422(models_dir):
    lags = [60000.0] * (_lag(models_dir, 7) + 1)
    assert client.post("/predict", json={"lags": lags, "horizon": 7}).status_code == 422


def test_horizonte_invalido_da_422():
    assert client.post("/predict", json={"lags": [60000.0] * 7, "horizon": 10}).status_code == 422


def test_precio_no_positivo_da_422():
    assert client.post("/predict", json={"lags": [60000.0] * 6 + [-1.0], "horizon": 7}).status_code == 422
