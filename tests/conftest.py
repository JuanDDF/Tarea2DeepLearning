"""Configuración de pruebas.

Si ``app/`` contiene los cuatro modelos entrenados, las pruebas los usan. Si no
(por ejemplo, en CI antes de exportar los modelos), se crean modelos de prueba
pequeños con la misma interfaz en una carpeta temporal.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

HORIZONS = (7, 14, 21, 28)
APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _scaler(n, mu, sd):
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()
    s.mean_, s.scale_ = np.full(n, mu), np.full(n, sd)
    s.var_, s.n_features_in_, s.n_samples_seen_ = s.scale_ ** 2, n, 1
    return s


def _make_dummy_models(d: Path, lag: int = 7):
    import tensorflow as tf

    for H in HORIZONS:
        m = tf.keras.Sequential([tf.keras.layers.Input((lag,)),
                                 tf.keras.layers.Dense(8, activation="relu"),
                                 tf.keras.layers.Dense(H)])
        m.save(d / f"model_h{H}.keras")
        joblib.dump(dict(x=_scaler(lag, np.log(30000.0), 0.5), y=_scaler(H, np.log(30000.0), 0.5),
                         use_log=True, lag=lag, arch=[8], mask=[False]), d / f"scalers_h{H}.joblib")


def pytest_configure(config):
    reales = all((APP_DIR / f"model_h{H}.keras").exists() and (APP_DIR / f"scalers_h{H}.joblib").exists()
                 for H in HORIZONS)
    if not reales and "T2_MODELS_DIR" not in os.environ:
        import tempfile

        d = Path(tempfile.mkdtemp(prefix="t2_models_"))
        _make_dummy_models(d)
        os.environ["T2_MODELS_DIR"] = str(d)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return Path(os.environ.get("T2_MODELS_DIR", APP_DIR))
