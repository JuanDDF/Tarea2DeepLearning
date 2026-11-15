"""Validación de los modelos exportados y sus escaladores."""
import joblib
import numpy as np
import pytest
import tensorflow as tf

HORIZONS = (7, 14, 21, 28)


@pytest.mark.parametrize("H", HORIZONS)
def test_model_prediction_shape(models_dir, H):
    model = tf.keras.models.load_model(models_dir / f"model_h{H}.keras", compile=False)
    sc = joblib.load(models_dir / f"scalers_h{H}.joblib")
    lag = sc["x"].n_features_in_
    assert model.input_shape[-1] == lag
    out = model.predict(np.zeros((3, lag)), verbose=0)
    assert out.shape == (3, H)
    assert sc["y"].n_features_in_ == H


@pytest.mark.parametrize("H", HORIZONS)
def test_escaladores_invertibles(models_dir, H):
    sc = joblib.load(models_dir / f"scalers_h{H}.joblib")
    x = np.log(np.linspace(20000, 90000, sc["x"].n_features_in_)).reshape(1, -1)
    assert np.allclose(sc["x"].inverse_transform(sc["x"].transform(x)), x)
