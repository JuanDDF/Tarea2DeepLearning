"""Modelos Pydantic de entrada y salida de la API."""
import math
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

VALID_HORIZONS = (7, 14, 21, 28)


class TimeSeriesData(BaseModel):
    """Últimos precios de cierre diarios (USD), del más antiguo al más reciente."""

    lags: List[float] = Field(..., min_length=1, description="Ventana de precios de cierre en USD")
    horizon: Literal[7, 14, 21, 28] = Field(7, description="Horizonte de predicción en días")

    @field_validator("lags")
    @classmethod
    def precios_validos(cls, v: List[float]) -> List[float]:
        if any((not math.isfinite(x)) or x <= 0 for x in v):
            raise ValueError("todos los precios deben ser números finitos y positivos")
        return v


class PredictionResponse(BaseModel):
    horizon: int
    lag: int
    prediction: List[float]


class ModelInfo(BaseModel):
    horizon: int
    lag: int
    architecture: List[int]
    dropout_mask: List[bool]
