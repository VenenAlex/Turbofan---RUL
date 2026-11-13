"""Request and response bodies of the RUL service, validated by pydantic."""

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, Field

from turbofan_rul.config import COLUMNS

Finite = Annotated[float, Field(allow_inf_nan=False)]
Band = Literal["Critical", "Warning", "Healthy"]
DriftLevel = Literal["stable", "moderate", "significant"]


class Reading(BaseModel):
    """One engine cycle, in the same order as a row of the NASA files."""

    unit: int = Field(ge=1, description="Engine id.")
    cycle: int = Field(ge=1, description="Flight number since the engine entered service.")
    settings: list[Finite] = Field(min_length=3, max_length=3, description="op1, op2, op3.")
    sensors: list[Finite] = Field(min_length=21, max_length=21, description="s1 to s21.")


class ReadingBatch(BaseModel):
    """Raw cycles for one or more engines, in any order."""

    readings: list[Reading] = Field(min_length=1, max_length=50_000)

    def to_frame(self) -> pd.DataFrame:
        """Lay the readings out like a NASA file.

        Returns
        -------
        pd.DataFrame
            One row per reading, with the 26 columns in ``COLUMNS``.
        """
        rows = [[r.unit, r.cycle, *r.settings, *r.sensors] for r in self.readings]
        return pd.DataFrame(rows, columns=COLUMNS)


class PredictionRequest(ReadingBatch):
    """Cycles to score, plus what to include in the answer."""

    explain: bool = Field(False, description="Add SHAP contributions at each engine's last cycle.")
    include_history: bool = Field(False, description="Add the prediction at every cycle.")


class Contribution(BaseModel):
    """How many cycles one feature added to (+) or removed from (-) the prediction."""

    feature: str
    value: float


class Explanation(BaseModel):
    """SHAP breakdown: ``base_value`` plus every contribution equals the prediction."""

    base_value: float
    contributions: list[Contribution] = Field(description="Largest effect first.")


class CyclePrediction(BaseModel):
    """The prediction the model would have made at one past cycle."""

    cycle: int
    predicted_rul: float


class EnginePrediction(BaseModel):
    """Prediction for one engine at its latest cycle."""

    unit: int
    last_cycle: int
    predicted_rul: float = Field(description="Predicted cycles left before failure.")
    health_band: Band
    alert: bool = Field(description="True when predicted_rul is at or below the alert threshold.")
    explanation: Explanation | None = None
    history: list[CyclePrediction] | None = None


class PredictionResponse(BaseModel):
    """Predictions for every engine in the request."""

    model_version: str
    alert_threshold: float
    engines: list[EnginePrediction]


class SensorDrift(BaseModel):
    """Population Stability Index of one sensor against the training data."""

    sensor: str
    psi: float
    status: DriftLevel


class DriftReport(BaseModel):
    """How far a batch of readings has moved from the training data."""

    model_version: str
    readings: int
    engines: int
    status: DriftLevel = Field(description="The worst sensor's status.")
    warning: str | None = Field(None, description="Set when the batch is too small to trust.")
    sensors: list[SensorDrift]


class Health(BaseModel):
    """Liveness answer, used by Docker and the dashboard."""

    status: Literal["ok"]
    model_version: str


class ModelCard(BaseModel):
    """What is being served, how it was trained and how well it scored."""

    name: str
    version: str
    run_id: str
    trained_at: str
    params: dict[str, float | int]
    cv_rmse: float
    test_metrics: dict[str, float]
    comparison: list[dict[str, float | str]]
    policy: dict
    data: dict[str, int | str]
    features: list[str]
