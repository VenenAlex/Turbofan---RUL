"""Load the production model bundle and turn raw engine cycles into predictions."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from turbofan_rul.config import FEATURES, MODEL_DIR
from turbofan_rul.data import validate_cycles
from turbofan_rul.features import add_features

MODEL_FILE = "model.json"
METADATA_FILE = "metadata.json"
PROFILE_FILE = "reference_profile.json"


@dataclass(frozen=True)
class Scores:
    """Everything one scoring call produces.

    Attributes
    ----------
    history : pd.DataFrame
        ``unit``, ``cycle`` and ``predicted_rul`` for every cycle received.
    contributions : pd.DataFrame or None
        SHAP contribution of each feature at each engine's last cycle, indexed by
        ``unit``, plus a ``base_value`` column. ``None`` unless asked for.
    """

    history: pd.DataFrame
    contributions: pd.DataFrame | None


@dataclass(frozen=True)
class ModelBundle:
    """The trained model plus the files exported with it by ``train.py``.

    Attributes
    ----------
    booster : xgb.Booster
        The trained XGBoost model.
    metadata : dict
        Version, MLflow run id, parameters, metrics and alert policy.
    profile : dict
        Training sensor distribution, used for drift checks.
    """

    booster: xgb.Booster
    metadata: dict
    profile: dict

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "ModelBundle":
        """Read a bundle exported by ``train.py``.

        Parameters
        ----------
        model_dir : Path, optional
            Folder holding ``model.json``, ``metadata.json`` and ``reference_profile.json``.

        Returns
        -------
        ModelBundle
            Ready to score.
        """
        booster = xgb.Booster()
        booster.load_model(model_dir / MODEL_FILE)
        metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        profile = json.loads((model_dir / PROFILE_FILE).read_text(encoding="utf-8"))
        return cls(booster, metadata, profile)

    def score(self, cycles: pd.DataFrame, explain: bool = False) -> Scores:
        """Predict remaining cycles at every cycle of every engine.

        Features are rolling windows that only look backwards, so the prediction
        at cycle ``c`` uses nothing after ``c``. That is what makes replaying an
        engine's history honest.

        Parameters
        ----------
        cycles : pd.DataFrame
            Raw engine cycles with the 26 NASA columns.
        explain : bool, optional
            Also return SHAP contributions at each engine's last cycle.

        Returns
        -------
        Scores
            Per-cycle predictions, and contributions when ``explain`` is True.

        Raises
        ------
        ValueError
            If ``cycles`` fails ``validate_cycles``.
        """
        feats = add_features(validate_cycles(cycles))
        history = feats[["unit", "cycle"]].copy()
        # Remaining life cannot be negative, so clip the rare tiny negative output.
        history["predicted_rul"] = np.clip(self._predict(feats[FEATURES]), 0, None)

        contributions = None
        if explain:
            last = feats.groupby("unit").tail(1).set_index("unit")[FEATURES]
            values = self.booster.predict(xgb.DMatrix(last), pred_contribs=True)
            contributions = pd.DataFrame(values, index=last.index, columns=[*FEATURES, "base_value"])
        return Scores(history, contributions)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raw model output as float64."""
        return self.booster.predict(xgb.DMatrix(X)).astype(float)
