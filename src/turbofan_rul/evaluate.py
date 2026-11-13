"""Scores used to judge RUL predictions, and the health-band rule."""

import numpy as np
from numpy.typing import ArrayLike

from turbofan_rul.config import ALERT_THRESHOLD, BAND_EDGES, MAINTENANCE_AT


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error, in cycles.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.

    Returns
    -------
    float
        Typical size of a prediction error, with big misses weighted more.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in cycles.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.

    Returns
    -------
    float
        Average size of a prediction error.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(d)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Share of the spread in true RUL that the predictions explain.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.

    Returns
    -------
    float
        1 is perfect; 0 is no better than always guessing the mean.
    """
    y = np.asarray(y_true, dtype=float)
    d = np.asarray(y_pred, dtype=float) - y
    return float(1 - np.sum(d**2) / np.sum((y - y.mean()) ** 2))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Asymmetric score from the C-MAPSS paper. Lower is better.

    With ``d = pred - true``, a late prediction (``d >= 0``) costs
    ``exp(d / 10) - 1`` and an early one costs ``exp(-d / 13) - 1``, so
    predicting too much remaining life is punished harder.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.

    Returns
    -------
    float
        Sum of the per-engine penalties.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def alert_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    alert_at: float = ALERT_THRESHOLD,
    due_at: float = MAINTENANCE_AT,
) -> dict[str, int]:
    """Count how well a "service this engine" alert would work.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.
    alert_at : float, optional
        The alert fires when the predicted RUL is at or below this.
    due_at : float, optional
        An engine truly needs service when its true RUL is at or below this.

    Returns
    -------
    dict of str to int
        ``due``: engines that truly need service, ``caught``: how many of those
        got an alert, ``missed``: how many did not, ``false_alarms``: alerts on
        engines that did not need service.
    """
    due = np.asarray(y_true) <= due_at
    flagged = np.asarray(y_pred) <= alert_at
    return {
        "due": int(due.sum()),
        "caught": int((due & flagged).sum()),
        "missed": int((due & ~flagged).sum()),
        "false_alarms": int((flagged & ~due).sum()),
    }


def health_band(rul: ArrayLike) -> np.ndarray:
    """Turn remaining cycles into a status a maintenance planner can act on.

    Parameters
    ----------
    rul : array-like
        Remaining cycles, true or predicted.

    Returns
    -------
    np.ndarray of str
        ``"Critical"`` (30 or fewer), ``"Warning"`` (31 to 60) or ``"Healthy"``.
    """
    r = np.asarray(rul, dtype=float)
    return np.select(
        [r <= BAND_EDGES["Critical"], r <= BAND_EDGES["Warning"]],
        ["Critical", "Warning"],
        default="Healthy",
    )


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Collect every score in one dictionary.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted remaining cycles.

    Returns
    -------
    dict of str to float
        ``rmse``, ``mae``, ``r2``, ``nasa`` and the alert counts from ``alert_metrics``.
    """
    return {
        "rmse": round(rmse(y_true, y_pred), 2),
        "mae": round(mae(y_true, y_pred), 2),
        "r2": round(r2(y_true, y_pred), 3),
        "nasa": round(nasa_score(y_true, y_pred), 1),
        **alert_metrics(y_true, y_pred),
    }
