"""Detect when incoming sensor data stops looking like the training data.

The model only knows the conditions it was trained on. If a sensor is swapped,
recalibrated or starts failing, or the fleet flies in different conditions, the
readings shift and the predictions become unreliable, often without any error
being raised. The Population Stability Index (PSI) measures that shift per sensor:

* below 0.1: stable
* 0.1 to 0.25: moderate shift, worth a look
* above 0.25: significant shift, do not trust the predictions

Only each engine's first ``BASELINE_CYCLES`` cycles are compared. Wear moves the
sensors on purpose (that is what the model reads), so comparing whole lifetimes
would flag a perfectly normal fleet just because its engines are younger or older
than the training engines. Early cycles are healthy in every engine, so there the
two sets should match unless the sensors or operating conditions really changed.
"""

import numpy as np
import pandas as pd

from turbofan_rul.config import SENSORS

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
BASELINE_CYCLES = 30  # the shortest FD001 test engine has 31 cycles
MIN_ENGINES = 50  # engines differ from birth; fewer than this gives noisy PSI


def healthy_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each engine's first ``BASELINE_CYCLES`` cycles.

    Parameters
    ----------
    df : pd.DataFrame
        Engine cycles with a ``cycle`` column.

    Returns
    -------
    pd.DataFrame
        The rows with ``cycle <= BASELINE_CYCLES``.
    """
    return df[df["cycle"] <= BASELINE_CYCLES]


def reference_profile(df: pd.DataFrame, bins: int = 10) -> dict[str, dict[str, list[float]]]:
    """Summarise how each sensor is distributed in the training engines' early cycles.

    Parameters
    ----------
    df : pd.DataFrame
        Training cycles with ``cycle`` and every column in ``SENSORS``.
    bins : int, optional
        Number of quantile buckets per sensor, by default 10 (deciles).

    Returns
    -------
    dict
        For each sensor, the inner bucket ``edges`` and the share of training
        readings that fell in each bucket (``expected``). Saved next to the model.
    """
    base = healthy_baseline(df)
    inner = np.linspace(0, 1, bins + 1)[1:-1]
    profile = {}
    for s in SENSORS:
        edges = np.unique(np.quantile(base[s], inner))  # unique: some sensors repeat values
        profile[s] = {"edges": edges.tolist(), "expected": _shares(base[s], edges).tolist()}
    return profile


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Compute the Population Stability Index between two sets of bucket shares.

    Parameters
    ----------
    expected, actual : np.ndarray
        Share of readings in each bucket; each sums to 1.
    eps : float, optional
        Floor for empty buckets, so the log stays finite.

    Returns
    -------
    float
        0 when the shares match; grows as they move apart.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(df: pd.DataFrame, profile: dict) -> dict[str, float]:
    """PSI of every sensor in the early cycles of ``df`` against the training profile.

    Parameters
    ----------
    df : pd.DataFrame
        New cycles with ``cycle`` and every column in ``SENSORS``.
    profile : dict
        Output of ``reference_profile`` on the training data.

    Returns
    -------
    dict of str to float
        PSI per sensor, rounded to 4 decimals.

    Raises
    ------
    ValueError
        If ``df`` has no readings from the first ``BASELINE_CYCLES`` cycles.
    """
    base = healthy_baseline(df)
    if base.empty:
        raise ValueError(f"drift needs readings from each engine's first {BASELINE_CYCLES} cycles")
    report = {}
    for s in SENSORS:
        edges = np.asarray(profile[s]["edges"])
        report[s] = round(psi(profile[s]["expected"], _shares(base[s], edges)), 4)
    return report


def drift_status(value: float) -> str:
    """Name the PSI range a value falls in.

    Parameters
    ----------
    value : float
        A PSI value.

    Returns
    -------
    str
        ``"stable"``, ``"moderate"`` or ``"significant"``.
    """
    if value > PSI_SIGNIFICANT:
        return "significant"
    return "moderate" if value > PSI_MODERATE else "stable"


def _shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Share of ``values`` in each bucket defined by the inner ``edges``."""
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / counts.sum()
