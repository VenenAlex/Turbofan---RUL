"""Turn raw sensor readings into model features."""

import pandas as pd

from turbofan_rul.config import SENSORS, WINDOW


def add_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add a rolling mean and rolling standard deviation of every sensor.

    The windows are computed separately for each engine, so one engine's
    readings never leak into another's.

    Parameters
    ----------
    df : pd.DataFrame
        Engine cycles sorted by ``unit`` then ``cycle``.
    window : int, optional
        Number of cycles in each window, by default ``WINDOW``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``<sensor>_mean`` and ``<sensor>_std`` columns.
        The first cycle of each engine gets a std of 0.
    """
    out = df.copy()
    for s in SENSORS:
        grp = out.groupby("unit")[s]
        out[f"{s}_mean"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        out[f"{s}_std"] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return out


def last_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each engine's latest cycle, which is the point we predict from.

    Parameters
    ----------
    df : pd.DataFrame
        Engine cycles with a ``unit`` column.

    Returns
    -------
    pd.DataFrame
        One row per engine, indexed by ``unit``.
    """
    return df.groupby("unit").last()
