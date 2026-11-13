"""Read the raw C-MAPSS files and build the training label."""

from pathlib import Path
from typing import IO, Literal

import numpy as np
import pandas as pd

from turbofan_rul.config import COLUMNS, DATA_DIR, DATASET, RUL_CAP


def load_cycles(
    split: Literal["train", "test"], data_dir: Path = DATA_DIR, dataset: str = DATASET
) -> pd.DataFrame:
    """Load one engine-cycle file.

    Parameters
    ----------
    split : {"train", "test"}
        Which file to read.
    data_dir : Path, optional
        Folder that holds the raw NASA text files.
    dataset : str, optional
        C-MAPSS subset, by default "FD001".

    Returns
    -------
    pd.DataFrame
        One row per engine per cycle, with the 26 columns in ``COLUMNS``.
    """
    return read_cycles(data_dir / f"{split}_{dataset}.txt")


def read_cycles(source: Path | IO) -> pd.DataFrame:
    """Read a file in the NASA C-MAPSS format: 26 space-separated columns, no header.

    Parameters
    ----------
    source : Path or file-like
        A path, or an open file such as a dashboard upload.

    Returns
    -------
    pd.DataFrame
        One row per engine per cycle, with the 26 columns in ``COLUMNS``.
    """
    return pd.read_csv(source, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(data_dir: Path = DATA_DIR, dataset: str = DATASET) -> np.ndarray:
    """Load the true remaining cycles of each test engine.

    Parameters
    ----------
    data_dir : Path, optional
        Folder that holds the raw NASA text files.
    dataset : str, optional
        C-MAPSS subset, by default "FD001".

    Returns
    -------
    np.ndarray
        One value per test engine, in unit order (unit 1 first).
    """
    return pd.read_csv(data_dir / f"RUL_{dataset}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()


def validate_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Check engine-cycle data before it reaches the model.

    The same checks guard the training pipeline and every API request, so bad
    data fails loudly instead of producing a quiet, wrong prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Engine cycles with the 26 columns in ``COLUMNS``.

    Returns
    -------
    pd.DataFrame
        A copy sorted by ``unit`` then ``cycle``, ready for ``add_features``.

    Raises
    ------
    ValueError
        If a column is missing, a value is missing or not finite, a unit or
        cycle is not a positive whole number, or a cycle appears twice.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    values = df[COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("readings contain missing or infinite values")
    ids = df[["unit", "cycle"]].to_numpy(dtype=float)
    if (ids < 1).any() or (ids % 1 != 0).any():
        raise ValueError("unit and cycle must be whole numbers of 1 or more")
    if df.duplicated(["unit", "cycle"]).any():
        raise ValueError("each (unit, cycle) pair may appear only once")
    out = df[COLUMNS].astype({"unit": int, "cycle": int})
    return out.sort_values(["unit", "cycle"], ignore_index=True)


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add the ``RUL`` label: cycles left before failure, capped at ``cap``.

    Only valid for training data, where every engine runs until it fails.

    Parameters
    ----------
    df : pd.DataFrame
        Engine cycles with ``unit`` and ``cycle`` columns.
    cap : int, optional
        Highest label value, by default ``RUL_CAP``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with a new ``RUL`` column.
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (last_cycle - out["cycle"]).clip(upper=cap)
    return out
