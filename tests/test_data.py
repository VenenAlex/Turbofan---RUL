import numpy as np
import pandas as pd
import pytest

from turbofan_rul.config import COLUMNS
from turbofan_rul.data import add_rul, validate_cycles


def raw_cycles(cycles: list[int]) -> pd.DataFrame:
    """Engine 1 with the given cycle numbers and every reading set to 0.5."""
    df = pd.DataFrame(0.5, index=range(len(cycles)), columns=COLUMNS)
    df["unit"], df["cycle"] = 1, cycles
    return df


def test_rul_counts_down_to_zero_and_is_capped():
    df = pd.DataFrame({"unit": [1, 1, 1, 1, 1, 2, 2, 2], "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})
    out = add_rul(df, cap=3)
    # unit 1 fails after cycle 5: 4, 3, 2, 1, 0 left, capped at 3
    assert out["RUL"].tolist() == [3, 3, 2, 1, 0, 2, 1, 0]


def test_validate_sorts_by_unit_then_cycle():
    out = validate_cycles(raw_cycles([3, 1, 2]))
    assert out["cycle"].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.drop(columns="s4"), "missing columns"),
        (lambda df: df.assign(s4=np.nan), "missing or infinite"),
        (lambda df: df.assign(cycle=[1, 1, 2]), "only once"),
        (lambda df: df.assign(cycle=[0, 1, 2]), "whole numbers"),
    ],
)
def test_validate_rejects_bad_data(change, message):
    with pytest.raises(ValueError, match=message):
        validate_cycles(change(raw_cycles([1, 2, 3])))
