import pytest

from turbofan_rul.config import DATA_DIR, FEATURES
from turbofan_rul.data import add_rul, load_cycles, load_test_rul
from turbofan_rul.features import add_features, last_cycles

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "train_FD001.txt").exists(), reason="raw NASA files are not in data/raw"
)


@needs_data
def test_shapes_match_the_notebook():
    train = add_features(add_rul(load_cycles("train")))
    test_last = last_cycles(add_features(load_cycles("test")))

    assert train[FEATURES].shape == (20631, 42)
    assert train["unit"].nunique() == 100
    assert train["RUL"].max() == 125
    assert test_last[FEATURES].shape == (100, 42)
    assert len(load_test_rul()) == 100
