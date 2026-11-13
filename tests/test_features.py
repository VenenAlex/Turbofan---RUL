import pandas as pd

from turbofan_rul.config import FEATURES, SENSORS
from turbofan_rul.features import add_features, last_cycles


def toy_cycles() -> pd.DataFrame:
    """Two engines, four cycles each. Unit 1 reads 1, 2, 3, 4; unit 2 reads 101 to 104."""
    rows = []
    for unit, start in [(1, 0.0), (2, 100.0)]:
        for cycle in range(1, 5):
            rows.append({"unit": unit, "cycle": cycle, **{s: start + cycle for s in SENSORS}})
    return pd.DataFrame(rows)


def test_adds_every_feature_column():
    out = add_features(toy_cycles(), window=2)
    assert set(FEATURES) <= set(out.columns)


def test_rolling_mean_covers_the_window():
    out = add_features(toy_cycles(), window=2)
    row = out[(out["unit"] == 1) & (out["cycle"] == 3)].iloc[0]
    assert row["s2_mean"] == 2.5  # mean of cycles 2 and 3


def test_windows_do_not_mix_engines():
    out = add_features(toy_cycles(), window=2)
    row = out[(out["unit"] == 2) & (out["cycle"] == 1)].iloc[0]
    assert row["s2_mean"] == 101.0  # not mixed with unit 1's last reading of 4
    assert row["s2_std"] == 0.0


def test_features_only_look_backwards():
    # Features at cycle 2 must not change when later cycles arrive; otherwise
    # replaying an engine's history (and the test-set score) would leak the future.
    full = add_features(toy_cycles(), window=3)
    cut = add_features(toy_cycles().query("cycle <= 2"), window=3)
    early = full[full["cycle"] <= 2][FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(early, cut[FEATURES].reset_index(drop=True))


def test_last_cycles_keeps_one_row_per_engine():
    last = last_cycles(add_features(toy_cycles()))
    assert last.index.tolist() == [1, 2]
    assert last["cycle"].tolist() == [4, 4]
