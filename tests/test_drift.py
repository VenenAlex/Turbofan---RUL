import numpy as np
import pandas as pd
import pytest

from turbofan_rul.config import SENSORS
from turbofan_rul.drift import drift_report, drift_status, psi, reference_profile


def sensors(seed: int, shift: float = 0.0, cycle_offset: int = 0) -> pd.DataFrame:
    """200 engines x 25 cycles of standard-normal readings."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({s: rng.normal(shift, 1.0, 5000) for s in SENSORS})
    df["unit"] = np.arange(5000) // 25 + 1
    df["cycle"] = np.arange(5000) % 25 + 1 + cycle_offset
    return df


def test_psi_is_zero_for_identical_shares():
    assert psi([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0)


def test_fresh_sample_of_the_same_data_is_stable():
    report = drift_report(sensors(seed=1), reference_profile(sensors(seed=0)))
    assert max(report.values()) < 0.1


def test_shifted_sensor_is_flagged_and_others_are_not():
    new = sensors(seed=1)
    new["s4"] += 1.0  # one standard deviation
    report = drift_report(new, reference_profile(sensors(seed=0)))
    assert drift_status(report["s4"]) == "significant"
    assert all(v < 0.1 for s, v in report.items() if s != "s4")


def test_late_life_readings_are_ignored():
    # Worn engines move their sensors on purpose; that is not drift.
    worn = sensors(seed=2, shift=5.0, cycle_offset=100)
    report = drift_report(pd.concat([sensors(seed=1), worn]), reference_profile(sensors(seed=0)))
    assert max(report.values()) < 0.1


def test_batch_without_early_cycles_is_rejected():
    with pytest.raises(ValueError, match="first 30 cycles"):
        drift_report(sensors(seed=1, cycle_offset=100), reference_profile(sensors(seed=0)))


@pytest.mark.parametrize(("value", "status"), [(0.05, "stable"), (0.2, "moderate"), (0.3, "significant")])
def test_status_ranges(value, status):
    assert drift_status(value) == status
