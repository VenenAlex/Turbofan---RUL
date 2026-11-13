import math

import pytest

from turbofan_rul.evaluate import alert_metrics, health_band, nasa_score, r2, rmse


def test_perfect_predictions_score_zero():
    assert rmse([10, 20], [10, 20]) == 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))


def test_r2_of_always_guessing_the_mean_is_zero():
    assert r2([1, 2, 3], [2, 2, 2]) == pytest.approx(0)


def test_nasa_punishes_late_more_than_early():
    late = nasa_score([50], [60])  # predicted 10 cycles too much life
    early = nasa_score([50], [40])  # predicted 10 cycles too little
    assert late == pytest.approx(math.exp(10 / 10) - 1)
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late > early


def test_alert_metrics():
    # due = true RUL <= 30, alert = predicted RUL <= 40
    # engine 1 due and alerted (caught), engine 2 due but not alerted (missed),
    # engine 3 alerted but not due (false alarm), engine 4 neither
    m = alert_metrics(y_true=[10, 20, 35, 80], y_pred=[15, 45, 38, 90], alert_at=40, due_at=30)
    assert m == {"due": 2, "caught": 1, "missed": 1, "false_alarms": 1}


def test_health_band_edges():
    bands = health_band([0, 30, 30.5, 60, 61]).tolist()
    assert bands == ["Critical", "Critical", "Warning", "Warning", "Healthy"]
