"""Prometheus metrics the service exposes at ``/metrics`` for Grafana."""

from collections import OrderedDict
from threading import Lock

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from turbofan_rul.config import SENSORS
from turbofan_rul.drift import MIN_ENGINES, drift_report, healthy_baseline

REQUESTS = Counter("rul_requests", "HTTP requests handled", ["route", "method", "status"])
LATENCY = Histogram("rul_request_latency_seconds", "Time to answer a request", ["route"])
ENGINES_SCORED = Counter("rul_engines_scored", "Engines that received a prediction")
ALERTS = Counter("rul_alerts", "Predictions at or below the alert threshold")
PREDICTED_RUL = Histogram(
    "rul_predicted_cycles",
    "Predicted remaining cycles at each engine's last cycle",
    buckets=(10, 20, 30, 40, 50, 60, 80, 100, 125, 150),
)
SENSOR_PSI = Gauge("rul_sensor_psi", "PSI of recent early-life readings vs training", ["sensor"])
DRIFT_ENGINES = Gauge("rul_drift_window_engines", "Engines in the live drift window")
MODEL_INFO = Gauge("rul_model_info", "The model being served (value is always 1)", ["version", "run_id"])


class DriftWindow:
    """Keep recent early-life readings and publish their drift per sensor.

    Readings are keyed by ``(unit, cycle)``, so scoring the same engine twice
    (for example, replaying it in the dashboard) does not count it twice.

    Parameters
    ----------
    profile : dict
        Training sensor distribution from ``reference_profile``.
    size : int, optional
        How many recent readings to keep, by default 5000.
    """

    def __init__(self, profile: dict, size: int = 5000) -> None:
        self.profile = profile
        self.size = size
        self.rows: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.lock = Lock()  # requests are served from several threads

    def update(self, cycles: pd.DataFrame) -> None:
        """Add new readings, then refresh the drift gauges once enough engines are in.

        Parameters
        ----------
        cycles : pd.DataFrame
            Raw readings with ``unit``, ``cycle`` and every column in ``SENSORS``.
        """
        base = healthy_baseline(cycles)
        with self.lock:
            for unit, cycle, values in zip(
                base["unit"], base["cycle"], base[SENSORS].to_numpy(dtype=float), strict=True
            ):
                self.rows.pop((unit, cycle), None)  # re-insert so the newest reading wins
                self.rows[(unit, cycle)] = values
            while len(self.rows) > self.size:
                self.rows.popitem(last=False)
            engines = len({unit for unit, _ in self.rows})
            recent = pd.DataFrame(np.array(list(self.rows.values())), columns=SENSORS)
            recent["cycle"] = [cycle for _, cycle in self.rows]
        DRIFT_ENGINES.set(engines)
        if engines >= MIN_ENGINES:
            for sensor, value in drift_report(recent, self.profile).items():
                SENSOR_PSI.labels(sensor).set(value)


def record_predictions(predicted_rul: pd.Series, alert_threshold: float) -> None:
    """Count the engines scored and alerted, and record their predictions.

    Parameters
    ----------
    predicted_rul : pd.Series
        Prediction at each engine's last cycle.
    alert_threshold : float
        Predictions at or below this count as alerts.
    """
    ENGINES_SCORED.inc(len(predicted_rul))
    ALERTS.inc(int((predicted_rul <= alert_threshold).sum()))
    for value in predicted_rul:
        PREDICTED_RUL.observe(value)
