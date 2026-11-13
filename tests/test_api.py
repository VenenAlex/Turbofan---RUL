import pandas as pd
import pytest
from fastapi.testclient import TestClient

from turbofan_rul.api.main import app
from turbofan_rul.config import DATA_DIR, MODEL_DIR
from turbofan_rul.data import load_cycles, load_test_rul
from turbofan_rul.evaluate import rmse

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.json").exists() or not (DATA_DIR / "test_FD001.txt").exists(),
    reason="needs the trained bundle in models/ and the NASA files in data/raw",
)


def to_readings(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "unit": int(r.unit),
            "cycle": int(r.cycle),
            "settings": [r.op1, r.op2, r.op3],
            "sensors": [getattr(r, f"s{i}") for i in range(1, 22)],
        }
        for r in df.itertuples()
    ]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the start-up that loads the model
        yield c


@pytest.fixture(scope="module")
def fleet() -> pd.DataFrame:
    return load_cycles("test")


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_model_card_describes_the_served_model(client):
    card = client.get("/api/v1/models/production").json()
    assert len(card["features"]) == 42
    assert card["test_metrics"]["rmse"] < 21.9  # must beat the linear baseline


def test_served_model_reproduces_the_offline_test_score(client, fleet):
    # Training-serving parity: scoring the raw test files through HTTP must give
    # the same RMSE that train.py logged for this model version.
    body = client.post("/api/v1/predictions", json={"readings": to_readings(fleet)}).json()
    predicted = [e["predicted_rul"] for e in sorted(body["engines"], key=lambda e: e["unit"])]
    card = client.get("/api/v1/models/production").json()
    assert rmse(load_test_rul(), predicted) == pytest.approx(card["test_metrics"]["rmse"], abs=0.01)


def test_explanation_adds_up_to_the_prediction(client, fleet):
    body = client.post(
        "/api/v1/predictions", json={"readings": to_readings(fleet[fleet.unit == 37]), "explain": True}
    ).json()
    engine = body["engines"][0]
    total = engine["explanation"]["base_value"] + sum(
        c["value"] for c in engine["explanation"]["contributions"]
    )
    assert total == pytest.approx(engine["predicted_rul"], abs=0.05)


def test_history_has_one_prediction_per_cycle(client, fleet):
    engine_cycles = fleet[fleet.unit == 37]
    body = client.post(
        "/api/v1/predictions", json={"readings": to_readings(engine_cycles), "include_history": True}
    ).json()
    history = body["engines"][0]["history"]
    assert len(history) == len(engine_cycles)
    assert history[-1]["predicted_rul"] == body["engines"][0]["predicted_rul"]


def test_reading_with_missing_sensors_is_rejected(client, fleet):
    reading = to_readings(fleet.head(1))[0]
    reading["sensors"] = reading["sensors"][:20]
    assert client.post("/api/v1/predictions", json={"readings": [reading]}).status_code == 422


def test_duplicate_cycle_is_rejected(client, fleet):
    readings = to_readings(fleet.head(1)) * 2
    response = client.post("/api/v1/predictions", json={"readings": readings})
    assert response.status_code == 422
    assert "only once" in response.json()["detail"]


def test_drift_report_flags_a_shifted_sensor(client, fleet):
    clean = client.post("/api/v1/drift-reports", json={"readings": to_readings(fleet)}).json()
    assert clean["status"] == "stable"

    shifted = fleet.assign(s4=fleet["s4"] + fleet["s4"].std())
    report = client.post("/api/v1/drift-reports", json={"readings": to_readings(shifted)}).json()
    psi = {s["sensor"]: s for s in report["sensors"]}
    assert psi["s4"]["status"] == "significant"
    assert report["warning"] is None


def test_small_drift_batch_carries_a_warning(client, fleet):
    report = client.post("/api/v1/drift-reports", json={"readings": to_readings(fleet[fleet.unit <= 5])})
    assert "noisy" in report.json()["warning"]


def test_metrics_are_exposed_for_prometheus(client, fleet):
    client.post("/api/v1/predictions", json={"readings": to_readings(fleet)})
    text = client.get("/metrics").text
    assert "rul_engines_scored_total" in text
    assert 'rul_sensor_psi{sensor="s4"}' in text
    assert 'route="/api/v1/predictions"' in text
