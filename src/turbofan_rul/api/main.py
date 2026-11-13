"""FastAPI service: RUL predictions, drift reports, model card and Prometheus metrics.

Run locally with ``uvicorn turbofan_rul.api.main:app --reload`` and open
http://localhost:8000/docs for the interactive API documentation.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from prometheus_client import make_asgi_app

from turbofan_rul.api.monitoring import (
    LATENCY,
    MODEL_INFO,
    REQUESTS,
    DriftWindow,
    record_predictions,
)
from turbofan_rul.api.schemas import (
    Contribution,
    CyclePrediction,
    DriftReport,
    EnginePrediction,
    Explanation,
    Health,
    ModelCard,
    PredictionRequest,
    PredictionResponse,
    ReadingBatch,
    SensorDrift,
)
from turbofan_rul.drift import MIN_ENGINES, drift_report, drift_status
from turbofan_rul.evaluate import health_band
from turbofan_rul.serving import ModelBundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at start-up instead of on every request."""
    bundle = ModelBundle.load()
    app.state.bundle = bundle
    app.state.drift = DriftWindow(bundle.profile)
    MODEL_INFO.labels(bundle.metadata["version"], bundle.metadata["run_id"]).set(1)
    yield


app = FastAPI(
    title="Turbofan RUL service",
    version="1.0.0",
    description="Predicts how many flight cycles a turbofan engine has left before failure "
    "(NASA C-MAPSS FD001), explains each prediction with SHAP and watches for sensor drift.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def track_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Count every request and time it, labelled by route template (not raw URL)."""
    start = time.perf_counter()
    response = await call_next(request)
    if not request.url.path.startswith("/metrics"):
        route = getattr(request.scope.get("route"), "path", "unmatched")
        REQUESTS.labels(route, request.method, str(response.status_code)).inc()
        LATENCY.labels(route).observe(time.perf_counter() - start)
    return response


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors to the interactive docs."""
    return RedirectResponse("/docs")


@app.get("/api/v1/health", response_model=Health, tags=["service"])
def get_health(request: Request) -> Health:
    """Confirm the service is up and say which model version it serves."""
    return Health(status="ok", model_version=request.app.state.bundle.metadata["version"])


@app.get("/api/v1/models/production", response_model=ModelCard, tags=["model"])
def get_production_model(request: Request) -> ModelCard:
    """Describe the served model: version, MLflow run, parameters, metrics and policy."""
    return ModelCard(**request.app.state.bundle.metadata)


@app.post(
    "/api/v1/predictions",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predictions"],
)
def create_predictions(body: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict the remaining cycles of every engine in the request, at its last cycle."""
    bundle: ModelBundle = request.app.state.bundle
    cycles = body.to_frame()
    try:
        scores = bundle.score(cycles, explain=body.explain)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    threshold = bundle.metadata["policy"]["alert_threshold"]
    last = scores.history.groupby("unit").tail(1).set_index("unit")
    histories = dict(tuple(scores.history.groupby("unit"))) if body.include_history else {}
    engines = [
        EnginePrediction(
            unit=int(unit),
            last_cycle=int(row["cycle"]),
            predicted_rul=round(row["predicted_rul"], 2),
            health_band=band,
            alert=bool(row["predicted_rul"] <= threshold),
            explanation=_explanation(scores.contributions.loc[unit]) if body.explain else None,
            history=_history(histories[unit]) if body.include_history else None,
        )
        for (unit, row), band in zip(last.iterrows(), health_band(last["predicted_rul"]), strict=True)
    ]
    request.app.state.drift.update(cycles)
    record_predictions(last["predicted_rul"], threshold)
    return PredictionResponse(
        model_version=bundle.metadata["version"], alert_threshold=threshold, engines=engines
    )


@app.post("/api/v1/drift-reports", response_model=DriftReport, tags=["monitoring"])
def create_drift_report(body: ReadingBatch, request: Request) -> DriftReport:
    """Compare a batch's early-life readings with the training data, sensor by sensor (PSI)."""
    bundle: ModelBundle = request.app.state.bundle
    cycles = body.to_frame()
    try:
        report = drift_report(cycles, bundle.profile)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    engines = int(cycles["unit"].nunique())
    warning = None
    if engines < MIN_ENGINES:
        warning = f"only {engines} engines; PSI is noisy below {MIN_ENGINES}"
    return DriftReport(
        model_version=bundle.metadata["version"],
        readings=len(cycles),
        engines=engines,
        status=drift_status(max(report.values())),
        warning=warning,
        sensors=[SensorDrift(sensor=s, psi=v, status=drift_status(v)) for s, v in report.items()],
    )


def _explanation(row: pd.Series) -> Explanation:
    """Turn one engine's SHAP row into an ``Explanation``, largest effect first."""
    contribs = row.drop("base_value")
    order = contribs.abs().sort_values(ascending=False).index
    return Explanation(
        base_value=round(float(row["base_value"]), 3),
        contributions=[Contribution(feature=f, value=round(float(contribs[f]), 3)) for f in order],
    )


def _history(engine: pd.DataFrame) -> list[CyclePrediction]:
    """Return the prediction at every cycle of one engine."""
    return [
        CyclePrediction(cycle=int(c), predicted_rul=round(float(p), 2))
        for c, p in zip(engine["cycle"], engine["predicted_rul"], strict=True)
    ]
