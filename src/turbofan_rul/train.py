"""Train, evaluate, track and register the RUL model.

Run ``python -m turbofan_rul.train`` from the project root. One run:

1. loads and validates the NASA files, then builds the rolling features;
2. scores two baselines (linear regression, random forest) with GroupKFold;
3. grid-searches XGBoost with GroupKFold, logging every combination to MLflow;
4. tests the best model once on the 100 held-out test engines;
5. registers it in the MLflow model registry; and
6. promotes it (``production`` alias plus an export to ``models/``) only when its
   cross-validated RMSE is no worse than the model already in production.
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from xgboost import XGBRegressor

from turbofan_rul.config import (
    ALERT_THRESHOLD,
    BAND_EDGES,
    DATA_DIR,
    DATASET,
    FEATURES,
    MAINTENANCE_AT,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL,
    ROOT,
    SENSORS,
)
from turbofan_rul.data import add_rul, load_cycles, load_test_rul, validate_cycles
from turbofan_rul.drift import reference_profile
from turbofan_rul.evaluate import evaluate
from turbofan_rul.features import add_features, last_cycles
from turbofan_rul.serving import METADATA_FILE, MODEL_FILE, PROFILE_FILE

SCORING = "neg_root_mean_squared_error"
log = logging.getLogger("turbofan_rul.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; ``None`` reads them from the command line.

    Returns
    -------
    argparse.Namespace
        ``config``, ``data_dir`` and ``model_dir``.
    """
    parser = argparse.ArgumentParser(description="Train and register the turbofan RUL model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR, help="where a promoted model is exported"
    )
    return parser.parse_args(argv)


def build_datasets(data_dir: Path) -> dict:
    """Load, validate and featurise the training and test engines.

    Parameters
    ----------
    data_dir : Path
        Folder with the raw NASA files.

    Returns
    -------
    dict
        ``train`` (all training cycles), ``X``, ``y``, ``groups`` (engine id per
        row, for GroupKFold), ``X_test`` (last cycle of each test engine) and ``y_test``.
    """
    train = add_features(add_rul(validate_cycles(load_cycles("train", data_dir))))
    test_last = last_cycles(add_features(validate_cycles(load_cycles("test", data_dir))))
    return {
        "train": train,
        "X": train[FEATURES].astype(float),  # s17 is read as int; the model sees floats
        "y": train["RUL"],
        "groups": train["unit"],
        "X_test": test_last[FEATURES].astype(float),
        "y_test": load_test_rul(data_dir),
    }


def setup_mlflow() -> None:
    """Point MLflow at the tracking server, or at a local SQLite file if none is set."""
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        artifacts = None
        if uri.startswith("sqlite:///"):  # local store: keep artifacts next to the database
            artifacts = (Path(uri.removeprefix("sqlite:///")).parent / "mlruns").as_uri()
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifacts)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def score_baseline(name: str, model: RegressorMixin, columns: list[str], data: dict, cv: GroupKFold) -> dict:
    """Cross-validate, fit and test one baseline model, logging it as a child run.

    Parameters
    ----------
    name : str
        Label for the comparison table.
    model : RegressorMixin
        An unfitted scikit-learn regressor.
    columns : list of str
        Feature columns the model sees.
    data : dict
        Output of ``build_datasets``.
    cv : GroupKFold
        Splitter that keeps each engine in a single fold.

    Returns
    -------
    dict
        ``model``, ``cv_rmse`` and the test scores from ``evaluate``.
    """
    X, X_test = data["X"][columns], data["X_test"][columns]
    with mlflow.start_run(run_name=name, nested=True):
        cv_rmse = -cross_val_score(model, X, data["y"], groups=data["groups"], cv=cv, scoring=SCORING).mean()
        model.fit(X, data["y"])
        row = {"model": name, "cv_rmse": round(cv_rmse, 2), **evaluate(data["y_test"], model.predict(X_test))}
        mlflow.set_tag("role", "baseline")
        mlflow.log_params({"model": type(model).__name__, "n_features": len(columns)})
        mlflow.log_metrics({k: v for k, v in row.items() if k != "model"})
    log.info("%s: CV RMSE %.2f, test RMSE %.2f", name, row["cv_rmse"], row["rmse"])
    return row


def tune_xgboost(cfg: dict, data: dict, cv: GroupKFold) -> GridSearchCV:
    """Grid-search XGBoost and log every combination as a child run.

    Parameters
    ----------
    cfg : dict
        The ``xgboost`` section of the config: ``fixed`` and ``grid`` parameters.
    data : dict
        Output of ``build_datasets``.
    cv : GroupKFold
        Splitter that keeps each engine in a single fold.

    Returns
    -------
    GridSearchCV
        Fitted search; ``best_estimator_`` is refit on all training engines.
    """
    search = GridSearchCV(
        XGBRegressor(**cfg["fixed"], random_state=cfg["random_state"]),
        cfg["grid"],
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
    )
    search.fit(data["X"], data["y"], groups=data["groups"])
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"xgb-grid-{i:02d}", nested=True):
            mlflow.set_tag("role", "grid-search")
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "cv_rmse": -res["mean_test_score"][i],
                    "cv_rmse_std": res["std_test_score"][i],
                    "fit_seconds": res["mean_fit_time"][i],
                }
            )
    log.info("best %s, CV RMSE %.2f", search.best_params_, -search.best_score_)
    return search


def production_cv_rmse(model_dir: Path) -> float | None:
    """CV RMSE of the model currently exported for serving, if there is one.

    Parameters
    ----------
    model_dir : Path
        Folder the API serves from.

    Returns
    -------
    float or None
        ``None`` when nothing has been exported yet.
    """
    path = model_dir / METADATA_FILE
    return json.loads(path.read_text(encoding="utf-8"))["cv_rmse"] if path.exists() else None


def export_bundle(model_dir: Path, model: XGBRegressor, metadata: dict, profile: dict) -> None:
    """Write the model, its metadata and the drift profile for the API.

    Parameters
    ----------
    model_dir : Path
        Destination folder, created if missing.
    model : XGBRegressor
        The model to serve.
    metadata : dict
        Version, run id, parameters, metrics and policy.
    profile : dict
        Training sensor distribution from ``reference_profile``.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / MODEL_FILE)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / PROFILE_FILE).write_text(json.dumps(profile), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    """Run the whole training pipeline.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; see ``parse_args``.

    Returns
    -------
    dict
        The metadata of the trained model, plus ``promoted`` (bool).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = build_datasets(args.data_dir)
    cv = GroupKFold(n_splits=cfg["cv_folds"])
    setup_mlflow()

    with mlflow.start_run(run_name="train") as run:
        train_file = args.data_dir / f"train_{DATASET}.txt"
        data_info = {
            "dataset": DATASET,
            "train_rows": len(data["train"]),
            "train_engines": int(data["groups"].nunique()),
            "test_engines": len(data["y_test"]),
            "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        }
        mlflow.log_params({"config": args.config.name, "cv_folds": cfg["cv_folds"], **data_info})

        rf = cfg["random_forest"]
        comparison = [
            score_baseline("Linear regression", LinearRegression(), SENSORS, data, cv),
            score_baseline(
                "Random forest",
                RandomForestRegressor(**rf, n_jobs=-1, random_state=cfg["random_state"]),
                FEATURES,
                data,
                cv,
            ),
        ]

        xgb_cfg = {**cfg["xgboost"], "random_state": cfg["random_state"]}
        search = tune_xgboost(xgb_cfg, data, cv)
        best = search.best_estimator_
        test = evaluate(data["y_test"], best.predict(data["X_test"]))
        cv_rmse = round(-search.best_score_, 2)
        comparison.append({"model": "XGBoost (tuned)", "cv_rmse": cv_rmse, **test})
        mlflow.log_metrics({"cv_rmse": cv_rmse, **{f"test_{k}": v for k, v in test.items()}})
        log.info("test: %s", test)

        info = mlflow.xgboost.log_model(
            best,
            name="model",
            registered_model_name=REGISTERED_MODEL,
            input_example=data["X"].head(3),
        )
        version = str(info.registered_model_version)
        metadata = {
            "name": REGISTERED_MODEL,
            "version": version,
            "run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "params": {**cfg["xgboost"]["fixed"], **search.best_params_},
            "cv_rmse": cv_rmse,
            "test_metrics": test,
            "comparison": comparison,
            "policy": {
                "alert_threshold": ALERT_THRESHOLD,
                "maintenance_at": MAINTENANCE_AT,
                "band_edges": BAND_EDGES,
            },
            "data": data_info,
            "features": FEATURES,
        }
        profile = reference_profile(data["train"])
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_dict(profile, "reference_profile.json")
        mlflow.log_table(pd.DataFrame(comparison), "comparison.json")

        current = production_cv_rmse(args.model_dir)
        promoted = current is None or cv_rmse <= current
        mlflow.set_tag("promoted", promoted)
        if promoted:
            export_bundle(args.model_dir, best, metadata, profile)
            MlflowClient().set_registered_model_alias(REGISTERED_MODEL, PRODUCTION_ALIAS, version)
            log.info("promoted version %s to %s -> %s", version, PRODUCTION_ALIAS, args.model_dir)
        else:
            log.info("kept production model (CV %.2f beats %.2f)", current, cv_rmse)
    return {**metadata, "promoted": promoted}


if __name__ == "__main__":
    main()
