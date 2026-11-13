import json

import pytest

from turbofan_rul.config import DATA_DIR, ROOT
from turbofan_rul.data import load_cycles
from turbofan_rul.serving import ModelBundle
from turbofan_rul.train import main

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (DATA_DIR / "train_FD001.txt").exists(), reason="raw NASA files missing"),
]


def test_pipeline_trains_registers_and_only_promotes_better_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    model_dir = tmp_path / "models"
    args = ["--config", str(ROOT / "configs" / "smoke.yaml"), "--model-dir", str(model_dir)]

    first = main(args)
    assert first["promoted"]
    assert first["version"] == "1"
    scores = ModelBundle.load(model_dir).score(load_cycles("test").query("unit <= 3"))
    assert scores.history["unit"].nunique() == 3

    # Pretend production already holds a far better model: the new run must
    # still be registered (version 2) but must not replace it.
    metadata_file = model_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["cv_rmse"] = 1.0
    metadata_file.write_text(json.dumps(metadata))

    second = main(args)
    assert second["version"] == "2"
    assert not second["promoted"]
    assert json.loads(metadata_file.read_text())["cv_rmse"] == 1.0
