from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _checkpoint(path: Path) -> None:
    from nilm.config import APPLIANCE_COLUMNS, DEFAULT_THRESHOLDS, FEATURE_COLUMNS
    from nilm.model import MultiApplianceNILM
    from nilm.train import save_checkpoint
    model = MultiApplianceNILM(6, 7, window_size=5, dropout=0)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    save_checkpoint(path, model, {"appliance_columns": list(APPLIANCE_COLUMNS), "feature_columns": list(FEATURE_COLUMNS), "thresholds": dict(DEFAULT_THRESHOLDS), "normalization": {"mean": 100.0, "std": 20.0, "index": 0}})


def test_one_checkpoint_returns_seven_named_probabilities(tmp_path: Path):
    from nilm.inference import NILMPredictor
    path = tmp_path / "model.pt"
    _checkpoint(path)
    predictor = NILMPredictor.from_checkpoint(path, device="cpu")
    result = predictor.predict(np.zeros((6, 5), dtype=np.float32))
    assert len(result) == 7
    assert all(value == pytest.approx(0.5) for value in result.values())


def test_predictor_rejects_wrong_feature_shape(tmp_path: Path):
    from nilm.inference import NILMPredictor
    path = tmp_path / "model.pt"
    _checkpoint(path)
    predictor = NILMPredictor.from_checkpoint(path, device="cpu")
    with pytest.raises(ValueError, match="6, 5"):
        predictor.predict(np.zeros((1, 5), dtype=np.float32))


def test_load_feature_window_csv_uses_checkpoint_column_order(tmp_path: Path):
    from nilm.inference import load_feature_window_csv
    path = tmp_path / "window.csv"
    pd.DataFrame({"Hour_X": [1, 2], "Aggregate": [100, 200], "Hour_Y": [3, 4]}).to_csv(path, index=False)
    matrix = load_feature_window_csv(path, ["Aggregate", "Hour_X", "Hour_Y"])
    np.testing.assert_array_equal(matrix, [[100, 200], [1, 2], [3, 4]])
