import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_generate_training_figures_writes_two_png_files(tmp_path: Path):
    from nilm.plotting import generate_training_figures

    history = [
        {"epoch": 1, "train_loss": 0.8, "validation_loss": 0.9, "train_macro_f1": 0.2, "validation_macro_f1": 0.18},
        {"epoch": 2, "train_loss": 0.6, "validation_loss": 0.7, "train_macro_f1": 0.3, "validation_macro_f1": 0.25},
    ]
    metrics = {
        "Television Site": {"f1": 0.6, "average_precision": 0.55},
        "Toaster": {"f1": 0.1, "average_precision": 0.08},
    }

    paths = generate_training_figures(history, metrics, tmp_path / "figures")

    assert set(paths) == {"training_history", "test_metrics"}
    for path in paths.values():
        image_path = Path(path)
        assert image_path.parent == tmp_path / "figures"
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_masked_bce_ignores_unobserved_targets():
    from nilm.train import masked_bce_with_logits
    logits = torch.zeros((1, 3))
    targets = torch.tensor([[1.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_bce_with_logits(logits, targets, mask, torch.ones(3))
    assert loss.item() == pytest.approx(np.log(2))


def test_class_weights_are_per_appliance_and_mask_aware():
    from nilm.train import per_appliance_class_weights
    targets = np.array([[1, 1], [0, 0], [0, 1], [0, 0]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 1], [1, 1], [1, 0]], dtype=np.float32)
    weights, counts = per_appliance_class_weights(targets, mask)
    np.testing.assert_allclose(weights, [3.0, 1.0])
    assert counts[0] == {"positive": 1, "negative": 3, "observed": 4}
    assert counts[1] == {"positive": 1, "negative": 1, "observed": 2}


def test_metrics_ignore_masked_rows_and_report_each_appliance():
    from nilm.train import multi_appliance_metrics
    result = multi_appliance_metrics(np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.9], [0.2, 0.8]]), np.array([[1, 0], [1, 1]]), ["a", "b"])
    assert result["a"]["f1"] == pytest.approx(1.0)
    assert result["b"]["observed"] == 1


def test_streaming_metrics_accumulate_without_retaining_all_predictions():
    from nilm.train import StreamingMultiApplianceMetrics
    metrics = StreamingMultiApplianceMetrics(["a", "b"], average_precision_bins=100)
    metrics.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.9, 0.8]]), torch.tensor([[1.0, 0.0]]))
    metrics.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.2, 0.7]]), torch.tensor([[1.0, 1.0]]))
    result = metrics.compute()
    assert result["a"]["f1"] == pytest.approx(1.0)
    assert result["b"]["observed"] == 1
    assert metrics.retained_sample_count == 0


def test_checkpoint_round_trip_preserves_multi_output_contract(tmp_path: Path):
    from nilm.model import MultiApplianceNILM
    from nilm.train import load_checkpoint, save_checkpoint
    model = MultiApplianceNILM(6, 7, window_size=5, dropout=0)
    path = tmp_path / "model.pt"
    from nilm.config import APPLIANCE_COLUMNS, DEFAULT_THRESHOLDS, FEATURE_COLUMNS
    save_checkpoint(path, model, {"appliance_columns": list(APPLIANCE_COLUMNS), "feature_columns": list(FEATURE_COLUMNS), "thresholds": dict(DEFAULT_THRESHOLDS), "normalization": {"mean": 100.0, "std": 20.0, "index": 0}})
    loaded = load_checkpoint(path)
    assert loaded["format_version"] == 2
    assert loaded["model_config"] == model.architecture_config()


def test_checkpoint_rejects_wrong_appliance_order(tmp_path: Path):
    from nilm.config import APPLIANCE_COLUMNS, DEFAULT_THRESHOLDS, FEATURE_COLUMNS
    from nilm.model import MultiApplianceNILM
    from nilm.train import load_checkpoint, save_checkpoint
    path = tmp_path / "bad.pt"
    save_checkpoint(path, MultiApplianceNILM(6, 7, 5), {"appliance_columns": list(reversed(APPLIANCE_COLUMNS)), "feature_columns": list(FEATURE_COLUMNS), "thresholds": dict(DEFAULT_THRESHOLDS), "normalization": {"mean": 0.0, "std": 1.0, "index": 0}})
    with pytest.raises(ValueError, match="appliance order"):
        load_checkpoint(path)


def test_train_from_cached_data_writes_one_seven_output_checkpoint(tmp_path: Path):
    from nilm.cache import build_cache
    from nilm.train import load_checkpoint, train_from_config
    from tests.test_nilm_cache import make_config
    config = make_config(tmp_path, rows_per_house=15, missing_toaster=False)
    object.__setattr__(config, "epochs", 1)
    object.__setattr__(config, "batch_size", 4)
    object.__setattr__(config, "stride", 2)
    object.__setattr__(config, "dropout", 0.0)
    build_cache(config, chunk_size=7, overwrite=True)
    result = train_from_config(config, device="cpu")
    checkpoint = load_checkpoint(result["checkpoint_path"])
    assert len(checkpoint["appliance_columns"]) == 7
    assert set(checkpoint["metrics"]) == set(config.appliance_columns)
    assert set(result["figure_paths"]) == {"training_history", "test_metrics"}
    assert all(Path(path).is_file() for path in result["figure_paths"].values())
    metrics_report = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics_report["figure_paths"] == result["figure_paths"]
