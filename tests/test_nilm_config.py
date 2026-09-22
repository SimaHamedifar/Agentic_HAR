import json
from pathlib import Path

import pytest


def _payload() -> dict:
    return {"artifact_dir": "artifacts/nilm", "data": {"data_file": "data/REFIT_multi_appliance.csv", "cache_dir": "cache/refit", "train_houses": [1, 6, 7, 15], "validation_houses": [2, 5], "test_houses": [3, 11]}}


def test_config_uses_sima_appliance_order_thresholds_and_six_features(tmp_path: Path):
    from nilm.config import APPLIANCE_COLUMNS, FEATURE_COLUMNS, NILMConfig
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    config = NILMConfig.from_json(path)
    assert config.appliance_columns == APPLIANCE_COLUMNS
    assert config.feature_columns == FEATURE_COLUMNS
    assert config.thresholds["Kettle"] == 2000.0
    assert config.thresholds["Microwave"] == 200.0
    assert config.data_file == (tmp_path / "data/REFIT_multi_appliance.csv").resolve()


def test_config_rejects_house_split_overlap(tmp_path: Path):
    from nilm.config import NILMConfig
    payload = _payload()
    payload["data"]["validation_houses"] = [6]
    with pytest.raises(ValueError, match="overlap"):
        NILMConfig.from_dict(payload, base_dir=tmp_path)


def test_config_requires_threshold_for_every_appliance(tmp_path: Path):
    from nilm.config import NILMConfig
    payload = _payload()
    payload["thresholds"] = {"Kettle": 2000}
    with pytest.raises(ValueError, match="threshold"):
        NILMConfig.from_dict(payload, base_dir=tmp_path)


def test_config_rejects_even_window_size(tmp_path: Path):
    from nilm.config import NILMConfig
    payload = _payload()
    payload["data"]["window_size"] = 38
    with pytest.raises(ValueError, match="odd"):
        NILMConfig.from_dict(payload, base_dir=tmp_path)


def test_config_rejects_non_sima_feature_order(tmp_path: Path):
    from nilm.config import NILMConfig
    payload = _payload()
    payload["data"]["feature_columns"] = ["Hour_X", "Aggregate", "Hour_Y", "DoW_X", "DoW_Y", "is_weekend"]
    with pytest.raises(ValueError, match="Sima"):
        NILMConfig.from_dict(payload, base_dir=tmp_path)
