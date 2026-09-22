from pathlib import Path

import json

import numpy as np
import pandas as pd


def write_merged_csv(path: Path, rows_per_house: int = 7, missing_toaster: bool = True) -> None:
    appliances = ["Television Site", "Toaster", "Microwave", "Kettle", "Computer Site", "Washing Machine", "Dishwasher"]
    thresholds = [15, 1000, 200, 2000, 20, 20, 10]
    rows = []
    for house in (1, 2, 3):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=house)
        for i in range(rows_per_house):
            row = {"Time": start + pd.Timedelta(seconds=i * 8), "Hour_X": 0.0, "Hour_Y": 1.0, "DoW_X": 0.5, "DoW_Y": -0.5, "is_weekend": 0, "Aggregate": 100 + i, "house": house}
            for name, threshold in zip(appliances, thresholds):
                row[name] = threshold + 1 if i % 3 == 0 else 0
            if missing_toaster:
                row["Toaster"] = np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_config(tmp_path: Path, rows_per_house: int = 7, missing_toaster: bool = True):
    from nilm.config import NILMConfig
    csv_path = tmp_path / "merged.csv"
    write_merged_csv(csv_path, rows_per_house, missing_toaster)
    return NILMConfig.from_dict({"artifact_dir": "artifacts", "data": {"data_file": str(csv_path), "cache_dir": "cache", "train_houses": [1], "validation_houses": [2], "test_houses": [3], "window_size": 5}}, base_dir=tmp_path)


def test_chunked_cache_preserves_houses_features_and_target_nan(tmp_path: Path):
    from nilm.cache import build_cache, load_cached_houses
    config = make_config(tmp_path)
    manifest = build_cache(config, chunk_size=4, overwrite=True)
    houses = load_cached_houses(config.cache_dir, [1], mmap_mode="r")
    assert manifest["row_counts"] == {"1": 7, "2": 7, "3": 7}
    assert houses[0].features.shape == (7, 6)
    assert houses[0].target_power.shape == (7, 7)
    assert np.isnan(houses[0].target_power[:, 1]).all()


def test_cache_resamples_multiple_readings_into_fixed_eight_second_bins(tmp_path: Path):
    from nilm.cache import build_cache, load_cached_houses
    config = make_config(tmp_path)
    frame = pd.read_csv(config.data_file)
    duplicate = frame.iloc[[0]].copy()
    duplicate["Time"] = "2024-01-02 00:00:02"
    duplicate["Aggregate"] = 200.0
    pd.concat([frame, duplicate], ignore_index=True).sort_values(["house", "Time"]).to_csv(config.data_file, index=False)

    manifest = build_cache(config, chunk_size=4, overwrite=True)
    house = load_cached_houses(config.cache_dir, [1])[0]

    assert manifest["row_counts"]["1"] == 7
    assert house.features[0, 0] == 150.0


def test_cache_loader_rejects_manifest_that_no_longer_matches_config(tmp_path: Path):
    from nilm.cache import build_cache, load_cached_houses
    config = make_config(tmp_path)
    build_cache(config, chunk_size=4, overwrite=True)
    manifest_path = config.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_columns"] = list(reversed(manifest["feature_columns"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "contract"):
        load_cached_houses(config.cache_dir, [1], expected_config=config)


def test_cache_drops_infinite_feature_rows_before_writing(tmp_path: Path):
    from nilm.cache import build_cache
    config = make_config(tmp_path)
    frame = pd.read_csv(config.data_file)
    frame["Aggregate"] = frame["Aggregate"].astype(float)
    frame.loc[(frame["house"] == 1) & (frame.index == 0), "Aggregate"] = np.inf
    frame.to_csv(config.data_file, index=False)
    manifest = build_cache(config, chunk_size=4, overwrite=True)
    assert manifest["row_counts"]["1"] == 6
