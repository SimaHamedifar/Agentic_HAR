import numpy as np
import pytest

from nilm import preprocessing


def _series() -> preprocessing.HouseSeries:
    features = np.column_stack([np.array([10, 20, 30, 40, 50], dtype=np.float32), np.tile(np.array([0, 1, 0, -1, 0], dtype=np.float32), (5, 1))])
    power = np.full((5, 7), np.nan, dtype=np.float32)
    power[:, 0] = [0, 0, 20, 0, 0]
    power[:, 3] = [0, 0, 2500, 0, 0]
    return preprocessing.HouseSeries(features=features, target_power=power, timestamps=np.arange(5, dtype=np.float64) * 8, house_id="H1", source="synthetic")


def test_targets_and_mask_use_thresholds_and_preserve_missing_labels():
    power = np.array([[0.0, np.nan, 250.0], [20.0, 1100.0, np.nan]])
    targets, mask = preprocessing.activation_targets_and_mask(power, thresholds=np.array([15.0, 1000.0, 200.0]))
    np.testing.assert_array_equal(targets, [[0, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(mask, [[1, 0, 1], [1, 1, 0]])


def test_valid_window_centres_exclude_windows_crossing_a_gap():
    timestamps = np.array([0, 8, 16, 24, 32, 80, 88, 96, 104, 112], dtype=float)
    centres = preprocessing.valid_window_centres(timestamps, 5, 8.0, 1.5)
    np.testing.assert_array_equal(centres, [2, 7])


def test_fit_normalization_uses_training_aggregate_only():
    stats = preprocessing.fit_aggregate_normalization([_series()], aggregate_index=0)
    assert stats.mean == pytest.approx(30.0)
    assert stats.std == pytest.approx(np.std([10, 20, 30, 40, 50]))


def test_window_dataset_returns_six_channels_seven_targets_and_mask():
    dataset = preprocessing.MultiApplianceWindowDataset([_series()], preprocessing.AggregateNormalization(30.0, 10.0, 0), window_size=5, thresholds=np.array([15, 1000, 200, 2000, 20, 20, 10]), expected_interval_seconds=8.0)
    window, targets, mask = dataset[0]
    assert tuple(window.shape) == (6, 5)
    np.testing.assert_allclose(window[0].numpy(), [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(targets.numpy(), [1, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(mask.numpy(), [1, 0, 0, 1, 0, 0, 0])


def test_dataset_drops_centres_with_no_observed_appliance():
    series = _series()
    series.target_power[2] = np.nan
    with pytest.raises(ValueError, match="no valid windows"):
        preprocessing.MultiApplianceWindowDataset([series], preprocessing.AggregateNormalization(30, 10, 0), 5, np.array([15, 1000, 200, 2000, 20, 20, 10]), 8.0)
