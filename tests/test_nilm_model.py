import pytest
import torch


def test_model_maps_six_channel_window_to_seven_logits():
    from nilm.model import MultiApplianceNILM
    model = MultiApplianceNILM(input_channels=6, num_appliances=7, window_size=37)
    assert tuple(model(torch.zeros(4, 6, 37)).shape) == (4, 7)


def test_model_rejects_wrong_channel_count():
    from nilm.model import MultiApplianceNILM
    model = MultiApplianceNILM(input_channels=6, num_appliances=7, window_size=37)
    with pytest.raises(ValueError, match="6"):
        model(torch.zeros(2, 1, 37))
