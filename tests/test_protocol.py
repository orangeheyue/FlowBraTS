import pytest
import torch

from src.data.protocol import MASKS, apply_mask, sample_masks, validate_mask


def test_exactly_fifteen_nonempty_masks():
    assert len(MASKS) == 15
    assert len(set(MASKS)) == 15
    assert all(any(mask) for mask in MASKS)


def test_sampling_is_per_sample_and_valid():
    masks = sample_masks(128)
    assert masks.shape == (128, 4)
    assert masks.any(dim=1).all()
    assert all(tuple(row.tolist()) in MASKS for row in masks)


def test_mask_is_applied_before_encoder_input():
    image = torch.ones(2, 4, 3, 3, 3)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    observed = apply_mask(image, mask)
    assert torch.equal(observed[:, :, 0, 0, 0].bool(), mask)


def test_all_zero_mask_is_rejected():
    with pytest.raises(ValueError):
        validate_mask(torch.zeros(1, 4, dtype=torch.bool))
