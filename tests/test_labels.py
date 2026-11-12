import torch

from src.losses import (empty_et_false_positive_loss, internal_labels,
                        region_probabilities, region_targets, segmentation_loss)


def test_original_et_four_maps_to_internal_three():
    target = torch.tensor([[[[0, 1, 2, 4]]]])
    assert internal_labels(target).tolist() == [[[[0, 1, 2, 3]]]]


def test_region_hierarchy_is_exact():
    target = torch.tensor([[[[0, 1, 2, 3]]]])
    wt, tc, et = region_targets(target)[0, :, 0, 0]
    assert wt.tolist() == [0, 1, 1, 1]
    assert tc.tolist() == [0, 1, 0, 1]
    assert et.tolist() == [0, 0, 0, 1]
    logits = torch.randn(2, 4, 3, 3, 3)
    regions = region_probabilities(logits)
    assert torch.all(regions[:, 2] <= regions[:, 1])
    assert torch.all(regions[:, 1] <= regions[:, 0])


def test_segmentation_loss_supports_autocast():
    logits = torch.randn(1, 4, 2, 2, 2)
    target = torch.randint(0, 4, (1, 2, 2, 2))
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = segmentation_loss(logits, target)
    assert torch.isfinite(loss)


def test_et_focused_loss_is_finite_and_penalizes_empty_et_false_positives():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    clean = torch.full((1, 4, 4, 4, 4), -5.0)
    clean[:, 0] = 5.0
    false_et = clean.clone()
    false_et[:, 3, 0, 0, 0] = 10.0
    assert empty_et_false_positive_loss(false_et, target) > empty_et_false_positive_loss(clean, target)
    loss = segmentation_loss(false_et, target, region_weights=[1, 1, 2],
        ce_class_weights=[1, 1, 1, 2], et_tversky_weight=.35, et_fp_weight=.25)
    assert torch.isfinite(loss)
