from src.evaluation import better, reference
from src.config import load_config


def result(d,h):return {'mean_dice':[d]*4,'mean_hd95':[h]*4}


def test_dice_first_and_hd95_ties():
    baseline=result(.8,10)
    assert better(baseline,None)
    assert better(result(.81,12),baseline)
    assert better(result(.8,9),baseline)
    assert not better(result(.79,1),baseline)
    assert not better(result(.80001,9.999),baseline)


def test_et_selection_has_wt_tc_guards():
    baseline={'mean_dice':[.92,.90,.75,.75],'mean_hd95':[4,4,4,4]}
    selection={'metric':'et_guarded','minimum_dice':[.915,.895]}
    improved={'mean_dice':[.919,.899,.77,.77],'mean_hd95':[5,5,6,6]}
    degraded={'mean_dice':[.91,.899,.80,.80],'mean_hd95':[3,3,3,3]}
    tied={'mean_dice':[.919,.899,.75,.75],'mean_hd95':[4,4,3,3]}
    assert better(improved,baseline,selection=selection)
    assert not better(degraded,baseline,selection=selection)
    assert better(tied,baseline,selection=selection)


def test_original_functions_and_all_masks_loaded():
    protocol=reference(load_config())
    assert len(protocol['masks_test'])==15
    assert len({tuple(mask) for mask in protocol['masks_test']})==15
    assert protocol['mask_name'][-1]=='flairt1cet1t2'
    assert protocol['test_dice_hd95_softmax'].__code__.co_filename.endswith('M2FTrans_v1/predict.py')


def test_fast_validation_subsets_and_test_always_all_fifteen():
    import pytest
    from src.evaluation import selected_masks
    protocol=reference(load_config())
    names=['flairt1cet1t2','flairt1t2','t1cet1','t1']
    pairs=selected_masks(protocol,'val',names)
    assert [name for _,name in pairs]==names
    assert len(selected_masks(protocol,'test'))==15
    with pytest.raises(ValueError,match='all 15'):selected_masks(protocol,'test',names)
    with pytest.raises(ValueError):selected_masks(protocol,'val',['unknown'])
    with pytest.raises(ValueError):selected_masks(protocol,'val',['t1','t1'])
