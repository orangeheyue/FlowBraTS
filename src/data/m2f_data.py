import collections
import collections.abc
import sys
from pathlib import Path


TRAIN_TRANSFORMS = (
    "Compose([RandCrop3D((80,80,80)), RandomRotion(10), "
    "RandomIntensityChange((0.1,0.1)), RandomFlip(0), "
    "NumpyType((np.float32, np.int64)),])"
)
TEST_TRANSFORMS = "Compose([NumpyType((np.float32, np.int64)),])"


def _import_dataset(m2f_root):
    """Desc: 原样导入 M2FTrans_v1 数据集与增强代码。"""
    # M2FTrans 的旧版 transforms.py 仍从 collections 访问 Sequence；
    # Python 3.10+ 已将其移动到 collections.abc。
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence
    root = str(Path(m2f_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from data.datasets_nii import (  # pylint: disable=import-outside-toplevel
        Brats_loadall_nii,
        Brats_loadall_test_nii,
    )
    return Brats_loadall_nii, Brats_loadall_test_nii


def build_dataset(m2f_root, data_root, split_dir, split, split_index=None):
    """Desc: 复用 M2FTrans_v1 的读取、标签和增强逻辑。

    Args:
        m2f_root: M2FTrans_v1 代码目录。
        data_root: 含 `vol/`、`seg/` 的预处理数据目录。
        split_dir: 原始 train/val/test.txt 所在目录。
        split: `train`、`val` 或 `test`。
    Output:
        M2FTrans 原生 Dataset。
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    train_cls, test_cls = _import_dataset(m2f_root)
    suffix = str(split_index) if split_index is not None else ""
    split_file = str(Path(split_dir) / f"{split}{suffix}.txt")
    if split == "train":
        return train_cls(
            transforms=TRAIN_TRANSFORMS,
            root=str(data_root),
            modal="all",
            num_cls=4,
            train_file=split_file,
        )
    return test_cls(
        transforms=TEST_TRANSFORMS,
        root=str(data_root),
        modal="all",
        test_file=split_file,
    )
