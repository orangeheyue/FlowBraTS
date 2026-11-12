"""Train-only lesion-aware patches; validation/test retain the original reader."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import MASKS


def crop_start(shape, size, center=None):
    if any(n < size for n in shape):
        raise ValueError(f"Volume {shape} is smaller than patch {size}")
    if center is None:
        return tuple(np.random.randint(n - size + 1) for n in shape)
    # Keep selected lesion voxel away from the crop boundary, with random jitter.
    return tuple(int(np.clip(c - size // 2 + np.random.randint(-size//4, size//4+1),
                             0, n-size)) for c,n in zip(center,shape))


class ETPatchDataset(Dataset):
    def __init__(self, root, split_dir, split_index=None, patch_size=80,
                 et_probability=.55, tc_probability=.25, full_probability=.2):
        suffix = str(split_index) if split_index is not None else ""
        self.names = sorted((Path(split_dir)/f"train{suffix}.txt").read_text().split())
        self.root = Path(root)
        self.size = patch_size
        self.et_probability = et_probability
        self.tc_probability = tc_probability
        self.full_probability = full_probability
        if not 0 <= et_probability + tc_probability <= 1:
            raise ValueError("Invalid lesion sampling probabilities")
        if not 0 <= full_probability <= 1:
            raise ValueError("Invalid full-modality probability")
        self.locations = OrderedDict()

    def __len__(self):
        return len(self.names)

    def coordinates(self, name, seg):
        if name not in self.locations:
            # Cache bounded coordinate samples in each worker, never full volumes.
            locations = []
            for region in ((seg == 3) | (seg == 4), (seg == 1) | (seg == 3) | (seg == 4)):
                indices = np.flatnonzero(region)
                if len(indices) > 8192:
                    indices = np.random.choice(indices, 8192, replace=False)
                locations.append(indices)
            self.locations[name] = locations
            if len(self.locations) > 256:
                self.locations.popitem(last=False)
        return self.locations[name]

    def __getitem__(self, index):
        name = self.names[index]
        image = np.load(self.root/'vol'/f'{name}_vol.npy', mmap_mode='r')
        seg = np.load(self.root/'seg'/f'{name}_seg.npy', mmap_mode='r')
        et, tc = self.coordinates(name, seg)
        draw = np.random.rand()
        indices = et if draw < self.et_probability else tc
        center = None
        if draw < self.et_probability + self.tc_probability and len(indices):
            center = np.unravel_index(indices[np.random.randint(len(indices))], seg.shape)
        start = crop_start(seg.shape, self.size, center)
        slices = tuple(slice(s,s+self.size) for s in start)
        x = np.array(image[slices], dtype=np.float32, copy=True)
        y = np.array(seg[slices], dtype=np.int64, copy=True)
        y[y == 4] = 3
        # Orthogonal rotations/flips preserve small integer-label lesions exactly.
        turns = np.random.randint(4)
        x, y = np.rot90(x, turns, axes=(0,1)), np.rot90(y, turns, axes=(0,1))
        for axis in range(3):
            if np.random.rand() < .5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        x = x * np.random.uniform(.9,1.1,size=(1,1,1,4))
        x = x + np.random.uniform(-.1,.1,size=(1,1,1,4))
        mask = np.ones(4,dtype=bool) if np.random.rand() < self.full_probability else np.array(MASKS[np.random.randint(15)])
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(3,0,1,2),dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y)), torch.from_numpy(mask), name)
