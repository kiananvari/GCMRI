import csv
from pathlib import Path
import time
import h5py
from typing import Callable, Optional, Union, Collection, Tuple
import torchvision.transforms.functional as F 

import torch
import numpy as np


from typing import Union, Optional
from torch.utils.data import Dataset

class BratsDataset(Dataset):
    """
    Takes data directory and creates a dataset objcet for BraTS dataset. 
    Need to simulate first using simulate_k_space.py
    """

    def __init__(
            self,
            data_dir: Union[str, Path], 
            nx:int = 256,
            ny:int = 256,
            contrasts: Collection[str] = ['t1', 't2', 'flair', 't1ce'], 
            jointly_reconstructing: bool = True,
            guided_single_contrast: bool = False,
            transforms: Optional[Callable] = None,
            data_key: str = "k_space",
            limit_volumes: Optional[Union[int, float]] = None   
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.nx = nx
        self.ny = ny
        self.transforms = transforms
        self.contrasts = np.array([contrast.lower() for contrast in contrasts])
        self.jointly_reconstructing = jointly_reconstructing
        self.guided_single_contrast = guided_single_contrast
        self.data_key = data_key

        if self.guided_single_contrast and self.jointly_reconstructing:
            raise ValueError(
                "guided_single_contrast=True is only valid when jointly_reconstructing=False"
            )

        # BraTS splits are expected to contain per-volume subdirectories.
        # On macOS, `.DS_Store` may appear and must be ignored.
        sample_dir = [p for p in data_dir.iterdir() if p.is_dir()]
        sample_dir.sort()

        slices = []
        data_list = []
        contrast_order = []
        
        start = time.time()
        first = True
        
        if limit_volumes is None:
            limit_volumes = len(sample_dir)
        elif isinstance(limit_volumes, float):
            limit_volumes = int(limit_volumes * len(sample_dir))
            
        for sample_path in sample_dir[:limit_volumes]:
            sample_files = sorted(sample_path.glob('*.h5'))
            if not sample_files:
                raise FileNotFoundError(
                    f"No .h5 files found in volume folder: {sample_path}. "
                    f"Expected simulated BraTS volumes as .h5 files under: {data_dir}"
                )
            sample_file_path = sample_files[0]
            with h5py.File(sample_file_path, 'r') as fr:
                k_space = fr[self.data_key]
                assert isinstance(k_space, h5py.Dataset)
                num_slices = k_space.shape[0]
                slices.append(num_slices)
                if first:
                    contrast_dataset = fr['contrasts']
                    assert isinstance(contrast_dataset, h5py.Dataset)
                    contrast_order = contrast_dataset[:].astype('U')
                    first = False

            data_list.append(sample_file_path)

        end = time.time()
        print(f'Elapsed time {(end-start)/60}')


        # Build a stable contrast order based on the requested list (YAML order), not the file order.
        # This ensures both joint reconstruction and guided packing are deterministic across train/test.
        contrast_order = np.char.lower(np.array(contrast_order, dtype='U'))
        contrast_to_index = {str(name): int(i) for i, name in enumerate(contrast_order.tolist())}
        selected_indices = []
        for c in self.contrasts.tolist():
            if str(c) not in contrast_to_index:
                raise KeyError(
                    f"Requested contrast {c!r} not found in file contrast list: {contrast_order.tolist()}"
                )
            selected_indices.append(contrast_to_index[str(c)])

        self._selected_contrast_order = self.contrasts
        self._selected_contrast_indices = np.array(selected_indices, dtype=int)
        # h5py requires fancy indices to be in increasing order; keep a sorted version for I/O
        # and a permutation to restore the requested canonical order.
        self._selected_contrast_indices_sorted = np.array(sorted(selected_indices), dtype=int)
        pos = {int(idx): int(i) for i, idx in enumerate(self._selected_contrast_indices_sorted.tolist())}
        self._selected_contrast_reorder = np.array([pos[int(i)] for i in self._selected_contrast_indices.tolist()], dtype=int)
        self.metrics_contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]

        # Keep `contrast_order_indexes` for backward compatibility with the joint fast-path loader.
        # (This mask is in file order.)
        self.contrast_order_indexes = np.isin(contrast_order, self._selected_contrast_order)

        # Expose a stable contrast order for downstream model construction.
        # In single-contrast mode we intentionally train a 1-contrast model on samples
        # drawn from different contrasts.
        if self.jointly_reconstructing:
            self.contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]
            self.num_contrasts_per_sample = int(len(self._selected_contrast_order))
        else:
            if self.guided_single_contrast:
                self.contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]
                self.num_contrasts_per_sample = int(len(self._selected_contrast_order))
            else:
                self.contrast_order = ['single']
                self.num_contrasts_per_sample = 1
        
        self.file_list = np.array(data_list)
        print(self._selected_contrast_order)
        self.slices = np.array(slices)
        self.cumulative_slice_sum = np.cumsum(self.slices)
        self.length = self.cumulative_slice_sum[-1]

        # Small per-worker cache for guided mode.
        # When the DataLoader groups indices for a slice consecutively, this avoids rereading
        # the same HDF5 slice C times (one per target contrast).
        self._guided_last_key: Optional[Tuple[int, int]] = None
        self._guided_last_stack: Optional[np.ndarray] = None

        print(f'Found {sum(self.slices)} slices')

    def __len__(self):
        if self.jointly_reconstructing:
            return int(self.length)
        return int(self.length) * int(len(self._selected_contrast_order))

    def __getitem__(self, index) -> torch.Tensor:
        if self.jointly_reconstructing:
            volume_index, slice_index = self.get_vol_slice_index(index)
            data = self.get_data_from_indecies(volume_index, slice_index)
        else:
            num_contrasts = int(len(self._selected_contrast_order))
            base_index = int(index) // num_contrasts
            contrast_local_index = int(index) % num_contrasts

            volume_index, slice_index = self.get_vol_slice_index(base_index)
            if self.guided_single_contrast:
                data = self.get_guided_target_first_from_indecies(volume_index, slice_index, contrast_local_index)
            else:
                data = self.get_single_contrast_from_indecies(volume_index, slice_index, contrast_local_index)

        if self.transforms:
            data = self.transforms(data)

        return data

    # get the volume index and slice index. This is done using the cumulative sum
    # of the number of slices.
    def get_vol_slice_index(self, index) -> Tuple[int, int]:
        volume_index = np.sum(self.cumulative_slice_sum <= index)
        # if volume index is zero, slice is just index
        if volume_index == 0:
            slice_index = index
        # if volume index is larger than 1, its the cumulative sum of slices of volumes before subtracted
        # from the index
        else:
            slice_index = index - self.cumulative_slice_sum[volume_index - 1] 
        
        return volume_index, slice_index 
    
    def get_data_from_indecies(self, volume_index, slice_index) -> torch.Tensor:
        file = self.file_list[volume_index]
        with h5py.File(file, 'r') as fr:
            dataset = fr[self.data_key]
            assert isinstance(dataset, h5py.Dataset)
            # Load in the requested canonical order.
            data = torch.as_tensor(dataset[slice_index, self._selected_contrast_indices_sorted])
            data = data[self._selected_contrast_reorder]
            data = F.center_crop(data, [self.ny, self.nx]).numpy()

        return data

    def _get_canonical_stack_guided_cached(self, volume_index: int, slice_index: int) -> np.ndarray:
        """Load all selected contrasts for a slice once (in canonical config order).

        Returns a numpy array shaped [C, ny, nx]. In guided mode, this is then reordered
        in-memory to put the target contrast in channel 0.
        """
        key = (int(volume_index), int(slice_index))
        if self._guided_last_key == key and self._guided_last_stack is not None:
            return self._guided_last_stack

        stack = self.get_data_from_indecies(volume_index, slice_index)
        # Ensure stable ownership (avoid referencing temporary tensors).
        stack = np.array(stack, copy=True)
        self._guided_last_key = key
        self._guided_last_stack = stack
        return stack

    def get_single_contrast_from_indecies(self, volume_index, slice_index, contrast_local_index: int) -> torch.Tensor:
        file = self.file_list[volume_index]
        contrast_index = int(self._selected_contrast_indices[int(contrast_local_index)])

        with h5py.File(file, 'r') as fr:
            dataset = fr[self.data_key]
            assert isinstance(dataset, h5py.Dataset)
            # Load only the requested contrast to avoid reading all contrasts from disk.
            data = torch.as_tensor(dataset[slice_index, contrast_index]).unsqueeze(0)
            data = F.center_crop(data, [self.ny, self.nx]).numpy()

        return data

    def get_guided_target_first_from_indecies(
        self, volume_index, slice_index, contrast_local_index: int
    ) -> torch.Tensor:
        """Return all contrasts stacked with the target contrast first.

        Only meaningful when jointly_reconstructing=False and guided_single_contrast=True.

        Output contrast axis order:
          [target, remaining contrasts in canonical config order]
        """
        canonical = self._get_canonical_stack_guided_cached(int(volume_index), int(slice_index))
        target_idx = int(contrast_local_index)

        # Target-first packing in memory (canonical is already in config order).
        target = canonical[target_idx : target_idx + 1]
        remaining = np.delete(canonical, target_idx, axis=0)
        return np.concatenate([target, remaining], axis=0)

    def get_contrast_id(self, index: int) -> int:
        """Return the selected-contrast index for a flattened dataset index.

        Only meaningful when jointly_reconstructing=False.
        """
        if self.jointly_reconstructing:
            return -1
        num_contrasts = int(len(self._selected_contrast_order))
        return int(index) % num_contrasts

    def get_contrast_label(self, index: int) -> str:
        if self.jointly_reconstructing:
            return "joint"
        return str(self._selected_contrast_order[self.get_contrast_id(index)])


