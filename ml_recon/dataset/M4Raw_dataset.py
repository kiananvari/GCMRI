import numpy as np
import os
from pathlib import Path
from typing import Union, Callable, List, Optional
import h5py

import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset

class M4Raw(Dataset):
    """
    This is a dataloader for the M4Raw dataset. It loads a slice from the M4Raw 
    dataset without any subsampling.
    Attributes:
        nx (int): The desired width of the k-space data.
        ny (int): The desired height of the k-space data.
        key (str): The key to access the k-space data in the HDF5 files.
        transforms (Callable, optional): A function/transform to apply to the k-space data.
        contrast_order (np.ndarray): The order of contrasts in the dataset.
        contrast_order_indexes (np.ndarray): Boolean array indicating which contrasts are used.
        slice_cumulative_sum (np.ndarray): Cumulative sum of slices in the dataset.
        length (int): Total number of slices in the dataset.
        file_names (List[str]): List of file paths for the dataset.
    Methods:
        __len__(): Returns the total number of slices in the dataset.
        __getitem__(index): Returns the k-space data for the given index.
        get_data_from_file(index): Retrieves the k-space data from the file for the given index.
        center_k_space(contrast_k): Centers the k-space data for each contrast image.
        resample_or_pad(k_space): Resamples or pads the k-space data to the desired height and width.
    """

    def __init__(
            self,
            data_dir: Union[str, Path],
            nx:int = 256,
            ny:int = 256,
            transforms: Union[Callable, None] = None, 
            data_key:str = 'kspace',
            contrasts: List[str] = ['t1', 't2', 'flair'], 
            jointly_reconstructing: bool = True,
            guided_single_contrast: bool = False,
            limit_volumes: Optional[Union[int, float]] = None
            ):

        # call super constructor
        super().__init__()
        if isinstance(data_dir, str): 
            data_dir = Path(data_dir)
        self.nx = nx
        self.ny = ny
        self.key = data_key
        self.jointly_reconstructing = jointly_reconstructing
        self.guided_single_contrast = guided_single_contrast

        if self.guided_single_contrast and self.jointly_reconstructing:
            raise ValueError(
                "guided_single_contrast=True is only valid when jointly_reconstructing=False"
            )

        self.transforms = transforms

        files = list(data_dir.iterdir())
        self.file_names = []
        slices = []
        contrast_order = []

        files.sort()

        if limit_volumes is None:
            limit_volumes = len(files)
        elif isinstance(limit_volumes, float):
            limit_volumes = int(limit_volumes * len(files))
            
        for file_path in files[:limit_volumes]:
            self.file_names.append(file_path)

            with h5py.File(file_path, 'r') as fr:
                dataset = fr[self.key]
                assert isinstance(dataset, h5py.Dataset)
                slices.append(dataset.shape[1])
                contrast_dataset = fr['contrasts']
                assert isinstance(contrast_dataset, h5py.Dataset)
                contrast_order = np.char.lower(contrast_dataset[:].astype('U'))

        # Canonical contrast ordering based on config list (not file order).
        requested = np.array([c.lower() for c in contrasts], dtype='U')
        contrast_to_index = {str(name): int(i) for i, name in enumerate(contrast_order.tolist())}
        selected_indices = []
        for c in requested.tolist():
            if str(c) not in contrast_to_index:
                raise KeyError(
                    f"Requested contrast {c!r} not found in file contrast list: {contrast_order.tolist()}"
                )
            selected_indices.append(contrast_to_index[str(c)])

        self._selected_contrast_order = requested
        self._selected_contrast_indices = np.array(selected_indices, dtype=int)
        # h5py requires fancy indices to be in increasing order; keep sorted indices for I/O
        # and a permutation to restore canonical config ordering.
        self._selected_contrast_indices_sorted = np.array(sorted(selected_indices), dtype=int)
        pos = {int(idx): int(i) for i, idx in enumerate(self._selected_contrast_indices_sorted.tolist())}
        self._selected_contrast_reorder = np.array([pos[int(i)] for i in self._selected_contrast_indices.tolist()], dtype=int)
        self.metrics_contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]

        # Keep mask for compatibility; this is in file order.
        self.contrast_order_indexes = np.isin(contrast_order, self._selected_contrast_order)

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

        self.slice_cumulative_sum = np.cumsum(slices) 
        self.length = self.slice_cumulative_sum[-1]

        # Small per-worker cache for guided mode.
        # Avoid rereading the same HDF5 slice C times when indices are grouped.
        self._guided_last_key: Optional[tuple[int, int]] = None
        self._guided_last_stack: Optional[np.ndarray] = None
        print(f'Found {self.length} slices!')
        print(f'Found {self.contrast_order} contrats!!')


    def __len__(self):
        if self.jointly_reconstructing:
            return int(self.length)
        return int(self.length) * int(len(self._selected_contrast_order))

    def __getitem__(self, index):
        if self.jointly_reconstructing:
            k_space = self.get_data_from_file(index)
        else:
            num_contrasts = int(len(self._selected_contrast_order))
            base_index = int(index) // num_contrasts
            contrast_local_index = int(index) % num_contrasts
            if self.guided_single_contrast:
                k_space = self.get_guided_target_first_from_file(base_index, contrast_local_index)
            else:
                k_space = self.get_single_contrast_from_file(base_index, contrast_local_index)
        k_space = self.resample_or_pad(k_space)


        if self.transforms:
            k_space = self.transforms(k_space)

        return k_space
    
    def get_data_from_file(self, index):
        volume_index, slice_index = self.get_file_indecies(index)
        cur_file = self.file_names[volume_index]
        
        
        with h5py.File(cur_file, 'r') as fr:
            dataset = fr[self.key]
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[self._selected_contrast_indices_sorted, slice_index]
            k_space = k_space[self._selected_contrast_reorder]
            if self.key == 'kspace':
                k_space = self.fill_missing_k_space(k_space) 
                
        return k_space 

    def _get_canonical_stack_guided_cached(self, base_index: int) -> np.ndarray:
        """Load all selected contrasts for a slice once (in canonical config order)."""
        volume_index, slice_index = self.get_file_indecies(int(base_index))
        key = (int(volume_index), int(slice_index))
        if self._guided_last_key == key and self._guided_last_stack is not None:
            return self._guided_last_stack

        cur_file = self.file_names[volume_index]
        with h5py.File(cur_file, 'r') as fr:
            dataset = fr[self.key]
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[self._selected_contrast_indices_sorted, slice_index]
            k_space = k_space[self._selected_contrast_reorder]
            if self.key == 'kspace':
                k_space = self.fill_missing_k_space(k_space)

        # Own the memory in the cache.
        k_space = np.array(k_space, copy=True)
        self._guided_last_key = key
        self._guided_last_stack = k_space
        return k_space

    def get_single_contrast_from_file(self, base_index: int, contrast_local_index: int):
        volume_index, slice_index = self.get_file_indecies(base_index)
        cur_file = self.file_names[volume_index]
        contrast_index = int(self._selected_contrast_indices[int(contrast_local_index)])

        with h5py.File(cur_file, 'r') as fr:
            dataset = fr[self.key]
            assert isinstance(dataset, h5py.Dataset)
            # Load only requested contrast.
            k_space = dataset[contrast_index, slice_index][np.newaxis, ...]
            if self.key == 'kspace':
                k_space = self.fill_missing_k_space(k_space)

        return k_space

    def get_guided_target_first_from_file(self, base_index: int, contrast_local_index: int):
        """Return all contrasts stacked with the target contrast first.

        Only meaningful when jointly_reconstructing=False and guided_single_contrast=True.

        Output contrast axis order:
          [target, remaining contrasts in canonical config order]
        """
        canonical = self._get_canonical_stack_guided_cached(int(base_index))
        target_idx = int(contrast_local_index)

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

    def get_file_indecies(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        return volume_index,slice_index

   
    def resample_or_pad(self, k_space):
        """Takes k-space data and resamples data to desired height and width. If 
        the image is larger, we crop. If the image is smaller, we pad with zeros

        Args:
            k_space (np.ndarray): k_space to be cropped or padded 
            reduce_fov (bool, optional): If we should reduce fov along readout dimension. Defaults to True.

        Returns:
            np.ndarray: cropped k_space
        """
        resample_height = self.ny
        resample_width = self.nx

        return F.center_crop(torch.from_numpy(k_space), [resample_height, resample_width]).numpy()
    
    @staticmethod
    def fill_missing_k_space(k_space):
        # if there is missing data on one of the coils, we replace it with very samll number.
        # for some reason there are zero values in k-space. We replace it with small numbers to 
        # help facilitate masking 
        
        contrast, coils, h, w = k_space.shape
        zero_fill_mask = np.ones_like(k_space) 
        zero_fill_mask[:, :, :, :29] = 0
        zero_fill_mask[:, :, :, -29:] = 0
    
        
        k_space[k_space == 0] = 1e-6
        k_space = k_space * zero_fill_mask

        return k_space

