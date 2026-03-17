import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union, Collection, Tuple
import h5py
import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset


class FastMRIKneeDataset(Dataset):
    """Slice-level dataloader for paired fastMRI knee PD/PDFS volumes.

    Expected H5 layout per volume:
      - kspace: [slices, contrasts, coils, H, W]
      - contrasts: ["pd", "pdfs"] (bytes or str)

    Returns k-space as [C, coils, H, W] in canonical contrast order.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        nx: int = 256,
        ny: int = 256,
        contrasts: Collection[str] = ("pd", "pdfs"),
        jointly_reconstructing: bool = True,
        guided_single_contrast: bool = False,
        transforms: Optional[Callable] = None,
        data_key: str = "kspace",
        limit_volumes: Optional[Union[int, float]] = None,
    ):
        assert contrasts, "Contrast list should not be empty"

        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.nx = nx
        self.ny = ny
        self.transforms = transforms
        self.contrasts = np.array([str(c).lower() for c in contrasts], dtype="U")
        self.jointly_reconstructing = jointly_reconstructing
        self.guided_single_contrast = guided_single_contrast
        self.data_key = data_key

        if self.guided_single_contrast and self.jointly_reconstructing:
            raise ValueError(
                "guided_single_contrast=True is only valid when jointly_reconstructing=False"
            )

        files = [p for p in data_dir.iterdir() if p.is_file()]
        files.sort()

        if limit_volumes is None:
            limit_volumes = len(files)
        elif isinstance(limit_volumes, float):
            limit_volumes = int(limit_volumes * len(files))

        files = files[:limit_volumes]

        slices = []
        contrast_order = []
        first = True
        for file_path in files:
            with h5py.File(file_path, "r") as fr:
                dataset = fr[self.data_key]
                assert isinstance(dataset, h5py.Dataset)
                slices.append(dataset.shape[0])
                if first:
                    contrast_dataset = fr["contrasts"]
                    assert isinstance(contrast_dataset, h5py.Dataset)
                    contrast_order = np.char.lower(contrast_dataset[:].astype("U"))
                    first = False

        # Stable contrast ordering based on requested list (not file order).
        contrast_order = np.array(contrast_order, dtype="U")
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
        self._selected_contrast_indices_sorted = np.array(sorted(selected_indices), dtype=int)
        pos = {int(idx): int(i) for i, idx in enumerate(self._selected_contrast_indices_sorted.tolist())}
        self._selected_contrast_reorder = np.array(
            [pos[int(i)] for i in self._selected_contrast_indices.tolist()], dtype=int
        )
        self.metrics_contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]

        # Keep for compatibility with any external utilities.
        self.contrast_order_indexes = np.isin(contrast_order, self._selected_contrast_order)

        if self.jointly_reconstructing:
            self.contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]
            self.num_contrasts_per_sample = int(len(self._selected_contrast_order))
        else:
            if self.guided_single_contrast:
                self.contrast_order = [str(c) for c in self._selected_contrast_order.tolist()]
                self.num_contrasts_per_sample = int(len(self._selected_contrast_order))
            else:
                self.contrast_order = ["single"]
                self.num_contrasts_per_sample = 1

        self.file_list = np.array(files)
        self.slices = np.array(slices)
        self.cumulative_slice_sum = np.cumsum(self.slices)
        self.length = int(self.cumulative_slice_sum[-1]) if len(self.cumulative_slice_sum) else 0

        self._guided_last_key: Optional[Tuple[int, int]] = None
        self._guided_last_stack: Optional[np.ndarray] = None
        print(f"Found {self.length} slices")

    def __len__(self) -> int:
        if self.jointly_reconstructing:
            return int(self.length)
        return int(self.length) * int(len(self._selected_contrast_order))

    def __getitem__(self, index) -> np.ndarray:
        if self.jointly_reconstructing:
            k_space = self.get_data_from_file(int(index))
        else:
            num_contrasts = int(len(self._selected_contrast_order))
            base_index = int(index) // num_contrasts
            contrast_local_index = int(index) % num_contrasts
            if self.guided_single_contrast:
                k_space = self.get_guided_target_first_from_file(base_index, contrast_local_index)
            else:
                k_space = self.get_single_contrast_from_file(base_index, contrast_local_index)

        if self.transforms:
            k_space = self.transforms(k_space)

        return k_space

    def get_file_indices(self, index: int) -> Tuple[int, int]:
        volume_index = int(np.sum(self.cumulative_slice_sum <= index))
        if volume_index == 0:
            slice_index = int(index)
        else:
            slice_index = int(index - self.cumulative_slice_sum[volume_index - 1])
        return volume_index, slice_index

    def get_data_from_file(self, index: int) -> np.ndarray:
        volume_index, slice_index = self.get_file_indices(int(index))
        file_path = self.file_list[volume_index]
        with h5py.File(file_path, "r") as fr:
            dataset = fr[self.data_key]
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[slice_index, self._selected_contrast_indices_sorted]
            k_space = k_space[self._selected_contrast_reorder]
            k_space = F.center_crop(torch.as_tensor(k_space), [self.ny, self.nx]).numpy()
        return k_space

    def _get_canonical_stack_guided_cached(self, base_index: int) -> np.ndarray:
        volume_index, slice_index = self.get_file_indices(int(base_index))
        key = (int(volume_index), int(slice_index))
        if self._guided_last_key == key and self._guided_last_stack is not None:
            return self._guided_last_stack

        stack = self.get_data_from_file(int(base_index))
        stack = np.array(stack, copy=True)
        self._guided_last_key = key
        self._guided_last_stack = stack
        return stack

    def get_single_contrast_from_file(self, base_index: int, contrast_local_index: int) -> np.ndarray:
        volume_index, slice_index = self.get_file_indices(int(base_index))
        file_path = self.file_list[volume_index]
        contrast_index = int(self._selected_contrast_indices[int(contrast_local_index)])
        with h5py.File(file_path, "r") as fr:
            dataset = fr[self.data_key]
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[slice_index, contrast_index][np.newaxis, ...]
            k_space = F.center_crop(torch.as_tensor(k_space), [self.ny, self.nx]).numpy()
        return k_space

    def get_guided_target_first_from_file(self, base_index: int, contrast_local_index: int) -> np.ndarray:
        canonical = self._get_canonical_stack_guided_cached(int(base_index))
        target_idx = int(contrast_local_index)
        target = canonical[target_idx : target_idx + 1]
        remaining = np.delete(canonical, target_idx, axis=0)
        return np.concatenate([target, remaining], axis=0)

    def get_contrast_id(self, index: int) -> int:
        if self.jointly_reconstructing:
            return -1
        num_contrasts = int(len(self._selected_contrast_order))
        return int(index) % num_contrasts

    def get_contrast_label(self, index: int) -> str:
        if self.jointly_reconstructing:
            return "joint"
        return str(self._selected_contrast_order[self.get_contrast_id(index)])
