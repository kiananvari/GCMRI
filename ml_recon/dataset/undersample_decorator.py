import numpy as np
from numpy.typing import NDArray
import torch
import os
import math
from torch.utils.data import Dataset

from typing import Union, Callable, Optional, Tuple
from ml_recon.utils.undersample_tools import (
    apply_undersampling_from_dist, 
    gen_pdf_columns,
    gen_pdf_bern, 
    ssdu_gaussian_selection
)

def get_unique_filename(base_name, extension=".png"):
    counter = 0
    filename = f"{base_name}{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}{counter}{extension}"
    return filename


PROBABILITY_DIST = os.environ.get('PROBABILITY_DIST', '')
class UndersampleDecorator(Dataset):
    """Decorator class that can be used on all datasets present in the dataset folder.
    The decorator wraps the original dataset and undersamples it based on parameters
    here. Can further do self supervised undersampling.

    """

    def __init__(
        self, 
        dataset, 
        R: float = 4, 
        R_values: Optional[list[float]] = None,
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
        self_supervised: bool = False, 
        R_hat: float = math.nan,
        original_ssdu_partioning: bool = False,
        sampling_method: str = '2d', 
        same_mask_every_epoch: bool = False,
        seed: Union[int, None] = 8,
        permute_contrasts: bool = False,
        num_permutations_per_slice: int = 1,
        permutation_seed: int = 0,
        include_identity_permutation: bool = True,
        exclude_contrasts: Optional[list[str]] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.contrasts = int(getattr(dataset, 'num_contrasts_per_sample', dataset[0].shape[0]))
        self.contrast_order = dataset.contrast_order # type: ignore
        self.metrics_contrast_order = getattr(dataset, 'metrics_contrast_order', self.contrast_order)
        self.sampling_type = sampling_method
        self.R_values = [float(r) for r in R_values] if R_values else None
        self.R = float(R)
        self.R_hat = R_hat
        self.acs_lines = acs_lines
        self.original_ssdu_partioning = original_ssdu_partioning
        self.same_mask_every_epoch = same_mask_every_epoch
        self.self_supervised = self_supervised
        self.seed = seed

        self.permute_contrasts = bool(permute_contrasts)
        self.num_permutations_per_slice = int(num_permutations_per_slice)
        self.permutation_seed = int(permutation_seed)
        self.include_identity_permutation = bool(include_identity_permutation)

        self.exclude_contrasts = [str(c).lower() for c in exclude_contrasts] if exclude_contrasts else []
        self.exclude_contrast_indices: list[int] = []
        self.include_contrast_mask: Optional[np.ndarray] = None
        if self.exclude_contrasts:
            contrast_order = [str(c).lower() for c in self.contrast_order]
            missing = [c for c in self.exclude_contrasts if c not in contrast_order]
            if missing:
                raise KeyError(
                    f"Excluded contrast(s) not found in dataset contrast order: {missing}. "
                    f"Available: {contrast_order}"
                )
            include_mask = np.ones(len(contrast_order), dtype=np.float32)
            for c in self.exclude_contrasts:
                idx = contrast_order.index(c)
                include_mask[idx] = 0.0
                self.exclude_contrast_indices.append(int(idx))
            self.include_contrast_mask = include_mask

        if self.num_permutations_per_slice < 1:
            raise ValueError("num_permutations_per_slice must be >= 1")

        # Only makes sense for multi-contrast samples.
        if self.permute_contrasts and self.contrasts <= 1:
            raise ValueError("permute_contrasts=True requires multi-contrast samples")

        # Avoid accidental mixing with explicit single-contrast dataset modes.
        if self.permute_contrasts and hasattr(dataset, 'jointly_reconstructing'):
            if bool(getattr(dataset, 'jointly_reconstructing')) is False:
                raise ValueError(
                    "permute_contrasts=True is intended for jointly_reconstructing=True (multi in/out) datasets"
                )

        self._base_len = int(len(dataset))
        self._effective_len = self._base_len * (self.num_permutations_per_slice if self.permute_contrasts else 1)

        # setting seeds for random masks
        rng = np.random.default_rng(seed)


        if self.sampling_type in ['2d', 'pi']:
            pdf_generator = gen_pdf_bern
        elif self.sampling_type == '1d':
            pdf_generator = gen_pdf_columns
        else:
            raise ValueError(f'Wrong sampling type! {self.sampling_type}')

        self.omega_prob = pdf_generator(dataset.nx, dataset.ny, 1 / self.R, poly_order, acs_lines)
        self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        self.omega_prob_by_r: dict[float, NDArray] = {}
        if self.R_values:
            for r in self.R_values:
                prob = pdf_generator(dataset.nx, dataset.ny, 1 / float(r), poly_order, acs_lines)
                self.omega_prob_by_r[float(r)] = np.tile(prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        
        if PROBABILITY_DIST == '':
            self.lambda_prob = pdf_generator(dataset.nx, dataset.ny, 1/R_hat, poly_order, acs_lines) 
            self.lambda_prob = np.tile(self.lambda_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        else:
            self.lambda_prob = torch.load(PROBABILITY_DIST)

        if self.same_mask_every_epoch:
            self.lambda_seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=self._effective_len)
        else:
            self.lambda_seeds = None

        self.transforms = transforms


    def __len__(self):
        return int(self._effective_len)

    def _get_contrast_permutation(self, base_index: int, perm_id: int) -> NDArray[np.int64]:
        """Deterministically generate one of K permutations for a given base sample.

        We include identity by default, then fill the remaining slots with unique random permutations
        (seeded by `permutation_seed + base_index`).

        Returns `perm` such that `x_perm = x[perm]`.
        """
        c = int(self.contrasts)
        if not self.permute_contrasts or self.num_permutations_per_slice <= 1:
            return np.arange(c, dtype=np.int64)

        rng = np.random.default_rng(int(self.permutation_seed) + int(base_index))
        perms: list[tuple[int, ...]] = []

        if self.include_identity_permutation:
            perms.append(tuple(range(c)))

        # Fill with unique random permutations.
        max_attempts = 1000
        attempts = 0
        while len(perms) < self.num_permutations_per_slice and attempts < max_attempts:
            p = tuple(int(i) for i in rng.permutation(c).tolist())
            if p not in perms:
                perms.append(p)
            attempts += 1

        # As a fallback (very unlikely for c=4), add deterministic cyclic shifts.
        if len(perms) < self.num_permutations_per_slice:
            base = np.arange(c)
            for shift in range(1, c):
                p = tuple(int(i) for i in np.roll(base, -shift).tolist())
                if p not in perms:
                    perms.append(p)
                if len(perms) >= self.num_permutations_per_slice:
                    break

        perm = list(perms[int(perm_id) % len(perms)])
        if self.exclude_contrast_indices and perm:
            if perm[0] in self.exclude_contrast_indices:
                swap_idx = None
                for j in range(1, len(perm)):
                    if perm[j] not in self.exclude_contrast_indices:
                        swap_idx = j
                        break
                if swap_idx is not None:
                    perm[0], perm[swap_idx] = perm[swap_idx], perm[0]
        return np.asarray(perm, dtype=np.int64)

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        r_value = self.R
        r_idx: Optional[int] = None
        if isinstance(index, tuple) and len(index) == 2:
            index, r_idx = index
            if self.R_values is not None:
                r_value = float(self.R_values[int(r_idx) % len(self.R_values)])
        elif not isinstance(index, int):
            raise TypeError(f"index must be int or (int, int); got {type(index)!r}")

        if self.permute_contrasts:
            base_index = int(index) // int(self.num_permutations_per_slice)
            perm_id = int(index) % int(self.num_permutations_per_slice)
        else:
            base_index = int(index)
            perm_id = 0

        k_space:NDArray = self.dataset[base_index] #[con, chan, h, w] 
        fully_sampled_k_space = k_space.copy()

        zero_fill_mask = (fully_sampled_k_space != 0)
        # Use the effective (possibly expanded) index so each permutation gets its own mask sample.
        omega_prob = self.omega_prob_by_r.get(r_value, self.omega_prob)
        first_undersampled, omega_mask = self.compute_initial_mask(
            int(index),
            k_space,
            omega_prob=omega_prob,
            R=r_value,
        )
        omega_mask = omega_mask.astype(np.float32)

        # only mask where there is data. Exlcude zero filled data
        omega_mask = omega_mask * zero_fill_mask

        output = {
            'undersampled': first_undersampled, 
            'fs_k_space': fully_sampled_k_space,
            'mask': omega_mask,
            'loss_mask': np.ones_like(omega_mask)
        }

        # If the underlying dataset is in "single-contrast" mode, propagate which contrast this
        # sample corresponds to so metrics can be aggregated like multi-contrast runs.
        if hasattr(self.dataset, 'get_contrast_id'):
            try:
                contrast_id = int(self.dataset.get_contrast_id(index))  # type: ignore
            except Exception:
                contrast_id = -1
            output['contrast_id'] = np.array([contrast_id], dtype=np.int64)

        if self.self_supervised:
            input_mask, loss_mask = self.create_self_supervised_masks(first_undersampled, omega_mask, index)
            output.update(
                {
                    'mask': input_mask.astype(np.float32),
                    'loss_mask': loss_mask.astype(np.float32),
                    'is_self_supervised': np.array([True])
                }
            )
        else:
            output.update(
                {
                    'is_self_supervised': np.array([False])
                }
            )

        # Contrast permutation ablation (multi in/out): apply permutation consistently across all
        # contrast-indexed tensors and export metadata so losses/metrics can be logged canonically.
        perm = None
        if self.permute_contrasts:
            perm = self._get_contrast_permutation(base_index=base_index, perm_id=perm_id)
            for key in ('undersampled', 'fs_k_space', 'mask', 'loss_mask'):
                output[key] = output[key][perm]
            output['contrast_perm'] = perm
            output['perm_id'] = np.array([perm_id], dtype=np.int64)

        if self.include_contrast_mask is not None:
            include_mask = self.include_contrast_mask
            if perm is not None:
                include_mask = include_mask[perm]
            output['include_contrast_mask'] = include_mask[:, np.newaxis, np.newaxis]

        for keys in output:
            output[keys] = torch.from_numpy(output[keys])

        if self.transforms: 
            output = self.transforms(output)


        return output

    def create_self_supervised_masks(self, under, mask_omega, index):
        if self.original_ssdu_partioning:
            input_mask, loss_mask = ssdu_gaussian_selection(mask_omega)

        else:
            #seed = self.lambda_seeds[index].item()
            if self.lambda_seeds is not None:
                # determenistic masks (same mask every epoch)
                _, mask_lambda = apply_undersampling_from_dist(self.lambda_seeds[index].item(), self.lambda_prob, under)
            else:
                _, mask_lambda = apply_undersampling_from_dist(None, self.lambda_prob, under)

            # loss mask is the disjoint set of the input mask
            input_mask = mask_omega * mask_lambda
            loss_mask = mask_omega * (1 - mask_lambda)

        return input_mask, loss_mask

    def compute_initial_mask(self, index, k_space, *, omega_prob: Optional[NDArray] = None, R: Optional[float] = None):
        if omega_prob is None:
            omega_prob = self.omega_prob
        if R is None:
            R = self.R
        # same mask every time since the random seed is the index value
        if self.sampling_type in ['2d', '1d']: 
            under, mask_omega  = apply_undersampling_from_dist(
                index + self.seed,
                omega_prob,
                k_space, 
            )
        elif self.sampling_type == 'pi':
            mask_omega = np.zeros_like(k_space, dtype=bool)
            for i in range(mask_omega.shape[0]):
                mask_omega[i, ..., i::int(R)] = 1
            w = mask_omega.shape[-1]
            mask_omega[..., w//2-self.acs_lines//2:w//2+self.acs_lines//2] = 1
            under = k_space * mask_omega
        else:
            raise ValueError(f'Could not load {self.sampling_type}')

        return under, mask_omega
