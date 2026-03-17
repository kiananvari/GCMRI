# import standard lib modules
from typing import Optional, Union, Literal, Sized
from pathlib import Path
import os

# import DL modules
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, Sampler
import numpy as np

# import my modules
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.utils import k_to_img
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.M4Raw_dataset import M4Raw
from ml_recon.dataset.FastMRI_dataset import FastMRIDataset
from ml_recon.dataset.FastMRI_knee_dataset import FastMRIKneeDataset
from ml_recon.dataset.test_dataset import TestDataset


ACS_LINES = int(os.getenv('ACS_LINES') or 10)

class UndersampledDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_name: Literal['fastmri', 'fastmri_knee', 'm4raw', 'brats'],
        data_dir: str,
        test_dir: Optional[str] = None,
        batch_size: int = 1,
        R: float = 6,
        R_values: Optional[list[float]] = None,
        R_curriculum_stages: Optional[object] = None,
        R_hat: float = 2.0,
        contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
        resolution: tuple[int, int] = (128, 128),
        num_workers: int = 0,
        poly_order: int = 8,
        norm_method: Union[Literal['k', 'img', 'image_mean', 'image_mean2', 'std'], None] = 'image_mean',
        self_supervised: bool = False,
        sampling_method: str = '2d',
        ssdu_partitioning: bool = False,
        limit_volumes: Optional[Union[int, float]] = None,
        same_mask_every_epoch: bool = False,
        seed: int = 8,
        jointly_reconstructing: bool = True,
        guided_single_contrast: bool = False,
        permute_contrasts: bool = False,
        num_permutations_per_slice: int = 1,
        permutation_seed: int = 0,
        include_identity_permutation: bool = True,
        contrast_excluding_training: Optional[list[str]] = None,
        val_permute_contrasts: bool = False,
        val_num_permutations_per_slice: int = 1,
        val_permutation_seed: int = 0,
        val_include_identity_permutation: bool = True,
        validate_on_target: bool = False,
        target_dataset: Optional[str] = None,
        target_dataset_path: Optional[str] = None,
        target_contrasts: Optional[list[str]] = None,
    ):
        """
            dataset_name:

        """

        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.test_dir = Path(test_dir) if test_dir else self.data_dir
        self.contrasts = contrasts
        self.num_contrasts = len(contrasts)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.resolution = resolution
        self.R = R
        self.R_values = [float(r) for r in R_values] if R_values else None
        self.R_curriculum_stages = R_curriculum_stages
        self._curriculum_sampler: Optional[Sampler] = None
        self.R_hat = R_hat
        self.poly_order = poly_order
        self.self_supervised = self_supervised
        self.ssdu_partioning = ssdu_partitioning
        self.sampling_method = sampling_method
        self.norm_method = norm_method
        self.limit_volumes = limit_volumes
        self.same_mask_every_epoch = same_mask_every_epoch
        self.seed=seed
        self.jointly_reconstructing = jointly_reconstructing
        self.guided_single_contrast = guided_single_contrast
        self.permute_contrasts = permute_contrasts
        self.num_permutations_per_slice = num_permutations_per_slice
        self.permutation_seed = permutation_seed
        self.include_identity_permutation = include_identity_permutation
        self.contrast_excluding_training = contrast_excluding_training
        self.val_permute_contrasts = val_permute_contrasts
        self.val_num_permutations_per_slice = val_num_permutations_per_slice
        self.val_permutation_seed = val_permutation_seed
        self.val_include_identity_permutation = val_include_identity_permutation
        self.validate_on_target = validate_on_target
        self.target_dataset = target_dataset
        self.target_dataset_path = Path(target_dataset_path) if target_dataset_path else None
        self.target_contrasts = target_contrasts

        self.dataset_class, self.test_data_key = self.setup_dataset_type(dataset_name)
        self.transforms = self.setup_data_normalization(norm_method)


    def setup(self, stage):
        super().setup(stage)

        # Get directories for different split folders.
        # Support both:
        # - a root directory containing train/val/test subfolders
        # - a directory that is already a split folder.
        train_dir = self.data_dir / 'train' if (self.data_dir / 'train').exists() else self.data_dir
        val_dir = self.data_dir / 'val' if (self.data_dir / 'val').exists() else self.data_dir

        if (self.test_dir / 'test').exists():
            test_dir = self.test_dir / 'test'
        else:
            test_dir = self.test_dir


        # keywords to control dataset data
        dataset_keyword_args = {
            'nx': self.resolution[0], 
            'ny': self.resolution[1],
            'contrasts': self.contrasts, 
            'limit_volumes': self.limit_volumes,
            'jointly_reconstructing': self.jointly_reconstructing,
            'guided_single_contrast': self.guided_single_contrast,
        }

        # keywords to control k-space undersamplig 
        undersample_keyword_args = {
            'R': self.R,
            'R_values': self.R_values,
            'R_hat': self.R_hat,
            'sampling_method': self.sampling_method,
            'self_supervised': self.self_supervised,
            'acs_lines' : ACS_LINES, 
            'poly_order': self.poly_order,
            'original_ssdu_partioning': self.ssdu_partioning,
            'same_mask_every_epoch': self.same_mask_every_epoch, #same mask every epoch
            'seed': self.seed
        }

        # Optional permutation ablation (multi in/out): expand dataset length and permute contrast axis.
        undersample_keyword_args.update(
            {
                'permute_contrasts': self.permute_contrasts,
                'num_permutations_per_slice': self.num_permutations_per_slice,
                'permutation_seed': self.permutation_seed,
                'include_identity_permutation': self.include_identity_permutation,
            }
        )

        train_undersample_keyword_args = dict(undersample_keyword_args)
        if self.contrast_excluding_training:
            train_undersample_keyword_args['exclude_contrasts'] = self.contrast_excluding_training

        # Validation permutation (optional). By default, keep canonical contrast order.
        val_undersample_keyword_args = dict(undersample_keyword_args)
        val_undersample_keyword_args.update(
            {
                'permute_contrasts': self.val_permute_contrasts,
                'num_permutations_per_slice': self.val_num_permutations_per_slice if self.val_permute_contrasts else 1,
                'permutation_seed': self.val_permutation_seed,
                'include_identity_permutation': self.val_include_identity_permutation,
            }
        )

        test_undersample_keyword_args = dict(undersample_keyword_args)
        test_undersample_keyword_args.update({'permute_contrasts': False, 'num_permutations_per_slice': 1})

        # undersampled training dataset
        self.train_dataset = UndersampleDecorator(
            self.dataset_class(
                train_dir, 
                **dataset_keyword_args
            ),
            transforms=self.transforms,
            **train_undersample_keyword_args
        )


        # undersampled validation dataset(s)
        self.val_r_values = list(self.R_values) if self.R_values else None
        if self.val_r_values:
            self.val_datasets = []
            for r in self.val_r_values:
                val_args = dict(val_undersample_keyword_args)
                val_args["R"] = float(r)
                val_args["R_values"] = None
                self.val_datasets.append(
                    UndersampleDecorator(
                        self.dataset_class(
                            val_dir,
                            **dataset_keyword_args
                        ),
                        transforms=self.transforms,
                        **val_args
                    )
                )
        else:
            self.val_dataset = UndersampleDecorator(
                self.dataset_class(
                    val_dir, 
                    **dataset_keyword_args
                ),
                transforms=self.transforms,
                **val_undersample_keyword_args
            )

        # undersampled test dataset(s)
        self.test_r_values = list(self.R_values) if self.R_values else None
        if self.test_r_values:
            self.test_datasets = []
            for r in self.test_r_values:
                test_args = dict(test_undersample_keyword_args)
                test_args["R"] = float(r)
                test_args["R_values"] = None
                self.test_datasets.append(
                    UndersampleDecorator(
                        self.dataset_class(
                            test_dir,
                            **dataset_keyword_args
                        ),
                        **test_args,
                        transforms=self.transforms,
                    )
                )
        else:
            self.test_dataset = UndersampleDecorator(
                self.dataset_class(
                    test_dir, 
                    **dataset_keyword_args
                ),
                **test_undersample_keyword_args,
                transforms=self.transforms,
            )

        self.contrast_order = self.train_dataset.contrast_order
        self.metrics_contrast_order = getattr(self.train_dataset, 'metrics_contrast_order', self.contrast_order)

        self.target_val_datasets = None
        self.target_val_dataset = None
        self.target_test_datasets = None
        self.target_test_dataset = None

        if self.validate_on_target:
            if not self.target_dataset or not self.target_dataset_path:
                raise ValueError("validate_on_target requires target_dataset and target_dataset_path")

            target_dataset_class, _ = self.setup_dataset_type(self.target_dataset)
            target_contrasts = self.target_contrasts or self.contrasts

            target_dataset_keyword_args = {
                'nx': self.resolution[0],
                'ny': self.resolution[1],
                'contrasts': target_contrasts,
                'limit_volumes': self.limit_volumes,
                'jointly_reconstructing': self.jointly_reconstructing,
                'guided_single_contrast': self.guided_single_contrast,
            }

            target_val_dir = self.target_dataset_path / 'val' if (self.target_dataset_path / 'val').exists() else self.target_dataset_path
            if (self.target_dataset_path / 'test').exists():
                target_test_dir = self.target_dataset_path / 'test'
            else:
                target_test_dir = self.target_dataset_path

            if self.val_r_values:
                self.target_val_datasets = []
                for r in self.val_r_values:
                    val_args = dict(val_undersample_keyword_args)
                    val_args["R"] = float(r)
                    val_args["R_values"] = None
                    self.target_val_datasets.append(
                        UndersampleDecorator(
                            target_dataset_class(
                                target_val_dir,
                                **target_dataset_keyword_args
                            ),
                            transforms=self.transforms,
                            **val_args
                        )
                    )
            else:
                self.target_val_dataset = UndersampleDecorator(
                    target_dataset_class(
                        target_val_dir,
                        **target_dataset_keyword_args
                    ),
                    transforms=self.transforms,
                    **val_undersample_keyword_args
                )

            if self.test_r_values:
                self.target_test_datasets = []
                for r in self.test_r_values:
                    test_args = dict(test_undersample_keyword_args)
                    test_args["R"] = float(r)
                    test_args["R_values"] = None
                    self.target_test_datasets.append(
                        UndersampleDecorator(
                            target_dataset_class(
                                target_test_dir,
                                **target_dataset_keyword_args
                            ),
                            **test_args,
                            transforms=self.transforms,
                        )
                    )
            else:
                self.target_test_dataset = UndersampleDecorator(
                    target_dataset_class(
                        target_test_dir,
                        **target_dataset_keyword_args
                    ),
                    **test_undersample_keyword_args,
                    transforms=self.transforms,
                )

        self.val_prefixes = self._build_prefixes(self.val_r_values, include_target=self.validate_on_target)
        self.test_prefixes = self._build_prefixes(self.test_r_values, include_target=self.validate_on_target)

    def train_dataloader(self):
        if self.R_values:
            base_sampler = RandomSampler(self.train_dataset)
            self._curriculum_sampler = _CurriculumBatchSampler(
                base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                r_values=self.R_values,
                stage_cfg=self.R_curriculum_stages,
                seed=int(self.seed),
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._curriculum_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        loaders = []

        if getattr(self, "val_r_values", None):
            loaders.extend(
                [
                    DataLoader(
                        ds,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )
                    for ds in self.val_datasets
                ]
            )
        else:
            loaders.append(
                DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )

        if self.validate_on_target:
            if self.target_val_datasets is not None:
                loaders.extend(
                    [
                        DataLoader(
                            ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=True,
                        )
                        for ds in self.target_val_datasets
                    ]
                )
            elif self.target_val_dataset is not None:
                loaders.append(
                    DataLoader(
                        self.target_val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )
                )

        if len(loaders) == 1:
            return loaders[0]
        return loaders

    def test_dataloader(self):
        loaders = []

        if getattr(self, "test_r_values", None):
            loaders.extend(
                [
                    DataLoader(
                        ds,
                        batch_size=1,
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )
                    for ds in self.test_datasets
                ]
            )
        else:
            loaders.append(
                DataLoader(
                    self.test_dataset,
                    batch_size=1,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )

        if self.validate_on_target:
            if self.target_test_datasets is not None:
                loaders.extend(
                    [
                        DataLoader(
                            ds,
                            batch_size=1,
                            num_workers=self.num_workers,
                            pin_memory=True,
                        )
                        for ds in self.target_test_datasets
                    ]
                )
            elif self.target_test_dataset is not None:
                loaders.append(
                    DataLoader(
                        self.target_test_dataset,
                        batch_size=1,
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )
                )

        if len(loaders) == 1:
            return loaders[0]
        return loaders

    def _build_prefixes(self, r_values, *, include_target: bool) -> list[str]:
        prefixes = []
        if r_values:
            for r in r_values:
                prefixes.append(self._format_r_prefix(r))
        else:
            prefixes.append("")

        if include_target:
            if r_values:
                for r in r_values:
                    prefixes.append(f"target/{self._format_r_prefix(r)}")
            else:
                prefixes.append("target/")
        return prefixes

    @staticmethod
    def _format_r_prefix(r_val: float) -> str:
        if float(r_val).is_integer():
            r_label = f"R{int(r_val)}"
        else:
            r_label = f"R{r_val}"
        return f"{r_label}/"

    def set_curriculum_epoch(self, epoch: int) -> None:
        sampler = getattr(self, "_curriculum_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

    def setup_dataset_type(self, dataset_name):
        dataset_name = str.lower(dataset_name)
        if dataset_name == 'brats': 
            dataset_class = BratsDataset
            test_data_key = 'ground_truth'
        elif dataset_name == 'fastmri':
            dataset_class = FastMRIDataset
            test_data_key = 'reconstruction_rss'
        elif dataset_name == 'fastmri_knee':
            dataset_class = FastMRIKneeDataset
            test_data_key = 'reconstruction_rss'
        elif dataset_name == 'm4raw':
            dataset_class = M4Raw
            test_data_key = 'reconstruction_rss'
        else: 
            raise ValueError(f'{dataset_name} is not a valid dataset name')
        return dataset_class, test_data_key

    def setup_data_normalization(self, norm_method):
        if norm_method == 'img':
            transforms = normalize_image_max()
        elif norm_method == 'k': 
            transforms = normalize_k_max() 
        elif norm_method == 'norm_l2':
            transforms = normalize_l2() 
        elif norm_method == 'image_mean':
            transforms = normalize_image_mean() 
        elif norm_method == 'std':
            transforms = normalize_image_std()
        else:
            transforms = None
        return transforms


class _CurriculumBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        base_sampler: Sampler[int],
        *,
        batch_size: int,
        drop_last: bool,
        r_values: list[float],
        stage_cfg: Optional[object],
        seed: int = 0,
    ) -> None:
        self.base_sampler = base_sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.r_values = list(r_values)
        self.stage_cfg = stage_cfg
        self.seed = int(seed)
        self.epoch = 0
        self._last_stage_idx: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        stage_idx, probs = self._get_stage_index_and_probs()
        if stage_idx is not None and stage_idx != self._last_stage_idx:
            self._last_stage_idx = stage_idx
            probs_str = ", ".join(f"{p:.3f}" for p in probs)
            print(f"[Curriculum] epoch={self.epoch} stage={stage_idx} probs=[{probs_str}]")

    def _normalize_probs(self, probs: list[float]) -> list[float]:
        arr = np.asarray(probs, dtype=np.float64)
        total = float(arr.sum())
        if total <= 0:
            return (np.ones_like(arr) / len(arr)).tolist()
        return (arr / total).tolist()

    def _get_stage_index_and_probs(self) -> tuple[Optional[int], list[float]]:
        default_probs = (np.ones(len(self.r_values)) / len(self.r_values)).tolist()
        if not self.stage_cfg:
            return None, default_probs

        stages = self.stage_cfg
        if isinstance(stages, dict) and "stages" in stages:
            stages = stages.get("stages")

        if not isinstance(stages, list) or not stages:
            return None, default_probs

        epoch = self.epoch
        elapsed = 0
        for idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            stage_epochs = int(stage.get("epochs", 0))
            probs = stage.get("probs") or stage.get("p") or stage.get("weights")
            if stage_epochs <= 0 or probs is None:
                continue
            if epoch < elapsed + stage_epochs:
                return idx, self._normalize_probs(list(probs))
            elapsed += stage_epochs

        for idx, stage in reversed(list(enumerate(stages))):
            if not isinstance(stage, dict):
                continue
            probs = stage.get("probs") or stage.get("p") or stage.get("weights")
            if probs is None:
                continue
            return idx, self._normalize_probs(list(probs))

        return None, default_probs

    def _get_stage_probs(self) -> list[float]:
        _, probs = self._get_stage_index_and_probs()
        return probs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + int(self.epoch))
        probs = self._get_stage_probs()
        r_idx = int(rng.choice(len(self.r_values), p=probs))
        batch: list[tuple[int, int]] = []
        for idx in self.base_sampler:
            batch.append((int(idx), r_idx))
            if len(batch) == self.batch_size:
                yield batch
                r_idx = int(rng.choice(len(self.r_values), p=probs))
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if isinstance(self.base_sampler, Sized):
            base_len = len(self.base_sampler)
        else:
            base_len = 0
        if self.drop_last:
            return base_len // self.batch_size
        return (base_len + self.batch_size - 1) // self.batch_size

class normalize_image_max(object):
    def __call__(self, data: dict):
        input = data['fs_k_space']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        return data

class normalize_k_max(object):
    def __call__(self, data):
        input = data['undersampled']
        scaling_factor = input.abs().amax((1, 2, 3), keepdim=True)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        return data

class normalize_l2(object):
    def __call__(self, data):
        input = data['fs_k_space']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.norm(2, (1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_mean(object):
    def __call__(self, data):
        input = data['undersampled']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.mean((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_std(object):
    def __call__(self, data):
        input = data['undersampled']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.std((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

