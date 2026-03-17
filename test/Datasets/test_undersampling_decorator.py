import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from ml_recon.dataset.FastMRI_dataset import FastMRIDataset
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from test.Datasets.test_BraTS_dataset import mock_brats_dataset_dir


ACS_LINES = 10
@pytest.fixture
def supervised_dataset(mock_brats_dataset_dir) -> UndersampleDecorator:
    dataset = BratsDataset(mock_brats_dataset_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(dataset, R=8, acs_lines=ACS_LINES, self_supervised=False)
    return undersample_dataset


def test_undersampled_slice(supervised_dataset):
    data = next(iter(supervised_dataset))

    width = data['undersampled'].shape[-1]
    center_width = np.floor(width/2).astype(int)
    height = data['undersampled'].shape[-2]
    center_height = np.floor(height/2).astype(int)

    slice_y = slice(center_height - ACS_LINES//2, center_height + ACS_LINES//2)
    slice_x = slice(center_width - ACS_LINES//2, center_width + ACS_LINES//2)
    acs = data['undersampled'][..., slice_y, slice_x]
    acs_mask = data['mask'][..., slice_y, slice_x]
       
    assert (acs != 0).all()
    assert (acs_mask == 1).all()

def test_fully_sampled(supervised_dataset):
    data = supervised_dataset[0]

    assert (data['fs_k_space'] != 0).all()


def test_non_deterministic(supervised_dataset):
    data1 = supervised_dataset[0]
    data2 = supervised_dataset[0]

    # should have same undersampling for each slice
    assert ((data1['undersampled'] != 0) == (data2['undersampled'] != 0)).all()

def test_non_deterministic_between_slices(supervised_dataset):
    data1 = supervised_dataset[0]
    data2 = supervised_dataset[1]
    
    # shouldn't have same undersampling between slices
    assert ((data1['undersampled'] != 0) != (data2['undersampled'] != 0)).any()

@pytest.mark.parametrize('sampling_method', ['2d', '1d', 'pi'])
def test_non_deterministic_self_supervsied(mock_brats_dataset_dir, sampling_method):
    dataset = BratsDataset(mock_brats_dataset_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(
        dataset, 
        R=4, 
        acs_lines=ACS_LINES, 
        self_supervised=True, 
        R_hat=2, 
        sampling_method=sampling_method 
        )
    
    data1 = undersample_dataset[0]
    data2 = undersample_dataset[0]

    # inital undersampling should be the same
    assert ((data1['undersampled'] != 0) == (data2['undersampled'] != 0)).all()
    # partitioning masks should be different
    assert ((data1['mask']) != (data2['mask'])).any()
    assert ((data1['loss_mask']) != (data2['loss_mask'])).any()
    torch.testing.assert_close(data1['fs_k_space'], data2['fs_k_space'])



def test_inital_mask_determenistic(supervised_dataset):
    data1 = supervised_dataset[0]
    data2 = supervised_dataset[0]
    
    # should have the same mask
    assert ((data1['undersampled'] != 0) == (data2['undersampled'] != 0)).all()


def test_contrast_permutation_expands_and_reorders(mock_brats_dataset_dir):
    base_ds = BratsDataset(mock_brats_dataset_dir, nx=128, ny=128)
    base_len = len(base_ds)

    perm_ds = UndersampleDecorator(
        base_ds,
        R=8,
        acs_lines=ACS_LINES,
        self_supervised=False,
        permute_contrasts=True,
        num_permutations_per_slice=4,
        permutation_seed=0,
        include_identity_permutation=True,
    )

    assert len(perm_ds) == base_len * 4

    # Base (canonical) sample
    base = base_ds[0]

    # perm_id=0 should be identity by default
    s0 = perm_ds[0]
    assert 'contrast_perm' in s0
    assert tuple(s0['contrast_perm'].tolist()) == (0, 1, 2, 3)
    torch.testing.assert_close(s0['fs_k_space'], torch.from_numpy(base))

    # perm_id=1..3 should be valid permutations and reorder fs_k_space accordingly
    for i in range(1, 4):
        si = perm_ds[i]
        perm = si['contrast_perm'].tolist()
        assert sorted(perm) == [0, 1, 2, 3]
        expected = torch.from_numpy(base[perm])
        torch.testing.assert_close(si['fs_k_space'], expected)

@pytest.mark.parametrize('sampling_method', ['2d', '1d', 'pi'])
def test_non_deterministic_self_supervsied(mock_brats_dataset_dir, sampling_method):
    dataset = BratsDataset(mock_brats_dataset_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(
        dataset, 
        R=4, 
        acs_lines=ACS_LINES, 
        self_supervised=True, 
        R_hat=2, 
        sampling_method=sampling_method, 
        )
    
    undersample_dataset.set_epoch(0)
    data1 = undersample_dataset[0]
    undersample_dataset.set_epoch(1)
    data2 = undersample_dataset[0]

    # inital undersampling should be the same
    assert ((data1['undersampled'] != 0) == (data2['undersampled'] != 0)).all()
    # partitioning masks should be different
    assert ((data1['mask']) != (data2['mask'])).any()
    assert ((data1['loss_mask']) != (data2['loss_mask'])).any()
    torch.testing.assert_close(data1['fs_k_space'], data2['fs_k_space'])
