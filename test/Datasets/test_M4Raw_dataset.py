import pytest
import numpy as np
from ml_recon.dataset.M4Raw_dataset import M4Raw
import tempfile
import os
import h5py 


SLICES = 10
VOLUMES = 3

@pytest.fixture(scope='session')
def mock_m4raw_dataset_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories for each sample
        for i in range(VOLUMES):  # Create sample directories
            file_path = os.path.join(temp_dir, f'data_{i}.h5')
            with h5py.File(file_path, 'w') as f:
                # Create a mock k-space dataset
                kspace_data = np.random.rand(3, SLICES, 4, 256, 256) + 1j * np.random.rand(3, SLICES, 4, 256, 256)
                f.create_dataset('kspace', data=kspace_data)
                
                # Create a mock contrasts dataset
                contrasts = np.array(['t1', 't2', 'flair'], dtype='S')
                f.create_dataset('contrasts', data=contrasts)
        
        yield temp_dir

def test_dataset_length(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'])

    assert len(dataloader) == VOLUMES * SLICES 

def test_dataset_outupt(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'])
    
    assert isinstance(dataloader[0], np.ndarray)
    assert dataloader[0].shape == (3, 4, 256, 256)
    assert dataloader[0].dtype == np.complex128
    assert dataloader.contrast_order == ['t1', 't2', 'flair']

def test_single_contrast(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1'])
    
    assert isinstance(dataloader[0], np.ndarray)
    assert dataloader[0].shape == (1, 4, 256, 256)
    assert dataloader[0].dtype == np.complex128
    assert dataloader.contrast_order == ['t1']

def test_volumes_not_same(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1'])
    
    assert (dataloader[0] != dataloader[1]).any(), \
    "should have different values between slices"


def test_single_contrast_mode_expands_length(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'], jointly_reconstructing=False)
    assert len(dataloader) == (VOLUMES * SLICES) * 3
    assert dataloader[0].shape == (1, 4, 256, 256)
    assert dataloader.contrast_order == ['single']


def test_guided_single_contrast_target_first_packing():
    # Build a tiny deterministic dataset with constant-valued contrasts so we can check ordering.
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'data_0.h5')
        with h5py.File(file_path, 'w') as f:
            # contrasts: ['t1','t2','flair'] in file order
            # shape: (C, S, coils, H, W)
            kspace_data = np.zeros((3, 1, 1, 8, 8), dtype=np.complex128)
            kspace_data[0, ...] = 1 + 0j  # t1
            kspace_data[1, ...] = 2 + 0j  # t2
            kspace_data[2, ...] = 3 + 0j  # flair
            # Use a non-'kspace' key so the loader won't apply fill_missing_k_space,
            # which would zero out this tiny synthetic example.
            f.create_dataset('kspace_raw', data=kspace_data)
            contrasts = np.array(['t1', 't2', 'flair'], dtype='S')
            f.create_dataset('contrasts', data=contrasts)

        ds = M4Raw(
            temp_dir,
            nx=8,
            ny=8,
            data_key='kspace_raw',
            contrasts=['t1', 't2', 'flair'],
            jointly_reconstructing=False,
            guided_single_contrast=True,
        )

        # Index mapping: base slice 0, then target local index cycles.
        # target=t1 -> [t1, t2, flair]
        x0 = ds[0]
        assert x0.shape[0] == 3
        assert np.allclose(x0[0].real.mean(), 1.0)
        assert np.allclose(x0[1].real.mean(), 2.0)
        assert np.allclose(x0[2].real.mean(), 3.0)

        # target=t2 -> [t2, t1, flair]
        x1 = ds[1]
        assert np.allclose(x1[0].real.mean(), 2.0)
        assert np.allclose(x1[1].real.mean(), 1.0)
        assert np.allclose(x1[2].real.mean(), 3.0)

        # target=flair -> [flair, t1, t2]
        x2 = ds[2]
        assert np.allclose(x2[0].real.mean(), 3.0)
        assert np.allclose(x2[1].real.mean(), 1.0)
        assert np.allclose(x2[2].real.mean(), 2.0)

if __name__ == '__main__':
    pytest.main()
