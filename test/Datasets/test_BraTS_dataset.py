import tempfile
import h5py
import numpy as np
import pytest
import os


from ml_recon.dataset.BraTS_dataset import BratsDataset

@pytest.fixture
def brats_dataset(mock_brats_dataset_dir) -> BratsDataset:
    path = mock_brats_dataset_dir
    dataset = BratsDataset(path, nx=128, ny=128)
    return dataset

@pytest.fixture
def mock_brats_dataset_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories for each sample
        for i in range(3):  # Create 3 sample directories
            sample_dir = os.path.join(temp_dir, f'sample_{i}')
            os.makedirs(sample_dir)
            
            # Create a mock HDF5 file in each sample directory
            file_path = os.path.join(sample_dir, 'data_i.h5')
            with h5py.File(file_path, 'w') as f:
                # Create a mock k-space dataset
                kspace_data = np.random.rand(3, 4, 8, 130, 130) + 1j * np.random.rand(3, 4, 8, 130, 130)
                f.create_dataset('k_space', data=kspace_data)
                
                # Create a mock contrasts dataset
                contrasts = np.array(['t1', 't2', 'flair', 't1ce'], dtype='S')
                f.create_dataset('contrasts', data=contrasts)
        
        yield temp_dir


def test_init(brats_dataset):
    contrast_order = brats_dataset.contrast_order
    assert 'flair' in contrast_order
    assert 't1' in contrast_order
    assert 't1ce' in contrast_order
    assert 't2' in contrast_order

def test_data(brats_dataset):
    data = brats_dataset[0]

    assert data.ndim == 4
    assert isinstance(data, np.ndarray)
    assert data.shape == (4, 8, 128, 128)
    assert data.dtype == np.complex128


def test_data_single_contrast_mode(mock_brats_dataset_dir):
    dataset = BratsDataset(mock_brats_dataset_dir, nx=128, ny=128, jointly_reconstructing=False)

    # Base dataset has 9 slices total (3 volumes x 3 slices). In single-contrast mode, each slice
    # is expanded into one sample per selected contrast.
    assert len(dataset) == 9 * 4

    data = dataset[0]
    assert data.ndim == 4
    assert isinstance(data, np.ndarray)
    assert data.shape == (1, 8, 128, 128)
    assert dataset.contrast_order == ['single']


def test_guided_single_contrast_target_first_packing():
    # Create a minimal deterministic BraTS-like folder with one volume folder and one .h5 file.
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_dir = os.path.join(temp_dir, 'sample_0')
        os.makedirs(sample_dir)
        file_path = os.path.join(sample_dir, 'data.h5')
        with h5py.File(file_path, 'w') as f:
            # k_space shape in this repo: (slices, contrasts, coils, H, W)
            kspace_data = np.zeros((1, 4, 1, 8, 8), dtype=np.complex128)
            # file order: ['t1','t2','flair','t1ce']
            kspace_data[:, 0, ...] = 1 + 0j
            kspace_data[:, 1, ...] = 2 + 0j
            kspace_data[:, 2, ...] = 3 + 0j
            kspace_data[:, 3, ...] = 4 + 0j
            f.create_dataset('k_space', data=kspace_data)
            contrasts = np.array(['t1', 't2', 'flair', 't1ce'], dtype='S')
            f.create_dataset('contrasts', data=contrasts)

        ds = BratsDataset(
            temp_dir,
            nx=8,
            ny=8,
            contrasts=['t1', 't2', 'flair', 't1ce'],
            jointly_reconstructing=False,
            guided_single_contrast=True,
        )

        # base slice 0, target=t1 -> [t1, t2, flair, t1ce]
        x0 = ds[0]
        assert x0.shape[0] == 4
        assert np.allclose(x0[0].real.mean(), 1.0)
        assert np.allclose(x0[1].real.mean(), 2.0)
        assert np.allclose(x0[2].real.mean(), 3.0)
        assert np.allclose(x0[3].real.mean(), 4.0)

        # target=t2 -> [t2, t1, flair, t1ce]
        x1 = ds[1]
        assert np.allclose(x1[0].real.mean(), 2.0)
        assert np.allclose(x1[1].real.mean(), 1.0)
        assert np.allclose(x1[2].real.mean(), 3.0)
        assert np.allclose(x1[3].real.mean(), 4.0)

        # target=flair -> [flair, t1, t2, t1ce]
        x2 = ds[2]
        assert np.allclose(x2[0].real.mean(), 3.0)
        assert np.allclose(x2[1].real.mean(), 1.0)
        assert np.allclose(x2[2].real.mean(), 2.0)
        assert np.allclose(x2[3].real.mean(), 4.0)

