from pathlib import Path

import numpy as np
import zarr
from perfcapture.dataset import Dataset


def _load_sample_photo() -> np.ndarray:
    """Load the sample photo as a numpy array.
    
    The sample photo is of shape (height=1,000, width=1,500, channels=3), dtype=uint8.
    """
    module_path = Path(__file__).parent.absolute()
    return np.load(module_path / "sample_photo.npz")["photo"]


def _create_dataset_from_image(
    path: Path, 
    shape: tuple[int, int] = (50_000, 10_000), 
    chunks: tuple[int, int] | bool = (5_000,  1_000),
    **kwargs
) -> None:
    z = zarr.open(path, mode='w', shape=shape, chunks=chunks, **kwargs)
    img = _load_sample_photo()[:, :, 0]
    z[:] = np.resize(img, shape)


class Uncompressed_1_Chunk(Dataset):
    def create(self) -> None:
        _create_dataset_from_image(path=self.path, chunks=False, compressor=None)


class LZ4_100_Chunks(Dataset):
    def create(self) -> None:
        # The default compressor is LZ4
        _create_dataset_from_image(path=self.path, compressor='default')


class Uncompressed_100_Chunks(Dataset):
    def create(self) -> None:
        _create_dataset_from_image(path=self.path, compressor=None)


class LZ4_10000_Chunks(Dataset):
    def create(self) -> None:
        # The default compressor is LZ4
        _create_dataset_from_image(path=self.path, chunks=(500, 100), compressor='default')


class Uncompressed_10000_Chunks(Dataset):
    def create(self) -> None:
        _create_dataset_from_image(path=self.path, chunks=(500, 100), compressor=None)
