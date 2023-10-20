import pathlib
from pathlib import Path
from typing import Final

import numpy as np
import zarr
from perfcapture.dataset import Dataset

#: The shape of the full array saved into Zarr or NPY files.
_SHAPE: Final[tuple[int, int]] = (50_000, 20_000)


def _load_sample_photo() -> np.ndarray:
    """Load the sample photo as a numpy array.
    
    The sample photo is of shape (height=1,000, width=1,500, channels=3), dtype=uint8.
    """
    module_path = Path(__file__).parent.absolute()
    return np.load(module_path / "sample_photo.npz")["photo"]


def _load_sample_image_and_resize(shape: tuple[int, int] = _SHAPE) -> np.ndarray:
    img = _load_sample_photo()[:, :, 0]
    return np.resize(img, shape)


def _create_zarr_from_image(
    path: Path, 
    shape: tuple[int, int] = _SHAPE, 
    chunks: tuple[int, int] | bool = (5_000,  1_000),
    dtype: np.dtype = np.uint8,
    **kwargs
) -> None:
    z = zarr.open(path, mode='w', shape=shape, chunks=chunks, dtype=dtype, **kwargs)
    z[:] = _load_sample_image_and_resize(shape=shape)


class Uncompressed_1_Chunk(Dataset):
    def create(self) -> None:
        _create_zarr_from_image(path=self.path, chunks=False, compressor=None)


class LZ4_200_Chunks(Dataset):
    def create(self) -> None:
        # The default compressor is LZ4
        _create_zarr_from_image(path=self.path, compressor='default')


class Uncompressed_200_Chunks(Dataset):
    def create(self) -> None:
        _create_zarr_from_image(path=self.path, compressor=None)


class LZ4_20000_Chunks(Dataset):
    def create(self) -> None:
        # The default compressor is LZ4
        _create_zarr_from_image(path=self.path, chunks=(500, 100), compressor='default')


class Uncompressed_20000_Chunks(Dataset):
    def create(self) -> None:
        _create_zarr_from_image(path=self.path, chunks=(500, 100), compressor=None)


class NumpyNPY(Dataset):
    def create(self) -> None:
        arr = _load_sample_image_and_resize()
        np.save(self.path, arr)
    
    def set_path(self, base_data_path: pathlib.Path) -> None:
        self._path = base_data_path / (self.name + '.npy')