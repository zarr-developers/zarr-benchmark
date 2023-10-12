"""Implement a minimal Dataset and Workload which shows the slow unaligned memcpy
behavior first highlighted in the Zarr Benchmarking & Performance meeting
on Thu 28th Sept 2023.

See https://github.com/zarr-developers/zarr-benchmark/issues/5
"""

from pathlib import Path
from typing import Final

import numpy as np
import zarr
from perfcapture.dataset import Dataset
from perfcapture.workload import Workload

_ROWS_PER_CHUNK: Final[int] = 5_000
_COLS_PER_CHUNK: Final[int] = 1_000


class SlowMemcpyDatasetAllZeroLZ4(Dataset):
    def create(self) -> None:
        # The default compressor is LZ4
        shape = (100_000, 70_000)
        z = _zarr_open(self.path, shape=shape, compressor='default')
        z[:] = _all_zero_array(shape)

class SlowMemcpyDatasetAllZeroUncompressed(Dataset):
    def create(self) -> None:
        shape = (10_000, 10_000)
        z = _zarr_open(self.path, shape=shape, compressor=None)
        z[:] = _all_zero_array(shape)
        

def _all_zero_array(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int16)

def _zarr_open(path, shape, compressor):
    return zarr.open(
        path,
        mode='w',
        shape=shape,
        chunks=(_ROWS_PER_CHUNK, _COLS_PER_CHUNK),
        compressor=compressor,
        dtype=np.int16,
    )

class SlowMemcpyWorkload(Workload):
    def init_datasets(self) -> tuple[Dataset]:
        return (SlowMemcpyDatasetAllZeroLZ4(), SlowMemcpyDatasetAllZeroUncompressed())
    
    def run(self, dataset_path: Path) -> None:
        """Load Zarr into RAM, in batches.
        
        Each batch reads all columns, but just one chunk of rows.
        """
        array = zarr.open(dataset_path, mode='r')
        # Each batch reads all columns, but just one chunk of rows.    
        num_batches = array.shape[0] // _ROWS_PER_CHUNK
        cols_total = array.shape[1]
        for i in range(num_batches):
            row_slice = slice(i * _ROWS_PER_CHUNK, (i+1) * _ROWS_PER_CHUNK)
            array.get_orthogonal_selection((row_slice, slice(0, cols_total)))