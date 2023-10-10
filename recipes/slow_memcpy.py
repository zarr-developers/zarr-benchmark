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

_ROWS_TOTAL: Final[int] = 100_000
_COLS_TOTAL: Final[int] = 70_000
_ROWS_PER_CHUNK: Final[int] = 5_000
_COLS_PER_CHUNK: Final[int] = 1_000

# Each batch reads all columns, but just one chunk of rows.
_NUM_BATCHES: Final[int] = _ROWS_TOTAL // _ROWS_PER_CHUNK

class SlowMemcpyDatasetAllZeroLZ4(Dataset):
    def create(self) -> None:
        array = _all_zero_array()
        # Note that, by default, zarr.save uses LZ4 compression.
        zarr.save(
            self.path,
            array, 
            mode='w',
            chunks=(_ROWS_PER_CHUNK, _COLS_PER_CHUNK))
        
class SlowMemcpyDatasetAllZeroUncompressed(Dataset):
    def create(self) -> None:
        array = _all_zero_array()
        array = zarr.array(array, chunks=(_ROWS_PER_CHUNK, _COLS_PER_CHUNK), compressor=None)
        zarr.save(self.path, array, mode='w')
        

def _all_zero_array() -> np.ndarray:
    return np.zeros((_ROWS_TOTAL, _COLS_TOTAL), dtype=np.int16)

class SlowMemcpyWorkload(Workload):
    def init_datasets(self) -> tuple[Dataset]:
        return (SlowMemcpyDatasetAllZeroLZ4(), SlowMemcpyDatasetAllZeroUncompressed())
    
    def run(self, dataset_path: Path) -> None:
        """Load Zarr into RAM, in batches.
        
        Each batch reads all columns, but just one chunk of rows.
        """
        array = zarr.open(dataset_path, mode='r').arr_0
        for i in range(_NUM_BATCHES):
            row_slice = slice(i * _ROWS_PER_CHUNK, (i+1) * _ROWS_PER_CHUNK)
            array.get_orthogonal_selection((row_slice, slice(0, _COLS_TOTAL)))