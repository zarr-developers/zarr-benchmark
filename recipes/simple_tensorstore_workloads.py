"""
`pip install tensorstore`
"""

from pathlib import Path

import numpy as np
import simple_datasets
import tensorstore as ts
from perfcapture.dataset import Dataset
from perfcapture.metrics import MetricsForRun
from perfcapture.workload import Workload


class TensorStoreLoadEntireArray(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (
            simple_datasets.Uncompressed_1_Chunk(),
            simple_datasets.LZ4_200_Chunks(),
            simple_datasets.Uncompressed_200_Chunks(),
            simple_datasets.LZ4_20000_Chunks(),
            simple_datasets.Uncompressed_20000_Chunks(),
            )

    def run(self, dataset_path: Path) -> MetricsForRun:
        """Load entire Zarr into RAM."""
        z = _tensorstore_load_zarr(dataset_path)
        return MetricsForRun(
            nbytes_in_final_array=z.nbytes,
        )

def _tensorstore_load_zarr(dataset_path: Path) -> np.ndarray:
    spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': str(dataset_path)},
    }
    ds = ts.open(spec)
    ds = ds.result()
    ds = ds.read()
    return ds.result()