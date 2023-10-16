"""Implement a minimal Dataset and Workload which shows the slow unaligned memcpy
behavior first highlighted in the Zarr Benchmarking & Performance meeting
on Thu 28th Sept 2023.

See https://github.com/zarr-developers/zarr-benchmark/issues/5
"""

from pathlib import Path

import simple_datasets
import zarr
from perfcapture.dataset import Dataset
from perfcapture.workload import Workload


class LoadEntireArray(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (
            simple_datasets.LZ4_100_chunks(),
            simple_datasets.Uncompressed_100_chunks(),
            simple_datasets.LZ4_10000_chunks(),
            simple_datasets.Uncompressed_10000_chunks(),
            )

    def run(self, dataset_path: Path) -> None:
        """Load entire Zarr into RAM."""
        zarr.load(dataset_path)
