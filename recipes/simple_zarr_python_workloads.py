"""Simple Zarr Python workloads."""

from pathlib import Path

import simple_datasets
import zarr
from perfcapture.dataset import Dataset
from perfcapture.workload import Workload


class ZarrPythonLoadEntireArray(Workload):
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
