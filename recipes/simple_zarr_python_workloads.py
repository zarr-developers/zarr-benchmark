"""Simple Zarr Python workloads."""

from pathlib import Path

import simple_datasets
import zarr
from perfcapture.dataset import Dataset
from perfcapture.metrics import MetricsForRun
from perfcapture.workload import Workload


class ZarrPythonLoadEntireArray(Workload):
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
        z = zarr.load(dataset_path)
        return MetricsForRun(
            nbytes_in_final_array=z.nbytes,
        )
