import pathlib

import numpy as np
from perfcapture.dataset import Dataset
from perfcapture.metrics import MetricsForRun
from perfcapture.workload import Workload
from simple_datasets import NumpyNPY


class NumpyLoadEntireArray(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (NumpyNPY(), )
    
    def run(self, dataset_path: pathlib.Path) -> MetricsForRun:
        a = np.load(dataset_path)
        return MetricsForRun(
            nbytes_in_final_array=a.nbytes,
        )