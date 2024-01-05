"""Read an Aperio SVS file via the Tifffile Zarr interface
as well as via FSStore and the fsspec.ReferenceFileSystem

pip install fsspec[http] tifffile imagecodecs imagecodecs_numcodecs
"""
from pathlib import Path
from typing import Final

import fsspec
import zarr
from perfcapture.dataset import Dataset
from perfcapture.workload import Workload
from tifffile import TiffFile
from tifffile.tifffile import tiff2fsspec
from zarr.storage import FSStore


_OUTPUT_TILE_SIZE_X: Final[int] = 1024
_OUTPUT_TILE_SIZE_Y: Final[int] = 1024
SVS_TIFF = "https://data.cytomine.coop/open/openslide/aperio-svs/CMU-1.svs"
SERIES_IDX = 0


class TifffileSVSDataset(Dataset):
    def create(self) -> None:
        fs, path = fsspec.core.url_to_fs(SVS_TIFF)
        fn = str(self.path.joinpath("CMU-1.svs").absolute())
        kc = str(self.path.joinpath("CMU-1.json").absolute())
        fs.get(path, fn)
        tiff2fsspec(
            fn,
            url=fn,
            out=kc,
            series=SERIES_IDX,
        )


class ReadViaTifffileWorkload(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (TifffileSVSDataset(),)
    
    def run(self, dataset_path: Path) -> None:
        """Read Aperio SVS file via tifffile.ZarrTiffStore"""
        svs = dataset_path.joinpath("CMU-1.svs")
        store = TiffFile(svs).aszarr(maxworkers=1)
        array = zarr.open_group(store, mode="r")[str(SERIES_IDX)]

        for idx_x in range(0, array.shape[1], _OUTPUT_TILE_SIZE_X):
            for idx_y in range(0, array.shape[0], _OUTPUT_TILE_SIZE_Y):
                _ = array[
                    idx_y: idx_y + _OUTPUT_TILE_SIZE_Y,
                    idx_x: idx_x + _OUTPUT_TILE_SIZE_X,
                    :
                ]


class ReadViaReferenceFileSystemWorkload(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (TifffileSVSDataset(),)

    def run(self, dataset_path: Path) -> None:
        """Read Aperio SVS via fsspec.ReferenceFileSystem"""
        fs = fsspec.filesystem(
            "reference",
            fo=str(dataset_path.joinpath("CMU-1.json")),
        )
        store = FSStore("", fs=fs)
        array = zarr.open_group(store, mode="r")[str(SERIES_IDX)]

        for idx_x in range(0, array.shape[1], _OUTPUT_TILE_SIZE_X):
            for idx_y in range(0, array.shape[0], _OUTPUT_TILE_SIZE_Y):
                _ = array[
                    idx_y: idx_y + _OUTPUT_TILE_SIZE_Y,
                    idx_x: idx_x + _OUTPUT_TILE_SIZE_X,
                    :
                ]
