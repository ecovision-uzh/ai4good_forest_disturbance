"""Read pixel values for a list of acquisitions of one sample.

Two readers return the same tensors:

* `PatchReader` reads the zarr patches (any crop size, slow-ish).
* `PixelCacheReader` reads `center_pixels.parquet`, a small table holding only
  the annotated centre pixel of every acquisition (crop size 1, fast).

Both return float32 tensors shaped (time, channel, height, width), with
channels in the order of `constants.S1_BANDS` / `constants.S2_BANDS`.
"""

from math import ceil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import zarr
from torch.nn import functional as F

from forest_disturbance.data.constants import BANDS, S2_ARRAYS


class PatchReader:
    """Read acquisitions from `patches/{sample_id}.zarr` stores."""

    def __init__(self, patches_dir: str | Path, image_size: int | None = None) -> None:
        """Args:
        patches_dir: Folder containing one `{sample_id}.zarr` store per sample.
        image_size: Side of the square centre crop, in 10 m pixels. None = full 252 px.
        """
        self.patches_dir = Path(patches_dir)
        self.image_size = image_size

    def read(self, sample_id: int, sensor: str, zarr_ids: list[int]) -> torch.Tensor:
        store = zarr.open_group(self.patches_dir / f"{sample_id}.zarr", mode="r")
        return read_sensor(store, sensor, zarr_ids, self.image_size)

    def read_scene_classification(self, sample_id: int, zarr_ids: list[int]) -> torch.Tensor:
        """Sentinel-2 scene classification (cloud) mask: (time, height, width), 20 m pixels.

        Class codes are ESA's: 4 = vegetation, 5 = bare soil, 6 = water, 8/9/10 = cloud,
        11 = snow. Used by the visualisation to ignore clouds when scaling colours.
        """
        store = zarr.open_group(self.patches_dir / f"{sample_id}.zarr", mode="r")
        array = store["s2_scl"]
        size = None if self.image_size is None else _native_size(self.image_size, 20)
        height, width = array.shape[-2:]
        rows, cols = slice(None), slice(None)
        if size is not None:
            top, left = height // 2 - size // 2, width // 2 - size // 2
            rows, cols = slice(top, top + size), slice(left, left + size)
        values = array[np.asarray(zarr_ids, dtype=np.int64), rows, cols]
        return torch.as_tensor(np.asarray(values), dtype=torch.uint8)


def read_sensor(
    store: zarr.Group, sensor: str, zarr_ids: list[int], image_size: int | None
) -> torch.Tensor:
    """Read one sensor from an open zarr store. `zarr_ids` index the time axis."""
    if sensor == "s1":
        return _read_array(store["s1"], zarr_ids, image_size)

    # S2 bands come at 10, 20 and 60 m. Crop each at its own resolution, then
    # upsample the 20 m and 60 m bands onto the 10 m grid.
    arrays = {
        name: _read_array(store[name], zarr_ids, _native_size(image_size, metres))
        for name, metres in S2_ARRAYS.items()
    }
    size = arrays["s2_10m"].shape[-2:]
    b10, b20, b60 = (
        x
        if x.shape[-2:] == size
        else F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        for x in arrays.values()
    )
    # Reassemble in wavelength order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
    return torch.cat(
        (b60[:, :1], b10[:, :3], b20[:, :3], b10[:, 3:], b20[:, 3:4], b60[:, 1:], b20[:, 4:]),
        dim=1,
    )


def _native_size(image_size: int | None, metres: int) -> int | None:
    return None if image_size is None else ceil(image_size / (metres // 10))


def _read_array(array: zarr.Array, zarr_ids: list[int], image_size: int | None) -> torch.Tensor:
    height, width = array.shape[-2:]
    if image_size is None:
        rows, cols = slice(None), slice(None)
    else:
        if image_size > min(height, width):
            raise ValueError(
                f"image_size={image_size} is larger than the stored array ({height} x {width})."
            )
        top, left = height // 2 - image_size // 2, width // 2 - image_size // 2
        rows, cols = slice(top, top + image_size), slice(left, left + image_size)
    if not zarr_ids:
        n_rows = len(range(*rows.indices(height)))
        n_cols = len(range(*cols.indices(width)))
        return torch.empty((0, array.shape[1], n_rows, n_cols), dtype=torch.float32)
    values = array[np.asarray(zarr_ids, dtype=np.int64), slice(None), rows, cols]
    return torch.as_tensor(values, dtype=torch.float32)


class PixelCacheReader:
    """Read centre pixels from `center_pixels.parquet` (built by scripts/build_pixel_cache.py)."""

    def __init__(self, cache_path: str | Path, sensors: list[str]) -> None:
        table = pl.read_parquet(cache_path).filter(pl.col("sensor").is_in(sensors))
        self.values: dict[str, np.ndarray] = {}
        self.row_of: dict[str, dict[tuple[int, int], int]] = {}
        for sensor in sensors:
            rows = table.filter(pl.col("sensor") == sensor)
            self.values[sensor] = rows.select(BANDS[sensor]).to_numpy().astype(np.float32)
            keys = zip(rows["sample_id"].to_list(), rows["zarr_id"].to_list(), strict=True)
            self.row_of[sensor] = {key: i for i, key in enumerate(keys)}

    def read(self, sample_id: int, sensor: str, zarr_ids: list[int]) -> torch.Tensor:
        rows = [self.row_of[sensor][(sample_id, zarr_id)] for zarr_id in zarr_ids]
        values = self.values[sensor][rows]  # (time, channel)
        return torch.from_numpy(values)[:, :, None, None]
