"""Extract the annotated centre pixel of every acquisition into one small table.

The baseline model only looks at the centre pixel. Reading it from the 1.7 TB
zarr patches is slow, so we read it once and save it to
`<data_root>/center_pixels.parquet` (56 MB). The values are produced
by the same reader the training code uses, so both give identical tensors.

Run once per dataset copy (about 5 minutes with 44 workers on 48 CPU cores):

    python scripts/build_pixel_cache.py --data-root $DISFOR_DATA_ROOT --workers 32
"""

import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import zarr

from forest_disturbance.data.constants import BANDS
from forest_disturbance.data.readers import read_sensor


def extract_sample(args: tuple[Path, int, dict[str, pl.DataFrame]]) -> pl.DataFrame:
    patches_dir, sample_id, frames_by_sensor = args
    store = zarr.open_group(patches_dir / f"{sample_id}.zarr", mode="r")
    parts = []
    for sensor, frames in frames_by_sensor.items():
        zarr_ids = frames["zarr_id"].to_list()
        values = read_sensor(store, sensor, zarr_ids, image_size=1)[:, :, 0, 0].numpy()
        columns = {band: values[:, i] for i, band in enumerate(BANDS[sensor])}
        parts.append(
            frames.select("sample_id", "sensor", "date", "zarr_id").with_columns(
                pl.Series(name, column, dtype=pl.Float32) for name, column in columns.items()
            )
        )
    return pl.concat(parts, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None, help="Default: <data-root>/center_pixels.parquet"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sample-ids", type=int, nargs="*", default=None, help="Only these samples (for testing)."
    )
    args = parser.parse_args()
    output = args.output or args.data_root / "center_pixels.parquet"

    frames = pl.read_parquet(args.data_root / "zarr_frames.parquet")
    if args.sample_ids:
        frames = frames.filter(pl.col("sample_id").is_in(args.sample_ids))
    jobs = [
        (
            args.data_root / "patches",
            int(sample_id[0]),
            dict(
                (
                    (str(sensor[0]), rows.sort("zarr_id"))
                    for sensor, rows in group.partition_by("sensor", as_dict=True).items()
                )
            ),
        )
        for sample_id, group in frames.partition_by("sample_id", as_dict=True).items()
    ]
    print(f"{len(jobs)} samples, {frames.height} acquisitions -> {output}")

    start, results = time.time(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, table in enumerate(pool.imap_unordered(extract_sample, jobs, chunksize=1), 1):
            results.append(table)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} samples, {time.time() - start:.0f} s", flush=True)

    columns = ["sample_id", "sensor", "date", "zarr_id", *BANDS["s1"], *BANDS["s2"]]
    table = (
        pl.concat(results, how="diagonal").select(columns).sort("sample_id", "sensor", "zarr_id")
    )
    assert table.height == frames.height, "Every indexed acquisition must be extracted."

    # Sanity check: no all-zero sensor, sensible value ranges.
    for sensor in ("s1", "s2"):
        values = table.filter(pl.col("sensor") == sensor).select(BANDS[sensor]).to_numpy()
        if values.size:
            print(
                f"{sensor}: shape={values.shape} min={np.nanmin(values)} max={np.nanmax(values)} mean={np.nanmean(values):.1f}"
            )
            assert np.any(values != 0), f"All {sensor} values are zero."
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output)
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
