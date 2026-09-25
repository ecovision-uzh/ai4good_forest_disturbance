"""Everything about one sample, ready to plot.

`Sample.load(889, data_root)` collects the tables a figure needs (labels, the
value of the annotated pixel on every date, the split, the metadata) and reads
image patches on demand through the tested data loader
(`forest_disturbance.data.readers.PatchReader`) — the visualisation code never
opens a zarr store itself.

    sample = Sample.load(889, root)
    dates, ndvi = sample.series("NDVI")          # 1-D series at the annotated pixel
    dates, cube = sample.patches("s2", n=5)      # image patches around the first event
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from forest_disturbance.data.constants import LABEL_NAMES, S1_BANDS, S2_BANDS
from forest_disturbance.data.readers import PatchReader
from forest_disturbance.viz import raster

#: Scene-classification codes where the ground is visible (vegetation, soil, water, unclassified).
LAND_CLASSES = (4, 5, 6, 7)

#: Series you can ask for by name.
SERIES_NAMES = (*S2_BANDS, *S1_BANDS, *raster.INDICES, "PC1", "PC2", "PC3")


@dataclass
class Sample:
    """One sample: its metadata, its annotations, its pixel values and its patches."""

    sample_id: int
    root: Path
    info: dict  # row of samples.parquet
    labels: pl.DataFrame  # rows of labels.parquet, sorted by period
    pixels: dict[str, pl.DataFrame]  # sensor -> centre-pixel values, sorted by date
    fold: dict[int, str] | None = None  # fold id -> "train" / "val"

    @classmethod
    def load(cls, sample_id: int, root: str | Path) -> "Sample":
        root = Path(root)
        info = pl.read_parquet(root / "samples.parquet").filter(pl.col("sample_id") == sample_id)
        if info.is_empty():
            raise ValueError(f"sample {sample_id} is not in {root / 'samples.parquet'}")
        pixels = pl.read_parquet(root / "center_pixels.parquet").filter(
            pl.col("sample_id") == sample_id
        )
        splits = pl.read_parquet(root / "splits.parquet").filter(pl.col("sample_id") == sample_id)
        return cls(
            sample_id=int(sample_id),
            root=root,
            info=info.row(0, named=True),
            labels=pl.read_parquet(root / "labels.parquet")
            .filter(pl.col("sample_id") == sample_id)
            .sort("period_idx"),
            pixels={
                str(sensor[0]): rows.sort("date")
                for sensor, rows in pixels.partition_by("sensor", as_dict=True).items()
            },
            fold={int(r["fold_id"]): str(r["split"]) for r in splits.iter_rows(named=True)} or None,
        )

    # --- 1-D series at the annotated pixel ---------------------------------------------

    def dates(self, sensor: str = "s2") -> list[date]:
        """Acquisition dates of one sensor, in order."""
        table = self.pixels.get(sensor)
        return [] if table is None else table["date"].to_list()

    def series(self, name: str, sensor: str | None = None) -> tuple[list[date], np.ndarray]:
        """Values of one band or index at the annotated pixel, over time.

        Args:
            name: A Sentinel-2 band ("B08"), a Sentinel-1 band ("VV"), an index
                ("NDVI", "NDMI", "NDWI", "NBR", "NDRE") or a principal component of
                all the bands of one sensor ("PC1", "PC2", ...). Bands come back in
                physical units: reflectance in [0, 1] for Sentinel-2, dB for Sentinel-1.
            sensor: Usually inferred from the name.
        """
        sensor = sensor or ("s1" if name in S1_BANDS else "s2")
        table = self.pixels.get(sensor)
        if table is None or table.is_empty():
            return [], np.zeros(0, dtype=np.float32)
        dates = table["date"].to_list()
        if name in raster.INDICES:
            values = raster.spectral_index(
                raster.reflectance(table.select(S2_BANDS).to_numpy()), name
            )
        elif name in S2_BANDS:
            values = raster.reflectance(table[name].to_numpy())
        elif name in S1_BANDS:
            values = raster.decibels(table[name].to_numpy())
        elif name.startswith("PC") and name[2:].isdigit():
            values = self._component(sensor, int(name[2:]) - 1)
        else:
            raise ValueError(f"unknown series {name!r}; known: {', '.join(SERIES_NAMES)}")
        return dates, np.asarray(values, dtype=np.float32)

    def _component(self, sensor: str, index: int) -> np.ndarray:
        """One principal component of every band of `sensor` at the annotated pixel.

        The bands are standardised first, so a component is a mix of shapes rather
        than of magnitudes, and its sign is fixed (largest loading positive) so the
        same component does not flip between two samples.
        """
        values = self.table(sensor)
        centred = (values - values.mean(0)) / np.maximum(values.std(0), 1e-6)
        _, _, components = np.linalg.svd(centred - centred.mean(0), full_matrices=False)
        if index >= len(components):
            raise ValueError(f"{sensor} has only {len(components)} components")
        axis = components[index]
        return centred @ (axis * np.sign(axis[np.argmax(np.abs(axis))]))

    def table(self, sensor: str = "s2") -> np.ndarray:
        """All bands of one sensor at the annotated pixel: (time, channel), physical units."""
        bands = S2_BANDS if sensor == "s2" else S1_BANDS
        values = self.pixels[sensor].select(bands).to_numpy()
        return raster.reflectance(values) if sensor == "s2" else raster.decibels(values)

    # --- image patches -------------------------------------------------------------------

    def patches(
        self,
        sensor: str = "s2",
        *,
        dates: list[date] | None = None,
        n: int = 5,
        crop: int | None = 64,
    ) -> tuple[list[date], np.ndarray]:
        """Read image patches: returns the dates used and a (time, channel, H, W) array.

        Args:
            sensor: "s1" or "s2".
            dates: Exact dates to read. By default `pick_dates(n)` chooses them.
            n: How many dates when `dates` is None.
            crop: Side of the centre crop in 10 m pixels; None keeps the full 252 px patch.
        """
        table = self.pixels.get(sensor)
        if table is None or table.is_empty():
            return [], np.zeros((0, 0, 0, 0), dtype=np.float32)
        wanted = dates if dates is not None else self.pick_dates(n, sensor=sensor)
        available = table["date"].to_list()
        chosen = [
            min(available, key=lambda d, target=target: abs((d - target).days)) for target in wanted
        ]
        zarr_ids = [int(table.filter(pl.col("date") == d)["zarr_id"][0]) for d in chosen]
        reader = PatchReader(self.root / "patches", image_size=crop)
        cube = reader.read(self.sample_id, sensor, zarr_ids).numpy()
        return chosen, cube

    def clear_mask(self, dates: list[date], *, crop: int | None = 64) -> np.ndarray:
        """True where the ground is visible (not cloud, shadow or snow), one map per date.

        Upsampled from the 20 m scene-classification mask to the size of the patches.
        """
        table = self.pixels["s2"]
        zarr_ids = [int(table.filter(pl.col("date") == d)["zarr_id"][0]) for d in dates]
        reader = PatchReader(self.root / "patches", image_size=crop)
        mask = reader.read_scene_classification(self.sample_id, zarr_ids).numpy()
        clear = np.isin(mask, LAND_CLASSES)
        if crop is not None and clear.shape[-1] != crop:  # 20 m mask -> 10 m grid
            factor = int(round(crop / clear.shape[-1]))
            clear = np.repeat(np.repeat(clear, factor, axis=-2), factor, axis=-1)[..., :crop, :crop]
        return clear

    def pick_dates(
        self,
        n: int = 5,
        *,
        sensor: str = "s2",
        clear_window_days: int = 25,
        same_season: bool = True,
        lead_days: int = 30,
    ) -> list[date]:
        """Choose `n` telling dates: the pair around the first disturbance, then whole years.

        The two most useful pictures are the one from about a month before the
        disturbance and the first one after it, so they always come first. The
        picture before is deliberately not the last acquisition before the annotated
        date: annotation dates can be weeks late (see `dev/CAMPAIGN_DATES_BRIEF.md`),
        and that last picture then already shows the disturbance, which is confusing.
        Use `lead_days=0` to go back to the last picture before. The remaining dates are taken
        one, two, ... years away *in the same season*, because a forest in February
        and the same forest in August look different for reasons that are not a
        disturbance. Within a few weeks of each target the least cloudy acquisition
        is used. Set `same_season=False` to spread the dates evenly instead.
        """
        available = self.dates(sensor)
        if not available:
            return []
        events = self.events()
        if events.is_empty() or not same_season:
            spread = np.linspace(0, len(available) - 1, n).round().astype(int)
            return sorted({available[int(i)] for i in spread})

        anchor = events["start"][0]
        before = [d for d in available if d < anchor]
        after = [d for d in available if d >= anchor]
        # A narrow window around the target, so the picture really is about a month
        # before the event rather than the clearest one of a whole season.
        target = anchor - timedelta(days=lead_days)
        first = self._clearest_near(before, target, min(clear_window_days, 14), sensor, exclude=[])
        chosen = [first] if first is not None else []
        # The first picture after the event is often the cloudy one that hid it; take
        # the clearest of the first few weeks instead, which still shows the change.
        follow = self._clearest_near(after, anchor, 20, sensor, exclude=chosen)
        chosen += [follow] if follow is not None else []
        years = 1
        while len(chosen) < n:
            targets = []
            if before:
                targets.append(anchor - timedelta(days=365 * years))
            if after:
                targets.append(anchor + timedelta(days=365 * years))
            if not targets or years > 6:
                break
            for target in targets:
                if len(chosen) >= n:
                    break
                pick = self._clearest_near(
                    available, target, clear_window_days, sensor, exclude=chosen
                )
                if pick is not None:
                    chosen.append(pick)
            years += 1
        return sorted(dict.fromkeys(chosen))[:n]

    def _clearest_near(
        self, available: list[date], target: date, window: int, sensor: str, *, exclude: list[date]
    ) -> date | None:
        """Least cloudy acquisition within `window` days of `target`.

        "Least cloudy" is the lowest blue reflectance, which cloud and haze raise.
        Acquisitions whose blue is implausibly low are skipped: those are usually a
        partly empty frame at the edge of a Sentinel-2 tile, which looks broken.
        """
        candidates = [d for d in available if d not in exclude]
        if not candidates:
            return None
        near = [d for d in candidates if abs((d - target).days) <= window]
        if not near:
            return min(candidates, key=lambda d: abs((d - target).days))
        if sensor != "s2":
            return near[len(near) // 2]
        blue = self._blue()
        floor = np.percentile([blue[d] for d in self.dates(sensor)], 2)
        usable = [d for d in near if blue[d] > floor] or near
        return min(usable, key=lambda d: blue[d])

    def _blue(self) -> dict[date, float]:
        """Blue reflectance of every Sentinel-2 acquisition, by date.

        Keyed by date rather than by position: the callers work with subsets of the
        timeline (only the dates before an event, say), and a positional index into
        the full table would then point at the wrong acquisition.
        """
        table = self.pixels.get("s2")
        if table is None or table.is_empty():
            return {}
        return dict(
            zip(table["date"].to_list(), raster.reflectance(table["B02"].to_numpy()), strict=True)
        )

    # --- annotations -----------------------------------------------------------------------

    def events(self) -> pl.DataFrame:
        """Label rows that are disturbances (codes 2xx), sorted by date."""
        return self.labels.filter(pl.col("label") >= 200).sort("start")

    def label_name(self, code: int) -> str:
        """The raw DISFOR name of a label code, e.g. 231 -> "Bark Beetle (with decline)".

        This is the fine-grained name. The class the benchmark actually asks for is
        coarser: use `configs/label_mapping.yaml` for that.
        """
        return LABEL_NAMES.get(int(code), str(code))

    @property
    def window(self) -> tuple[date, date]:
        """First and last day of the annotated period."""
        return self.info["window_start"], self.info["window_end"]

    def caption(self) -> str:
        """One line describing the sample, for a figure title."""
        folds = ", ".join(
            f"fold {f}" for f, split in sorted((self.fold or {}).items()) if split == "val"
        )
        location = (
            f"{abs(self.info['lat']):.2f}°{'N' if self.info['lat'] >= 0 else 'S'} "
            f"{abs(self.info['lon']):.2f}°{'E' if self.info['lon'] >= 0 else 'W'}"
        )
        parts = [f"sample {self.sample_id}", location, f"campaign {self.info['dataset']}"]
        if folds:
            parts.append(f"validation of {folds}")
        return "  ·  ".join(parts)
