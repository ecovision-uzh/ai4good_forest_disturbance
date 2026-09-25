"""Lightning DataModule: builds the train and validation datasets and their loaders."""

from pathlib import Path

import lightning as L
import polars as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from forest_disturbance.data.batch import collate
from forest_disturbance.data.dataset import DisforDataset
from forest_disturbance.data.labels import LabelMapping, load_labels
from forest_disturbance.data.readers import PatchReader, PixelCacheReader


class DisforDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        root: str | Path,
        mapping: LabelMapping,
        fold: int = 0,
        sensors: list[str] = ("s2",),
        target_sensors: list[str] | None = ("s2",),
        reader: str = "pixel_cache",
        image_size: int | None = 1,
        days_before: int = 30,
        years_context: int = 1,
        days_context: int = 15,
        normalization: dict | None = None,
        batch_size: int = 16,
        num_workers: int = 8,
        sampling_alpha: float = 0.5,
    ) -> None:
        """Args:
        root: Dataset folder (labels.parquet, splits.parquet, zarr_frames.parquet, patches/, ...).
        mapping: Label codes -> class ids.
        fold: Cross-validation fold (0-4). Fold 0 is the default for quick experiments.
        sensors: Sensors to load.
        target_sensors: Sensors whose acquisition dates become examples. None = all loaded sensors
            (with s1 this multiplies the number of examples by about 4).
        reader: "pixel_cache" (centre pixel only, fast) or "zarr" (patches, any image_size).
        image_size: Centre crop side in 10 m pixels; None = full 252 px patch (zarr only).
        days_before, years_context, days_context: Time windows, see data/dataset.py.
        normalization: Per sensor {"mean": [...], "std": [...]}.
        batch_size, num_workers: DataLoader settings.
        sampling_alpha: Oversample rare classes during training; 0 = no oversampling.
        """
        super().__init__()
        self.root = Path(root)
        self.mapping = mapping
        self.fold = fold
        self.sensors = list(sensors)
        self.target_sensors = None if target_sensors is None else list(target_sensors)
        self.reader_kind = reader
        self.image_size = image_size
        self.windows = {
            "days_before": days_before,
            "years_context": years_context,
            "days_context": days_context,
        }
        self.normalization = normalization
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_alpha = sampling_alpha
        self.labels: pl.DataFrame | None = None
        self.frames: pl.DataFrame | None = None
        self.train_set: DisforDataset | None = None
        self.val_set: DisforDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return
        self.labels = load_labels(self.root / "labels.parquet")
        self.frames = pl.read_parquet(self.root / "zarr_frames.parquet")
        reader = self._make_reader()
        splits = pl.read_parquet(self.root / "splits.parquet").filter(
            pl.col("fold_id") == self.fold
        )
        datasets = {}
        for split in ("train", "val"):
            sample_ids = splits.filter(pl.col("split") == split).select("sample_id")
            datasets[split] = DisforDataset(
                self.frames.join(sample_ids, on="sample_id"),
                self.labels,
                self.mapping,
                reader,
                sensors=self.sensors,
                target_sensors=self.target_sensors,
                normalization=self.normalization,
                **self.windows,
            )
        self.train_set, self.val_set = datasets["train"], datasets["val"]
        print(
            f"fold {self.fold}: {len(self.train_set)} training and {len(self.val_set)} validation examples"
        )

    def _make_reader(self) -> PatchReader | PixelCacheReader:
        if self.reader_kind == "zarr":
            return PatchReader(self.root / "patches", image_size=self.image_size)
        if self.reader_kind == "pixel_cache":
            assert self.image_size == 1, (
                "The pixel cache only holds the centre pixel: set data.image_size=1."
            )
            return PixelCacheReader(self.root / "center_pixels.parquet", sensors=self.sensors)
        raise ValueError(f"Unknown reader {self.reader_kind!r}: use 'pixel_cache' or 'zarr'.")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.sampling_alpha > 0:
            weights = self.train_set.sampling_weights(self.sampling_alpha)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return self._loader(self.train_set, shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def _loader(self, dataset: DisforDataset, **kwargs) -> DataLoader:
        workers = self.num_workers
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
            # "spawn" avoids dead-locks between zarr's background threads and forked workers.
            multiprocessing_context="spawn" if workers > 0 else None,
            **kwargs,
        )
