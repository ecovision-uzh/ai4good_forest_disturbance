"""PyTorch dataset: one example = one target acquisition of one sample.

For a target date D, each sensor returns two kinds of time series:

    recent  : acquisitions in [D - days_before, D]          (includes D itself)
    yearly  : acquisitions in [D - k years +- days_context]  for k = 1..years_context

    time ──────────────────────────────────────────────────────────────▶
          [ yearly k=1 ]                                   [ recent ]
          D-1y-15d .. D-1y+15d                           D-30d .. D

The model compares "now" (recent) with "the same season last year" (yearly):
a forest that was green last June and is bare this June was probably cut.

One example is a dict:

    {
      "sample_id": int, "target_date": datetime.date, "target_sensor": "s2",
      "inputs": {"s2": {"recent": TimeSeries, "yearly": [TimeSeries, ...]}},
      "target": {"label": int, "days_since_event": float},
    }

`collate` pads the variable-length series of a batch (see batch.py).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from forest_disturbance.data.frames import select_target_frames
from forest_disturbance.data.labels import LabelMapping, targets_for_dates


@dataclass(frozen=True)
class TimeSeries:
    dates: list[date]
    values: torch.Tensor  # (time, channel, height, width)


class DisforDataset(Dataset):
    def __init__(
        self,
        frames: pl.DataFrame,
        labels: pl.DataFrame,
        mapping: LabelMapping,
        reader,
        *,
        sensors: list[str],
        target_sensors: list[str] | None = None,
        days_before: int,
        years_context: int,
        days_context: int,
        normalization: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Args:
        frames: Rows of zarr_frames.parquet for the samples of this split.
        labels: labels.parquet.
        mapping: Label codes -> class ids.
        reader: PatchReader or PixelCacheReader; provides `read(sample_id, sensor, zarr_ids)`.
        sensors: Sensors to load, e.g. ["s2"] or ["s1", "s2"].
        target_sensors: Sensors whose acquisition dates become examples (default: all loaded sensors).
        days_before, years_context, days_context: Window sizes (see module docstring).
        normalization: Per sensor {"mean": [...], "std": [...]}: values become (x - mean) / std.
        """
        self.reader = reader
        self.sensors = sensors
        self.days_before = days_before
        self.years_context = years_context
        self.days_context = days_context
        self.normalization = {
            sensor: (
                torch.tensor(stats["mean"])[:, None, None],
                torch.tensor(stats["std"])[:, None, None],
            )
            for sensor, stats in (normalization or {}).items()
        }

        frames = frames.filter(pl.col("sensor").is_in(sensors)).sort(
            "sample_id", "sensor", "date", "zarr_id"
        )
        # Per sample and sensor: acquisition dates (as day numbers) and zarr ids, sorted by date.
        self.timelines: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {
            (int(sample_id), str(sensor)): (
                np.array([d.toordinal() for d in rows["date"].to_list()], dtype=np.int64),
                rows["zarr_id"].to_numpy(),
            )
            for (sample_id, sensor), rows in frames.partition_by(
                "sample_id", "sensor", as_dict=True
            ).items()
        }

        targets = (
            select_target_frames(
                frames,
                labels,
                days_before=days_before,
                years_context=years_context,
                days_context=days_context,
            )
            .filter(pl.col("sensor").is_in(target_sensors or sensors))
            .sort("sample_id", "date", "sensor", "zarr_id")
        )
        # One row per example: sample_id, sensor, date, zarr_id, label, days_since_event.
        self.examples = targets_for_dates(targets, labels, mapping)

    def __len__(self) -> int:
        return self.examples.height

    def __getitem__(self, index: int) -> dict:
        row = self.examples.row(index, named=True)
        return {
            "sample_id": row["sample_id"],
            "target_date": row["date"],
            "target_sensor": row["sensor"],
            "inputs": {
                sensor: self.load_windows(row["sample_id"], sensor, row["date"])
                for sensor in self.sensors
            },
            "target": {
                "label": int(row["label"]),
                "days_since_event": float(row["days_since_event"]),
            },
        }

    def load_windows(self, sample_id: int, sensor: str, target_date: date) -> dict:
        """Load the recent and yearly time series of one sensor around `target_date`."""
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        day_numbers, zarr_ids = self.timelines.get((sample_id, sensor), empty)

        windows = [(target_date - timedelta(days=self.days_before), target_date)]
        for years in range(1, self.years_context + 1):
            center = _minus_years(target_date, years)
            windows.append(
                (
                    center - timedelta(days=self.days_context),
                    center + timedelta(days=self.days_context),
                )
            )
        # Positions (in the sorted timeline) of the acquisitions inside each window.
        positions = [
            np.arange(
                np.searchsorted(day_numbers, start.toordinal(), side="left"),
                np.searchsorted(day_numbers, end.toordinal(), side="right"),
            )
            for start, end in windows
        ]

        # Read every needed acquisition once, then split the result into windows.
        needed = sorted({int(zarr_ids[p]) for window in positions for p in window})
        values = self.reader.read(sample_id, sensor, needed)
        if sensor in self.normalization:
            mean, std = self.normalization[sensor]
            values.sub_(mean).div_(std)
        index_of = {zarr_id: i for i, zarr_id in enumerate(needed)}

        series = [
            TimeSeries(
                dates=[date.fromordinal(int(day_numbers[p])) for p in window],
                values=values[[index_of[int(zarr_ids[p])] for p in window]],
            )
            for window in positions
        ]
        return {"recent": series[0], "yearly": series[1:]}

    def sampling_weights(self, alpha: float) -> torch.Tensor:
        """Per-example weight = (number of examples of its class) ** -alpha.

        alpha=0: uniform sampling. alpha=1: every class is drawn equally often.
        """
        labels = self.examples["label"].to_numpy()
        classes, counts = np.unique(labels, return_counts=True)
        count_of = dict(zip(classes.tolist(), counts.tolist(), strict=True))
        return torch.tensor(
            [float(count_of[label]) ** -alpha for label in labels.tolist()], dtype=torch.float64
        )


def _minus_years(day: date, years: int) -> date:
    if day.month == 2 and day.day == 29:
        return day.replace(year=day.year - years, day=28)
    return day.replace(year=day.year - years)
