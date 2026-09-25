"""Shared helpers for the event-level metrics."""

import re
from pathlib import Path

import polars as pl

from forest_disturbance.data.labels import LabelMapping


def disturbance_events(
    labels: pl.DataFrame, mapping: LabelMapping, sample_ids: list[int]
) -> pl.DataFrame:
    """One row per annotated disturbance that the model is expected to detect.

    Ignored and filtered label codes are not events. Sorted by sample and start date.
    """
    events = labels.filter(
        pl.col("sample_id").is_in(sample_ids)
        & pl.col("label").is_in(list(mapping.code_to_class))
        & ~pl.col("label").is_in(mapping.filtered_codes)
        & ~pl.col("label").is_in(mapping.ignored_codes)
    )
    return (
        events.with_columns(
            pl.col("sample_id").cast(pl.Int64),
            pl.col("label").cast(pl.Int64),
            pl.col("label").replace_strict(mapping.code_to_class).cast(pl.Int64).alias("class_id"),
        )
        .with_columns(pl.col("class_id").replace_strict(mapping.class_names).alias("class_name"))
        .select(
            "sample_id", "period_idx", "label", "class_id", "class_name", "start", "end_evidence"
        )
        .sort("sample_id", "start", "end_evidence", "period_idx")
    )


def probability_column(class_name: str) -> str:
    """'Clear Cut' -> 'proba_clear_cut' (column name in prediction files)."""
    return "proba_" + re.sub(r"[^0-9a-zA-Z]+", "_", class_name.strip().lower()).strip("_")


def s2_dates(frames: pl.DataFrame) -> pl.DataFrame:
    """Unique (sample_id, date) of the usable Sentinel-2 acquisitions."""
    return (
        frames.filter(pl.col("sensor") == "s2")
        .select(pl.col("sample_id").cast(pl.Int64), "date")
        .unique()
        .sort("sample_id", "date")
    )


def load_split_sample_ids(data_root: str | Path, split: str, fold: int) -> list[int]:
    splits = pl.read_parquet(Path(data_root) / "splits.parquet")
    rows = splits.filter((pl.col("fold_id") == fold) & (pl.col("split") == split))
    return sorted(int(s) for s in rows["sample_id"].unique())
