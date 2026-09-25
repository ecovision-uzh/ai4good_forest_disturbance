"""Choose which acquisitions become training/validation examples ("target frames").

`zarr_frames.parquet` lists every usable acquisition: one row per
(sample_id, sensor, date) with `zarr_id`, its position on the zarr time axis.
Cloudy Sentinel-2 acquisitions are already removed from this table.

An acquisition on date D is a target frame when:

1. D lies inside the annotated period of its sample (first label start ..
   last label end_evidence);
2. the "recent" window [D - days_before, D] holds another acquisition
   (an earlier one, or a second sensor on the same date);
3. for each of the `years_context` previous years, the window
   [D - k years - days_context, D - k years + days_context] holds an acquisition.
"""

import polars as pl


def select_target_frames(
    frames: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    days_before: int,
    years_context: int,
    days_context: int,
) -> pl.DataFrame:
    """Return the rows of `frames` that satisfy the three rules above."""
    # Rule 1: inside the annotated period.
    periods = labels.group_by("sample_id").agg(
        pl.col("start").min().alias("first_day"), pl.col("end_evidence").max().alias("last_day")
    )
    candidates = (
        frames.join(periods, on="sample_id")
        .filter(pl.col("date").is_between(pl.col("first_day"), pl.col("last_day")))
        .drop("first_day", "last_day")
    )
    if candidates.is_empty():
        return candidates

    dates = candidates.select("sample_id", "date").unique().sort("sample_id", "date")
    timeline = frames.select("sample_id", pl.col("date").alias("frame_date")).sort(
        "sample_id", "frame_date"
    )

    # Rule 2: another acquisition in the recent window.
    if days_before > 0:
        earlier = _has_frame_between(
            dates,
            timeline,
            start=pl.col("date").dt.offset_by(f"-{days_before}d"),
            end=pl.col("date").dt.offset_by("-1d"),
        )
        two_sensors_same_day = (
            timeline.group_by("sample_id", "frame_date")
            .len()
            .filter(pl.col("len") >= 2)
            .select("sample_id", pl.col("frame_date").alias("date"))
        )
        dates = dates.join(
            pl.concat([earlier, two_sensors_same_day]), on=["sample_id", "date"], how="semi"
        )

    # Rule 3: an acquisition around the same date in each previous year.
    for years in range(1, years_context + 1):
        center = pl.col("date").dt.offset_by(f"-{years}y")
        dates = _has_frame_between(
            dates,
            timeline,
            start=center.dt.offset_by(f"-{days_context}d"),
            end=center.dt.offset_by(f"{days_context}d"),
        )

    return candidates.join(dates, on=["sample_id", "date"], how="semi")


def _has_frame_between(
    dates: pl.DataFrame, timeline: pl.DataFrame, *, start: pl.Expr, end: pl.Expr
) -> pl.DataFrame:
    """Keep the (sample_id, date) rows whose window [start, end] contains a timeline frame."""
    windows = dates.with_columns(start.alias("window_start"), end.alias("window_end")).sort(
        "sample_id", "window_end"
    )
    # For each window, find the latest frame on or before its end, then check it is after its start.
    return (
        windows.join_asof(
            timeline,
            left_on="window_end",
            right_on="frame_date",
            by="sample_id",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(pl.col("frame_date") >= pl.col("window_start"))
        .select("sample_id", "date")
    )
