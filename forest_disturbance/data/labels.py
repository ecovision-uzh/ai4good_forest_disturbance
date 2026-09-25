"""Turn the annotation table into one training target per (sample, date).

`labels.parquet` describes each sample's history as a list of periods:

    sample_id period_idx label start       end_evidence is_event
    0         0          110   2016-11-10  2022-03-09   False   <- undisturbed forest
    0         1          212   2022-04-08  2022-04-08   True    <- thinning
    0         2          211   2024-04-02  2024-04-02   True    <- clear cut

For an acquisition date we look for the disturbance that is active on that date
or, if none, the most recent one before it. The target is

* `label`: class id of that disturbance (see configs/label_mapping.yaml), or the
  "No Disturbance" id when the sample has no disturbance up to that date;
* `days_since_event`: 0 while the disturbance is active, then days since its
  `end_evidence`; +inf when there is no past disturbance.

Ignored label codes (e.g. drought) produce the target IGNORE_INDEX (-100):
the loss and the metrics skip those dates.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

from forest_disturbance.data.constants import IGNORE_INDEX


@dataclass(frozen=True)
class LabelMapping:
    """How raw DISFOR label codes become classifier ids (configs/label_mapping.yaml)."""

    code_to_class: dict[int, int]  # disturbance label code -> class id
    class_names: dict[int, str]  # class id -> display name
    no_disturbance_id: int
    ignored_codes: frozenset[int]  # kept, but their dates get IGNORE_INDEX
    filtered_codes: frozenset[int]  # removed from the label table entirely

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def disturbance_names(self) -> list[str]:
        """Names of the disturbance classes (every class except No Disturbance), by id."""
        return [n for i, n in sorted(self.class_names.items()) if i != self.no_disturbance_id]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelMapping":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            code_to_class={int(k): int(v) for k, v in raw["disturbance_classes"].items()},
            class_names={int(k): str(v) for k, v in raw["class_names"].items()},
            no_disturbance_id=int(raw["no_disturbance_id"]),
            ignored_codes=frozenset(int(c) for c in raw["ignored_codes"]),
            filtered_codes=frozenset(int(c) for c in raw["filtered_codes"]),
        )


def load_labels(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path).select(
        "sample_id", "period_idx", "label", "start", "end_evidence", "is_event"
    )


def targets_for_dates(
    queries: pl.DataFrame, labels: pl.DataFrame, mapping: LabelMapping
) -> pl.DataFrame:
    """Append `label` and `days_since_event` columns to `queries` (needs sample_id, date).

    Row order of `queries` is preserved.
    """
    queries = queries.with_row_index("query_id")
    disturbances = labels.filter(
        pl.col("label").is_in(set(mapping.code_to_class) | mapping.ignored_codes)
        & ~pl.col("label").is_in(mapping.filtered_codes)
    )
    candidates = queries.select("query_id", "sample_id", "date").join(disturbances, on="sample_id")

    # 1. A disturbance whose evidence interval contains the date ("active").
    active = candidates.filter(
        (pl.col("start") <= pl.col("date")) & (pl.col("date") <= pl.col("end_evidence"))
    )
    overlaps = active.group_by("query_id").len().filter(pl.col("len") > 1)
    if not overlaps.is_empty():
        row = active.join(overlaps, on="query_id").row(0, named=True)
        raise ValueError(
            f"Overlapping disturbances for sample {row['sample_id']} on {row['date']}."
        )
    active = active.with_columns(pl.lit(True).alias("is_active"))

    # 2. Otherwise the disturbance that ended most recently before the date.
    past = (
        candidates.filter(pl.col("end_evidence") <= pl.col("date"))
        .sort(["query_id", "end_evidence", "period_idx"], descending=[False, True, True])
        .unique("query_id", keep="first", maintain_order=True)
        .with_columns(pl.lit(False).alias("is_active"))
    )
    chosen = (
        pl.concat([active, past])
        .sort(["query_id", "is_active"], descending=[False, True])
        .unique("query_id", keep="first", maintain_order=True)
    )

    ignored = pl.col("label").is_in(mapping.ignored_codes)
    targets = chosen.select(
        "query_id",
        pl.when(ignored)
        .then(IGNORE_INDEX)
        .otherwise(pl.col("label").replace_strict(mapping.code_to_class, default=IGNORE_INDEX))
        .cast(pl.Int64)
        .alias("label"),
        pl.when(ignored)
        .then(float(IGNORE_INDEX))
        .when(pl.col("is_active"))
        .then(0.0)
        .otherwise((pl.col("date") - pl.col("end_evidence")).dt.total_days().cast(pl.Float64))
        .alias("days_since_event"),
    )
    return (
        queries.join(targets, on="query_id", how="left")
        .with_columns(
            pl.col("label").fill_null(mapping.no_disturbance_id),
            pl.col("days_since_event").fill_null(float("inf")),
        )
        .sort("query_id")
        .drop("query_id")
    )
