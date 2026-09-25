"""Non-operational (NO) metrics: how good are the raw, unfiltered per-date predictions?

Every prediction row (one sample, one date) becomes an "alert" when the summed
probability of all disturbance classes is >= `threshold`; its class is the most
likely disturbance class. No temporal filtering is applied. This is a model
design tool, not a deployment score (see operational.py for that).

For an event (annotated disturbance) and a horizon H days, its window is

    start .................. end_evidence ......... end_evidence + H
    |<------------------------- event window ------------------->|
    (cut earlier if the next event of the same sample starts before)

Rows are set aside ("ignored": neither right nor wrong) when they fall
* between the last clear Sentinel-2 image before the event and the event start
  (the disturbance may already have happened, we cannot tell);
* inside an ignored label (e.g. drought);
* after the window of an event, up to one year after its end (the forest still
  looks disturbed; alerts there are not counted as false).

Metrics per horizon H (keys end in `_{H}d`):
    binary_recall          fraction of events with at least one alert in their window
    alert_precision        alerts in an event window / (those + alerts outside any window)
    PR_AUC                 area under precision-recall when sweeping the threshold
                           (recall over events, precision over rows)
    window_agent_accuracy  disturbance type from the mean class probabilities over the
    window_agent_f1_macro  window, compared with the true type (ignores detection)
Plus `detection_rate_1y` and `average_detection_delay_days_1y` (first alert in a 365 d window).
A diagnostic: not documented in docs/, which only cover the operational metrics.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from forest_disturbance.data.labels import LabelMapping
from forest_disturbance.metrics.events import disturbance_events, probability_column

HORIZONS_DAYS = (14, 30, 60, 365)
ONE_YEAR = 365


def evaluate_non_operational(
    predictions: pl.DataFrame,
    *,
    labels: pl.DataFrame,
    s2_dates: pl.DataFrame,
    mapping: LabelMapping,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    threshold: float = 0.5,
    sample_ids: list[int] | None = None,
) -> dict[str, float | int | None]:
    """Args:
    predictions: One row per (sample_id, date) with a `proba_<class>` column per class.
    labels: labels.parquet.
    s2_dates: Usable Sentinel-2 dates per sample (events.s2_dates(zarr_frames)).
    mapping: Label codes -> class ids.
    horizons: Event window extensions in days.
    threshold: Minimum total disturbance probability for an alert.
    sample_ids: Samples to evaluate. Default: the samples present in `predictions`.
        Samples without predictions then count as missed events.
    """
    predictions = predictions.with_columns(
        pl.col("sample_id").cast(pl.Int64), pl.col("date").cast(pl.Date)
    )
    if sample_ids is None:
        sample_ids = sorted(predictions["sample_id"].unique().to_list())
    labels = labels.filter(pl.col("sample_id").is_in(sample_ids))
    events = disturbance_events(labels, mapping, sample_ids)
    events = events.with_columns(pl.col("start").shift(-1).over("sample_id").alias("next_start"))

    # --- Turn probabilities into alerts --------------------------------------------------
    class_ids = sorted(mapping.class_names)
    scores = (
        predictions.select([probability_column(mapping.class_names[i]) for i in class_ids])
        .to_numpy()
        .astype(np.float64)
    )
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(), (
        "Probabilities must lie in [0, 1]."
    )
    disturbance_ids = np.array([i for i in class_ids if i != mapping.no_disturbance_id])
    disturbance_scores = scores[:, disturbance_ids]
    disturbance_probability = disturbance_scores.sum(axis=1)
    alert_class = np.where(
        disturbance_probability >= threshold,
        disturbance_ids[disturbance_scores.argmax(axis=1)],
        mapping.no_disturbance_id,
    )
    is_alert = alert_class != mapping.no_disturbance_id

    rows = _RowIndex(predictions)
    before_event = _before_event_mask(rows, events, s2_dates)
    ignored_label = _ignored_label_mask(
        rows,
        labels.filter(
            pl.col("label").is_in(mapping.ignored_codes)
            & ~pl.col("label").is_in(mapping.filtered_codes)
        ),
    )
    before_event &= ~ignored_label
    ignored = before_event | ignored_label

    event_list = list(events.iter_rows(named=True))
    result: dict[str, float | int | None] = {
        "event_count": len(event_list),
        "alert_count": int(is_alert.sum()),
    }

    # --- Detection delay within one year ---------------------------------------------------
    delays = []
    for event in event_list:
        window = rows.window(event, _window_end(event, ONE_YEAR), exclude=ignored)
        alert_dates = rows.dates[window[is_alert[window]]]
        if alert_dates.size:
            delays.append(
                max(0, int((alert_dates.min() - np.datetime64(event["start"], "D")).astype(int)))
            )
    result["detection_rate_1y"] = len(delays) / len(event_list) if event_list else None
    result["average_detection_delay_days_1y"] = float(np.mean(delays)) if delays else None

    for horizon in horizons:
        positive, after_window = _horizon_masks(rows, event_list, horizon, ignored)
        suffix = f"_{horizon}d"

        # Event recall: an event is detected if its window holds at least one alert.
        detected, agent_true, agent_predicted = 0, [], []
        for event in event_list:
            window = rows.window(event, _window_end(event, horizon), exclude=ignored)
            if window.size and is_alert[window].any():
                detected += 1
            if window.size:
                mean_scores = disturbance_scores[window].mean(axis=0)
                agent_true.append(int(event["class_id"]))
                agent_predicted.append(int(disturbance_ids[np.argmax(mean_scores)]))
        result["binary_recall" + suffix] = detected / len(event_list) if event_list else 0.0

        # Alert precision: alerts in an event window vs alerts that are neither matched nor ignored.
        matched = int((is_alert & positive).sum())
        set_aside = int((is_alert & (ignored | after_window)).sum())
        false_alerts = int(is_alert.sum()) - matched - set_aside
        result["alert_precision" + suffix] = (
            matched / (matched + false_alerts) if matched + false_alerts else None
        )
        result["false_alert_count" + suffix] = false_alerts

        result["PR_AUC" + suffix] = _pr_auc(
            rows, event_list, horizon, disturbance_probability, positive, ~(ignored | after_window)
        )
        accuracy, f1 = _agent_scores(agent_true, agent_predicted, disturbance_ids)
        result["window_agent_accuracy" + suffix] = accuracy
        result["window_agent_f1_macro" + suffix] = f1
    return result


class _RowIndex:
    """Fast lookup of prediction rows per sample and date."""

    def __init__(self, predictions: pl.DataFrame) -> None:
        self.sample_ids = predictions["sample_id"].to_numpy()
        self.dates = predictions["date"].to_numpy().astype("datetime64[D]")
        self.count = predictions.height
        order = np.argsort(self.sample_ids, kind="stable")
        unique, first = np.unique(self.sample_ids[order], return_index=True)
        bounds = np.append(first, len(order))
        self.by_sample = {int(s): order[bounds[i] : bounds[i + 1]] for i, s in enumerate(unique)}

    def of_sample(self, sample_id: int) -> np.ndarray:
        return self.by_sample.get(int(sample_id), np.zeros(0, dtype=np.int64))

    def window(self, event: dict, end: date, exclude: np.ndarray) -> np.ndarray:
        """Row indices of the event's sample with start <= date <= end, not excluded."""
        idx = self.of_sample(event["sample_id"])
        keep = (self.dates[idx] >= np.datetime64(event["start"], "D")) & (
            self.dates[idx] <= np.datetime64(end, "D")
        )
        idx = idx[keep]
        return idx[~exclude[idx]]


def _window_end(event: dict, horizon: int | None) -> date:
    """Last day of the event window: end_evidence + horizon, but before the next event starts."""
    end = date.max if horizon is None else event["end_evidence"] + timedelta(days=horizon)
    if event["next_start"] is not None:
        end = min(end, event["next_start"] - timedelta(days=1))
    return end


def _before_event_mask(rows: _RowIndex, events: pl.DataFrame, s2_dates: pl.DataFrame) -> np.ndarray:
    """Rows strictly between the last usable S2 image before an event and the event start."""
    mask = np.zeros(rows.count, dtype=bool)
    starts = (
        events.select("sample_id", pl.col("start").alias("event_start"))
        .unique()
        .sort("sample_id", "event_start")
    )
    previous = starts.join_asof(
        s2_dates.rename({"date": "previous_s2"}).filter(
            pl.col("sample_id").is_in(starts["sample_id"].unique().to_list())
        ),
        left_on="event_start",
        right_on="previous_s2",
        by="sample_id",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    missing = previous.filter(pl.col("previous_s2").is_null())
    if not missing.is_empty():
        raise ValueError(
            f"{missing.height} events have no earlier usable S2 image, e.g. {missing.head(3).to_dicts()}."
        )
    for sample_id, event_start, previous_s2 in previous.iter_rows():
        idx = rows.of_sample(sample_id)
        dates = rows.dates[idx]
        mask[
            idx[
                (dates > np.datetime64(previous_s2, "D"))
                & (dates < np.datetime64(event_start, "D"))
            ]
        ] = True
    return mask


def _ignored_label_mask(rows: _RowIndex, ignored_labels: pl.DataFrame) -> np.ndarray:
    """Rows inside an ignored label's [start, end_evidence] interval."""
    mask = np.zeros(rows.count, dtype=bool)
    for sample_id, start, end in ignored_labels.select(
        "sample_id", "start", "end_evidence"
    ).iter_rows():
        idx = rows.of_sample(sample_id)
        dates = rows.dates[idx]
        mask[idx[(dates >= np.datetime64(start, "D")) & (dates <= np.datetime64(end, "D"))]] = True
    return mask


def _horizon_masks(
    rows: _RowIndex, events: list[dict], horizon: int, ignored: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`positive`: rows inside an event window. `after_window`: rows after it, up to one year."""
    positive = np.zeros(rows.count, dtype=bool)
    after_window = np.zeros(rows.count, dtype=bool)
    for event in events:
        idx = rows.of_sample(event["sample_id"])
        idx = idx[rows.dates[idx] <= np.datetime64(_window_end(event, ONE_YEAR), "D")]
        days_after_end = (rows.dates[idx] - np.datetime64(event["end_evidence"], "D")).astype(
            np.int64
        )
        started = rows.dates[idx] >= np.datetime64(event["start"], "D")
        positive[idx[started & (days_after_end <= horizon)]] = True
        after_window[idx[days_after_end > horizon]] = True
    positive &= ~ignored
    after_window &= ~(ignored | positive)
    return positive, after_window


def _pr_auc(
    rows: _RowIndex,
    events: list[dict],
    horizon: int,
    scores: np.ndarray,
    positive: np.ndarray,
    evaluable: np.ndarray,
) -> float | None:
    """Average precision where recall counts events and precision counts rows."""
    targets, row_scores = positive[evaluable], scores[evaluable]
    if not targets.any() or targets.all():
        return None
    # Row precision at every distinct threshold, from high to low.
    order = np.argsort(-row_scores, kind="stable")
    sorted_targets, sorted_scores = targets[order].astype(np.int64), row_scores[order]
    last_of_value = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_pos = np.cumsum(sorted_targets)[last_of_value]
    false_pos = np.cumsum(1 - sorted_targets)[last_of_value]
    thresholds = sorted_scores[last_of_value]
    precision = true_pos / (true_pos + false_pos)
    # Event recall: an event is found at threshold t if its best evaluable row scores >= t.
    best = np.full(len(events), -np.inf)
    for k, event in enumerate(events):
        window = rows.window(event, _window_end(event, horizon), exclude=~evaluable)
        if window.size:
            best[k] = scores[window].max()
    best = np.sort(best[np.isfinite(best)])
    recall = (best.size - np.searchsorted(best, thresholds, side="left")) / len(events)
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def _agent_scores(
    true: list[int], predicted: list[int], class_ids: np.ndarray
) -> tuple[float | None, float | None]:
    """Accuracy and macro F1 (over classes present in `true`) of the disturbance type."""
    if not true:
        return None, None
    position = {int(c): i for i, c in enumerate(class_ids)}
    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for t, p in zip(true, predicted, strict=True):
        confusion[position[t], position[p]] += 1
    hits, support, predicted_count = (
        np.diag(confusion),
        confusion.sum(axis=1),
        confusion.sum(axis=0),
    )
    precision = np.divide(
        hits, predicted_count, out=np.zeros(len(class_ids)), where=predicted_count > 0
    )
    recall = np.divide(hits, support, out=np.zeros(len(class_ids)), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(class_ids)),
        where=precision + recall > 0,
    )
    return float(np.mean(np.array(true) == np.array(predicted))), float(np.mean(f1[support > 0]))
