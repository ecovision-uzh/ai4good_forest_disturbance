"""Operational (O) metrics: score the alerts a deployed monitoring system would send.

A real system cannot raise an alarm every time one image looks odd (clouds,
shadows, noise). So per-date probabilities first go through a filter:

1. CUSUM accumulator, one per sample and disturbance class. For each date:
       accumulator = max(0, accumulator + (probability - alpha))
   An alert is sent when accumulator >= threshold. Several days of moderately high
   probability, or one very high one, are needed.
2. Rest period ("cooldown"): after an alert, that class stays silent for
   `rest_period_days`. The accumulator restarts from 0.

Each alert is then matched to at most one annotated event:

    B window        |<---- event window: start - B .. end_evidence + t_after_days ---->|
    (<= 14 days, from the previous clear S2 image)

* An alert inside a not-yet-detected event window of its sample = true positive.
  Slow "continuous" events (biotic) may absorb several alerts.
* Any other alert = false positive. An event without alert = false negative (a miss).
* Delay (lag) = alert date - event start; can be negative down to -14 days (B window).

Binary metrics ignore the class. Class metrics repeat the matching for each class
separately (a Wind alert only matches a Wind event). Evaluation of a sample starts
one year after its first observation (the model needs history) and stops before
its first ignored label (e.g. drought). See docs/04_metrics.md.
"""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from forest_disturbance.data.constants import CONTINUOUS_LABELS
from forest_disturbance.data.labels import LabelMapping
from forest_disturbance.metrics.events import disturbance_events, probability_column


@dataclass(frozen=True)
class Event:
    event_id: str  # "sample_id:period_idx"
    sample_id: int
    class_name: str
    start: date
    end_evidence: date
    window_start: date
    is_continuous: bool


@dataclass(frozen=True)
class Alert:
    sample_id: int
    date: date
    class_name: str | None  # None for a class-less alert


def evaluate_operational(
    predictions: pl.DataFrame,
    *,
    labels: pl.DataFrame,
    timeline: pl.DataFrame,
    mapping: LabelMapping,
    alpha: dict[str, float],
    threshold: dict[str, float],
    rest_period_days: int = 365,
    binary_alpha: float = 0.2,
    binary_threshold: float = 6.0,
    t_before_days: int = 14,
    t_after_days: int = 365,
    start_offset_years: int = 1,
    sample_ids: list[int] | None = None,
) -> dict:
    """Args:
    predictions: One row per (sample_id, date[, sensor]) with `proba_<class>` columns.
    labels: labels.parquet.
    timeline: zarr_frames.parquet (all sensors): used for the start of monitoring and
        for the previous clear S2 image before each event.
    mapping: Label codes -> class ids.
    alpha, threshold: CUSUM parameters per disturbance class name.
    rest_period_days: Cooldown after an alert.
    binary_alpha, binary_threshold: CUSUM parameters of the class-less stream
        built from 1 - p(No Disturbance) ("negative_binary").
    t_before_days: Maximum length of the B window before an event.
    t_after_days: How long after an event's end an alert can still match it (the horizon).
    start_offset_years: Years of history before monitoring starts.
    sample_ids: Samples to evaluate; default: those in `predictions`.

    Returns:
        {"binary": {...}, "per_class": {...}, "macro": {...}, "lag": {...},
         "negative_binary": {"binary": {...}, "lag": {...}}, "counts": {...}, "alerts": [...]}
    """
    predictions = predictions.with_columns(
        pl.col("sample_id").cast(pl.Int64), pl.col("date").cast(pl.Date)
    )
    timeline = timeline.select(
        pl.col("sample_id").cast(pl.Int64), pl.col("date").cast(pl.Date), "sensor"
    )
    if sample_ids is None:
        sample_ids = sorted(predictions["sample_id"].unique().to_list())
    labels = labels.filter(pl.col("sample_id").is_in(sample_ids))

    # 1. Monitoring period per sample: [first observation + N years, first ignored label).
    first_seen = dict(
        timeline.filter(pl.col("sample_id").is_in(sample_ids))
        .group_by("sample_id")
        .agg(pl.col("date").min())
        .iter_rows()
    )
    missing = set(sample_ids) - set(first_seen)
    assert not missing, f"No observations for samples {sorted(missing)[:5]}."
    monitoring_start = {s: _add_years(d, start_offset_years) for s, d in first_seen.items()}
    monitoring_end = dict(
        labels.filter(
            pl.col("label").is_in(mapping.ignored_codes)
            & ~pl.col("label").is_in(mapping.filtered_codes)
        )
        .group_by("sample_id")
        .agg(pl.col("start").min())
        .iter_rows()
    )

    # 2. Events to detect, with their B window start.
    all_events = _make_events(labels, timeline, mapping, sample_ids, t_before_days)
    events = [e for e in all_events if e.start < monitoring_end.get(e.sample_id, date.max)]
    early = [e for e in events if e.start < monitoring_start[e.sample_id]]
    assert not early, (
        f"Event before the monitoring start (increase history or check data): {early[0]}."
    )

    # 3. Keep predictions inside the monitoring period.
    period = pl.DataFrame(
        {
            "sample_id": list(monitoring_start),
            "monitor_from": list(monitoring_start.values()),
            "monitor_until": [monitoring_end.get(s, date.max) for s in monitoring_start],
        },
        schema={"sample_id": pl.Int64, "monitor_from": pl.Date, "monitor_until": pl.Date},
    )
    predictions = (
        predictions.join(period, on="sample_id")
        .filter(
            (pl.col("date") >= pl.col("monitor_from")) & (pl.col("date") < pl.col("monitor_until"))
        )
        .sort("sample_id", "date", maintain_order=True)
    )

    # 4. Class alerts from the CUSUM filter, then matching and metrics.
    classes = mapping.disturbance_names
    probabilities = (
        predictions.select([probability_column(c) for c in classes]).to_numpy().astype(np.float64)
    )
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    alerts = cusum_alerts(predictions, probabilities, classes, alpha, threshold, rest_period_days)
    matches = match_alerts(alerts, events, t_after_days)
    result = {
        "binary": _binary_counts(alerts, events, matches),
        "per_class": {},
        "lag": _lag(alerts, events, matches),
        "counts": {
            "events": len(events),
            "censored_events": len(all_events) - len(events),
            "alerts": len(alerts),
        },
    }
    for class_name in classes:
        class_alerts = [a for a in alerts if a.class_name == class_name]
        class_events = [e for e in events if e.class_name == class_name]
        class_matches = match_alerts(class_alerts, class_events, t_after_days)
        tp = len(set(class_matches.values()))
        fp = sum(i not in class_matches for i in range(len(class_alerts)))
        result["per_class"][class_name] = _precision_recall_f1(tp, fp, len(class_events) - tp)
    result["macro"] = {
        m: float(np.mean([c[m] for c in result["per_class"].values()]))
        for m in ("precision", "recall", "f1")
    }

    # 5. Class-less stream: probability of "any disturbance" = 1 - p(No Disturbance).
    no_disturbance = mapping.class_names[mapping.no_disturbance_id]
    any_disturbance = (
        1.0 - predictions[probability_column(no_disturbance)].to_numpy().astype(np.float64)[:, None]
    )
    binary_alerts = cusum_alerts(
        predictions,
        any_disturbance,
        [None],
        {None: binary_alpha},
        {None: binary_threshold},
        rest_period_days,
    )
    binary_matches = match_alerts(binary_alerts, events, t_after_days)
    result["negative_binary"] = {
        "binary": _binary_counts(binary_alerts, events, binary_matches),
        "lag": _lag(binary_alerts, events, binary_matches),
    }

    id_to_event = {e.event_id: e for e in events}
    result["alerts"] = [
        {
            "sample_id": a.sample_id,
            "date": a.date.isoformat(),
            "predicted_class": a.class_name,
            "matched_event_id": matches.get(i),
            "true_class": id_to_event[matches[i]].class_name if i in matches else None,
            "lag_days": (a.date - id_to_event[matches[i]].start).days if i in matches else None,
        }
        for i, a in enumerate(alerts)
    ]
    return result


def cusum_alerts(
    predictions: pl.DataFrame,
    probabilities: np.ndarray,
    classes: list[str | None],
    alpha: dict,
    threshold: dict,
    rest_period_days: int,
) -> list[Alert]:
    """Run the CUSUM filter. `probabilities` has one column per entry of `classes`.

    Rows are sorted by sample and date. Rows sharing a date (S1 and S2) are added together.
    """
    sample_ids = predictions["sample_id"].to_numpy()
    dates = predictions["date"].to_list()
    alerts: list[Alert] = []
    # Boundaries of consecutive (sample, date) groups.
    new_group = np.ones(len(dates), dtype=bool)
    new_group[1:] = (sample_ids[1:] != sample_ids[:-1]) | np.array(
        [dates[i] != dates[i - 1] for i in range(1, len(dates))], dtype=bool
    )
    starts = np.flatnonzero(new_group)
    ends = np.append(starts[1:], len(dates))

    for k, class_name in enumerate(classes):
        increments = probabilities[:, k] - alpha[class_name]
        accumulator, rest_until, current_sample = 0.0, date.min, None
        for start, end in zip(starts, ends, strict=True):
            sample_id, day = int(sample_ids[start]), dates[start]
            if sample_id != current_sample:
                accumulator, rest_until, current_sample = 0.0, date.min, sample_id
            if day <= rest_until:
                continue
            accumulator = max(0.0, accumulator + float(np.sum(increments[start:end])))
            if accumulator >= threshold[class_name]:
                alerts.append(Alert(sample_id, day, class_name))
                accumulator = 0.0
                rest_until = day + timedelta(days=rest_period_days)
    return sorted(alerts, key=lambda a: (a.sample_id, a.date, classes.index(a.class_name)))


def match_alerts(alerts: list[Alert], events: list[Event], t_after_days: int) -> dict[int, str]:
    """Match alerts to events, in date order. Returns {alert index: event_id}.

    Alerts of the same sample and date form one group and share one match.
    """
    events_of: dict[int, list[Event]] = {}
    for event in events:
        events_of.setdefault(event.sample_id, []).append(event)
    groups: dict[tuple[int, date], list[int]] = {}
    for i, alert in enumerate(alerts):
        groups.setdefault((alert.sample_id, alert.date), []).append(i)

    detected: set[str] = set()
    matches: dict[int, str] = {}
    for (sample_id, day), indices in sorted(groups.items()):
        in_window = [
            e
            for e in events_of.get(sample_id, [])
            if e.window_start <= day <= e.end_evidence + timedelta(days=t_after_days)
        ]
        event = next((e for e in in_window if e.event_id not in detected), None)
        if event is None:  # continuous events can match again
            event = next((e for e in in_window if e.is_continuous), None)
        if event is None:
            continue
        detected.add(event.event_id)
        matches.update({i: event.event_id for i in indices})
    return matches


def _make_events(
    labels: pl.DataFrame,
    timeline: pl.DataFrame,
    mapping: LabelMapping,
    sample_ids: list[int],
    t_before_days: int,
) -> list[Event]:
    s2_dates: dict[int, list[date]] = {}
    for sample_id, day in (
        timeline.filter(pl.col("sensor") == "s2")
        .select("sample_id", "date")
        .unique()
        .sort("sample_id", "date")
        .iter_rows()
    ):
        s2_dates.setdefault(sample_id, []).append(day)
    events = []
    for row in disturbance_events(labels, mapping, sample_ids).iter_rows(named=True):
        # B window: at most t_before_days, and never before the previous clear S2 image.
        window_start = row["start"] - timedelta(days=t_before_days)
        dates = s2_dates.get(row["sample_id"], [])
        previous = bisect_left(dates, row["start"]) - 1
        if previous >= 0:
            window_start = max(window_start, dates[previous])
        events.append(
            Event(
                event_id=f"{row['sample_id']}:{row['period_idx']}",
                sample_id=row["sample_id"],
                class_name=row["class_name"],
                start=row["start"],
                end_evidence=row["end_evidence"],
                window_start=window_start,
                is_continuous=row["label"] in CONTINUOUS_LABELS,
            )
        )
    return events


def _binary_counts(alerts: list[Alert], events: list[Event], matches: dict[int, str]) -> dict:
    tp = len(set(matches.values()))
    alert_days = {}
    for i, alert in enumerate(alerts):
        alert_days.setdefault(
            (alert.sample_id, alert.date), i
        )  # first alert of each (sample, date)
    fp = sum(i not in matches for i in alert_days.values())
    return _precision_recall_f1(tp, fp, len(events) - tp)


def _lag(alerts: list[Alert], events: list[Event], matches: dict[int, str]) -> dict:
    """Detection delay of each detected event: its earliest matched alert."""
    start = {e.event_id: e.start for e in events}
    lag: dict[str, int] = {}
    for i, event_id in matches.items():
        days = (alerts[i].date - start[event_id]).days
        lag[event_id] = min(lag.get(event_id, days), days)
    values = list(lag.values())
    return {
        "mean_days": float(np.mean(values)) if values else None,
        "median_days": float(np.median(values)) if values else None,
    }


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _add_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(month=2, day=28, year=day.year + years)


def flatten(result: dict, prefix: str = "operational", suffix: str = "") -> dict[str, float]:
    """Scalar metrics of `evaluate_operational` as flat {name: value}, e.g. for wandb.

    `suffix` names the horizon, e.g. "_60d" -> "operational/binary_f1_60d".
    """
    flat = {}
    for name in ("precision", "recall", "f1"):
        flat[f"{prefix}/binary_{name}{suffix}"] = result["binary"][name]
        flat[f"{prefix}/macro_{name}{suffix}"] = result["macro"][name]
        flat[f"{prefix}/any_disturbance_{name}{suffix}"] = result["negative_binary"]["binary"][name]
    for class_name, metrics in result["per_class"].items():
        flat[f"{prefix}/f1_{class_name.lower().replace(' ', '_')}{suffix}"] = metrics["f1"]
    flat[f"{prefix}/lag_median_days{suffix}"] = result["lag"]["median_days"]
    flat[f"{prefix}/any_disturbance_lag_median_days{suffix}"] = result["negative_binary"]["lag"][
        "median_days"
    ]
    return {k: v for k, v in flat.items() if v is not None}
