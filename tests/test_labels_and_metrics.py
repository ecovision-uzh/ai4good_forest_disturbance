"""Small hand-made cases for the label logic and the event metrics."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from forest_disturbance.data.constants import IGNORE_INDEX
from forest_disturbance.data.labels import targets_for_dates
from forest_disturbance.metrics.events import probability_column
from forest_disturbance.metrics.non_operational import evaluate_non_operational
from forest_disturbance.metrics.operational import evaluate_operational

LABELS = pl.DataFrame(
    {
        "sample_id": [1, 1, 1, 2, 2],
        "period_idx": [0, 1, 2, 0, 1],
        "label": [110, 211, 120, 110, 241],  # forest, clear cut, regrowth | forest, drought
        "start": [
            date(2018, 1, 1),
            date(2020, 5, 1),
            date(2020, 5, 11),
            date(2018, 1, 1),
            date(2021, 7, 1),
        ],
        "end_evidence": [
            date(2020, 4, 20),
            date(2020, 5, 10),
            date(2022, 12, 31),
            date(2021, 6, 30),
            date(2021, 9, 1),
        ],
        "is_event": [False, True, False, False, True],
    }
)


def test_targets(mapping):
    queries = pl.DataFrame(
        {
            "sample_id": [1, 1, 1, 2, 2],
            "date": [
                date(2019, 1, 1),
                date(2020, 5, 5),
                date(2020, 6, 9),
                date(2020, 1, 1),
                date(2021, 8, 1),
            ],
        }
    )
    targets = targets_for_dates(queries, LABELS, mapping)
    assert targets["label"].to_list() == [
        6,
        0,
        0,
        6,
        IGNORE_INDEX,
    ]  # none, clear cut (active), clear cut (past), none, ignored
    assert targets["days_since_event"].to_list() == [
        float("inf"),
        0.0,
        30.0,
        float("inf"),
        float(IGNORE_INDEX),
    ]


def _predictions(mapping, days: list[date], clear_cut_probability: list[float]) -> pl.DataFrame:
    table = {"sample_id": [1] * len(days), "date": days, "sensor": ["s2"] * len(days)}
    for class_id, name in mapping.class_names.items():
        values = np.array(clear_cut_probability) if class_id == 0 else np.zeros(len(days))
        if class_id == mapping.no_disturbance_id:
            values = 1.0 - np.array(clear_cut_probability)
        table[probability_column(name)] = values
    return pl.DataFrame(table)


def test_event_metrics_detect_a_clear_cut(mapping):
    days = [
        date(2019, 1, 1),
        date(2019, 6, 1),
        date(2020, 4, 20),
        date(2020, 5, 3),
        date(2020, 5, 8),
        date(2020, 5, 13),
        date(2020, 6, 1),
    ]
    probability = [0.0, 0.0, 0.0, 0.9, 0.95, 0.99, 0.2]
    predictions = _predictions(mapping, days, probability)
    timeline = pl.DataFrame(
        {"sample_id": [1] * len(days), "date": days, "sensor": ["s2"] * len(days)}
    )
    s2 = timeline.select("sample_id", "date")

    no = evaluate_non_operational(
        predictions, labels=LABELS, s2_dates=s2, mapping=mapping, sample_ids=[1]
    )
    assert no["event_count"] == 1
    assert no["binary_recall_14d"] == 1.0
    assert no["alert_precision_14d"] == 1.0
    assert (
        no["average_detection_delay_days_1y"] == 2.0
    )  # first alert 2020-05-03, event start 2020-05-01

    alpha = {name: 0.5 for name in mapping.disturbance_names}
    threshold = {name: 0.8 for name in mapping.disturbance_names}
    o = evaluate_operational(
        predictions,
        labels=LABELS,
        timeline=timeline,
        mapping=mapping,
        alpha=alpha,
        threshold=threshold,
        sample_ids=[1],
    )
    # CUSUM: 0.9-0.5=0.4, +0.45=0.85 >= 0.8 -> one alert on 2020-05-08, then 365 days of rest.
    assert [a["date"] for a in o["alerts"]] == ["2020-05-08"]
    assert o["binary"]["tp"] == 1 and o["binary"]["fp"] == 0
    assert o["per_class"]["Clear Cut"]["f1"] == 1.0
    assert o["lag"]["median_days"] == 7.0


@pytest.mark.parametrize("alpha", [0.99])
def test_operational_silence_is_a_miss(mapping, alpha):
    days = [date(2019, 1, 1), date(2020, 5, 3), date(2020, 5, 8)]
    predictions = _predictions(mapping, days, [0.1, 0.9, 0.9])
    timeline = pl.DataFrame({"sample_id": [1] * 3, "date": days, "sensor": ["s2"] * 3})
    params = {name: alpha for name in mapping.disturbance_names}
    o = evaluate_operational(
        predictions,
        labels=LABELS,
        timeline=timeline,
        mapping=mapping,
        alpha=params,
        threshold=params,
        sample_ids=[1],
    )
    assert o["binary"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1}
