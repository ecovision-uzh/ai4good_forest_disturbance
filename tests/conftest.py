import os
from datetime import date, timedelta
from pathlib import Path

import pytest
import torch

from forest_disturbance.data.dataset import TimeSeries
from forest_disturbance.data.labels import LabelMapping

REPO = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def mapping() -> LabelMapping:
    return LabelMapping.from_yaml(REPO / "configs/label_mapping.yaml")


@pytest.fixture
def data_root() -> Path:
    """Real dataset folder; tests using it are skipped when DISFOR_DATA_ROOT is not set."""
    root = os.environ.get("DISFOR_DATA_ROOT")
    if not root:
        pytest.skip("DISFOR_DATA_ROOT is not set")
    return Path(root)


def random_example(sample_id: int = 0, label: int = 6, recent: int = 5, yearly: int = 3) -> dict:
    """A fake dataset example with S2 centre-pixel values in the dataset's value range."""
    target = date(2020, 6, 1)

    def series(n: int, last: date) -> TimeSeries:
        dates = [last - timedelta(days=5 * (n - 1 - i)) for i in range(n)]
        return TimeSeries(dates, torch.rand(n, 12, 1, 1) * 0.5)

    return {
        "sample_id": sample_id,
        "target_date": target,
        "target_sensor": "s2",
        "inputs": {
            "s2": {"recent": series(recent, target), "yearly": [series(yearly, date(2019, 6, 10))]}
        },
        "target": {"label": label, "days_since_event": 10.0 if label != 6 else float("inf")},
    }
