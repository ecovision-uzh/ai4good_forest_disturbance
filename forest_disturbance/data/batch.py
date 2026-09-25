"""Stack examples into a batch. Time series have different lengths, so we pad them.

A batch is a plain dict of tensors (B = batch size, T = longest series in the batch):

    batch["sample_id"]         LongTensor [B]
    batch["target_date"]       LongTensor [B]          day number, date.toordinal()
    batch["target_sensor"]     list[str]  [B]
    batch["inputs"]["s2"]["recent"]          one padded series:
        ["values"]  FloatTensor [B, T, C, H, W]   zeros where padded
        ["dates"]   LongTensor  [B, T]            day numbers, 0 where padded
        ["mask"]    BoolTensor  [B, T]            True = real acquisition
    batch["inputs"]["s2"]["yearly"]          list (one per previous year) of padded series
    batch["target"]["label"]            LongTensor  [B]
    batch["target"]["days_since_event"] FloatTensor [B]
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from forest_disturbance.data.dataset import TimeSeries


def collate(examples: list[dict]) -> dict:
    sensors = list(examples[0]["inputs"])
    inputs = {}
    for sensor in sensors:
        per_example = [example["inputs"][sensor] for example in examples]
        years = len(per_example[0]["yearly"])
        inputs[sensor] = {
            "recent": pad_series([windows["recent"] for windows in per_example]),
            "yearly": [
                pad_series([windows["yearly"][k] for windows in per_example]) for k in range(years)
            ],
        }
    return {
        "sample_id": torch.tensor([e["sample_id"] for e in examples], dtype=torch.long),
        "target_date": torch.tensor(
            [e["target_date"].toordinal() for e in examples], dtype=torch.long
        ),
        "target_sensor": [e["target_sensor"] for e in examples],
        "inputs": inputs,
        "target": {
            "label": torch.tensor([e["target"]["label"] for e in examples], dtype=torch.long),
            "days_since_event": torch.tensor(
                [e["target"]["days_since_event"] for e in examples], dtype=torch.float32
            ),
        },
    }


def pad_series(series: list[TimeSeries]) -> dict[str, torch.Tensor]:
    values = pad_sequence([s.values for s in series], batch_first=True)
    lengths = torch.tensor([len(s.dates) for s in series])
    mask = torch.arange(values.shape[1])[None, :] < lengths[:, None]
    dates = pad_sequence(
        [torch.tensor([d.toordinal() for d in s.dates], dtype=torch.long) for s in series],
        batch_first=True,
    )
    return {"values": values, "dates": dates, "mask": mask}
