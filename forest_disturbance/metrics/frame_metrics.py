"""Frame-level metrics: every validation acquisition counts as one independent prediction.

These are quick to compute and good for monitoring training, but they do not
say whether *events* are detected in time. For that, see non_operational.py and
operational.py (docs/03_metrics.md).

Three views of the same predictions (class = argmax of the logits):

* class   : 7-way classification (6 disturbance classes + No Disturbance), macro-averaged.
* binary  : "recent disturbance?" yes/no. Positive = disturbance at most `forget_days`
            old; negative = no disturbance or older than `negative_after_days`.
            Frames in between are skipped.
* agent   : only frames of disturbed samples: is the disturbance type right?
"""

import warnings

import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from forest_disturbance.data.constants import IGNORE_INDEX


class FrameMetrics(nn.Module):
    def __init__(
        self,
        num_classes: int,
        no_disturbance_id: int,
        forget_days: float,
        negative_after_days: float | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.no_disturbance_id = no_disturbance_id
        self.forget_days = forget_days
        self.negative_after_days = (
            2.0 * forget_days if negative_after_days is None else negative_after_days
        )

        per_class = {"num_classes": num_classes, "average": None, "zero_division": 0}
        self.classification = MetricCollection(
            {
                "precision": MulticlassPrecision(**per_class),
                "recall": MulticlassRecall(**per_class),
                "f1": MulticlassF1Score(**per_class),
                "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            }
        )
        self.agent = self.classification.clone()
        self.binary = MetricCollection(
            {
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                "accuracy": BinaryAccuracy(),
            }
        )
        self.class_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        self.agent_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: dict) -> None:
        predicted = logits.argmax(dim=1)
        label, age = target["label"], target["days_since_event"]
        valid = (label != IGNORE_INDEX) & (label >= 0) & (label < self.num_classes)
        if valid.any():
            self.classification.update(predicted[valid], label[valid])
            self.class_confusion.update(predicted[valid], label[valid])

        finite_age = torch.isfinite(age)
        disturbed = valid & (label != self.no_disturbance_id)
        positive = disturbed & finite_age & (age <= self.forget_days)
        negative = (valid & (label == self.no_disturbance_id)) | (
            disturbed & finite_age & (age >= self.negative_after_days)
        )
        scored = positive | negative
        if scored.any():
            self.binary.update(
                (predicted[scored] != self.no_disturbance_id).long(), positive[scored].long()
            )

        if disturbed.any():
            self.agent.update(predicted[disturbed], label[disturbed])
            self.agent_confusion.update(predicted[disturbed], label[disturbed])

    def compute(self, prefix: str) -> dict[str, torch.Tensor]:
        with (
            warnings.catch_warnings()
        ):  # metrics without any update (e.g. no disturbed frame) return 0
            warnings.filterwarnings(
                "ignore", message=r"The ``compute`` method of metric .* was called before"
            )
            classification, agent = self.classification.compute(), self.agent.compute()
            binary = self.binary.compute()
            class_cm, agent_cm = self.class_confusion.compute(), self.agent_confusion.compute()
        disturbance = (
            torch.arange(self.num_classes, device=agent_cm.device) != self.no_disturbance_id
        )
        return {
            f"{prefix}/frame_class_precision": _mean_present(
                classification["precision"], class_cm.sum(0) > 0
            ),
            f"{prefix}/frame_class_recall": _mean_present(
                classification["recall"], class_cm.sum(1) > 0
            ),
            f"{prefix}/frame_class_f1": _mean_present(
                classification["f1"], class_cm.sum(0) + class_cm.sum(1) > 0
            ),
            f"{prefix}/frame_class_accuracy": classification["accuracy"],
            f"{prefix}/frame_binary_precision": binary["precision"],
            f"{prefix}/frame_binary_recall": binary["recall"],
            f"{prefix}/frame_binary_f1": binary["f1"],
            f"{prefix}/frame_binary_accuracy": binary["accuracy"],
            f"{prefix}/frame_agent_f1": _mean_present(
                agent["f1"][disturbance],
                agent_cm[:, disturbance].sum(0) + agent_cm[disturbance].sum(1) > 0,
            ),
            f"{prefix}/frame_agent_accuracy": agent["accuracy"],
        }

    def reset(self) -> None:
        for metric in (
            self.classification,
            self.agent,
            self.binary,
            self.class_confusion,
            self.agent_confusion,
        ):
            metric.reset()


def _mean_present(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Average over classes that occur in the targets or the predictions."""
    return values[present].mean() if present.any() else values.sum() * 0.0
