"""Training loss: "which disturbance happened recently?"

A frame right after a clear cut is labelled "Clear Cut". But 3 years later the
forest looks healthy again and nobody could tell from the recent images. So the
label is *forgotten* with time:

    event age  0 ........ ~200 days ... 215 ... ~245 days ........ ▶
    loss       cross-entropy(true class)  |  blend  | cross-entropy(No Disturbance)

Between `forget_days - 4 * transition_days` and `forget_days + 4 * transition_days`
the loss smoothly moves from the true class to "either the true class or No
Disturbance, whichever the model already prefers" (the minimum of both losses).
After that, only No Disturbance is correct. The age of the event comes from the
labels (`days_since_event`); the model does not predict it.
"""

import torch
from torch import nn
from torch.nn import functional as F

from forest_disturbance.data.constants import IGNORE_INDEX


class RecentDisturbanceLoss(nn.Module):
    def __init__(
        self,
        no_disturbance_id: int,
        forget_days: float = 215.0,
        transition_days: float = 30.0,
    ) -> None:
        super().__init__()
        self.no_disturbance_id = no_disturbance_id
        self.forget_days = forget_days
        self.transition_days = transition_days

    def forward(self, logits: torch.Tensor, target: dict) -> torch.Tensor:
        """logits [B, classes] and the batch targets -> mean loss (a scalar)."""
        label, age = target["label"], target["days_since_event"]
        valid = label != IGNORE_INDEX

        # Per-example cross-entropy against the true class and against "No Disturbance".
        ce_true = self._cross_entropy(logits, label)
        ce_none = self._cross_entropy(logits, torch.full_like(label, self.no_disturbance_id))

        forget = self.forget_factor(age)  # 0 = recent event, 1 = forgotten
        loss = (1.0 - forget) * ce_true + forget * torch.minimum(ce_true, ce_none)
        loss = torch.where(self.hard_forget_mask(age), ce_none, loss)
        # `logits.sum() * 0.0` keeps the result attached to the graph when nothing is valid.
        return loss[valid].mean() if valid.any() else logits.sum() * 0.0

    def forget_factor(self, age: torch.Tensor) -> torch.Tensor:
        """Sigmoid going from 0 (event younger than forget_days) to 1 (older)."""
        width = max(self.transition_days, torch.finfo(torch.float32).eps)
        finite_age = torch.where(
            torch.isfinite(age), age, torch.full_like(age, self.forget_days + width * 100.0)
        )
        factor = torch.sigmoid((finite_age - self.forget_days) / width)
        return torch.where(self.hard_forget_mask(age), torch.ones_like(factor), factor)

    def hard_forget_mask(self, age: torch.Tensor) -> torch.Tensor:
        """True when there is no event (age = inf) or it is far older than forget_days."""
        width = max(self.transition_days, torch.finfo(torch.float32).eps)
        return ~torch.isfinite(age) | (age >= self.forget_days + 4.0 * width)

    def _cross_entropy(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if not (label != IGNORE_INDEX).any():
            return logits.sum(dim=1) * 0.0
        return F.cross_entropy(logits, label, ignore_index=IGNORE_INDEX, reduction="none")
