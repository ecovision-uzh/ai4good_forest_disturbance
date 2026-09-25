"""Baseline: an MLP on simple statistics of the centre-pixel time series.

For each target acquisition the model sees two short Sentinel-2 series of the
annotated pixel: "recent" (last 30 days, including the target date) and "yearly"
(same season, one year earlier). It works in three steps:

  1. Frame features   each acquisition: 12 bands + NDVI, NDMI, NDWI  (15 numbers)
                      (optional: MLPs turn them into learned features)
  2. Statistics       per series: mean and std over time -> one vector for "recent",
                      one for "yearly"  (optional: one MLP on each)
  3. Prediction       concatenate -> MLP trunk -> class logits
                      (6 disturbance classes + No Disturbance)

An optional MLP is switched off by setting its output size to null in the config.
configs/statistical_mlp.yaml uses none of them (statistics of the raw values);
configs/statistical_mlp_notemporal.yaml uses all of them.

See docs/03_baseline_model.md for a picture.
"""

import torch
from torch import nn

from forest_disturbance.data.constants import S2_BANDS
from forest_disturbance.models.layers import MLP


class StatisticalMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        frame_dim: int | None = None,
        frame_hidden_dims: list[int] = (),
        stats_mlp_dim: int | None = None,
        stats_mlp_hidden_dims: list[int] = (),
        statistics: list[str] = ("mean", "std"),
        series_dim: int | None = None,
        series_hidden_dims: list[int] = (),
        trunk_dim: int = 64,
        trunk_hidden_dims: list[int] = (64, 64, 64),
        dropout: float = 0.0,
    ) -> None:
        """Args:
        num_classes: Number of output classes.
        frame_dim, frame_hidden_dims: MLP on each acquisition's 15 values. None = no MLP.
        stats_mlp_dim, stats_mlp_hidden_dims: A second per-acquisition MLP, right before
            the statistics. None = no MLP.
        statistics: Statistics over time, any of mean, std, min, max.
        series_dim, series_hidden_dims: One MLP on the "recent" statistics and one on the
            "yearly" statistics. None = no MLP.
        trunk_dim, trunk_hidden_dims: MLP on the concatenated series features.
        dropout: Dropout after every hidden layer.
        """
        super().__init__()
        self.statistics = list(statistics)
        # Step 1: per-acquisition features (12 bands + 3 spectral indices).
        num_inputs = len(S2_BANDS) + 3
        self.frame_mlp = _optional_mlp(num_inputs, frame_hidden_dims, frame_dim, dropout)
        frame_features = num_inputs if frame_dim is None else frame_dim
        self.stats_mlp = _optional_mlp(
            frame_features, stats_mlp_hidden_dims, stats_mlp_dim, dropout
        )
        # Step 2: one MLP per series type, after the statistics.
        self.recent_mlp = _optional_mlp(None, series_hidden_dims, series_dim, dropout)
        self.yearly_mlp = _optional_mlp(None, series_hidden_dims, series_dim, dropout)
        # Step 3: shared trunk, then the classification head (the trunk ends on a Linear,
        # so two Linear layers follow each other: kept as in the benchmark code).
        self.trunk = MLP(None, list(trunk_hidden_dims), trunk_dim, dropout)
        self.class_head = MLP(trunk_dim, [], num_classes, dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns class logits [B, num_classes]."""
        s2 = batch["inputs"]["s2"]
        recent = self.summarize(s2["recent"])
        if s2["yearly"]:
            # All previous years are pooled into one series.
            yearly = self.summarize(
                {
                    key: torch.cat([year[key] for year in s2["yearly"]], dim=1)
                    for key in ("values", "dates", "mask")
                }
            )
        else:
            yearly = torch.zeros_like(recent)
        features = torch.cat((self.recent_mlp(recent), self.yearly_mlp(yearly)), dim=-1)
        return self.class_head(self.trunk(features))

    def summarize(self, series: dict[str, torch.Tensor]) -> torch.Tensor:
        """One padded series [B, T, 12, H, W] -> statistics over time [B, features * len(statistics)]."""
        height, width = series["values"].shape[-2:]
        pixels = series["values"][
            :, :, :, height // 2, width // 2
        ]  # labelled (centre) pixel: [B, T, 12]
        frames = torch.cat((pixels, spectral_indices(pixels)), dim=-1)  # [B, T, 15]
        frames = self.stats_mlp(self.frame_mlp(frames))
        return torch.cat(
            [masked_statistic(frames, series["mask"], name) for name in self.statistics], dim=-1
        )


def _optional_mlp(
    input_dim: int | None, hidden_dims: list[int], output_dim: int | None, dropout: float
) -> nn.Module:
    """An MLP, or nothing (identity) when output_dim is None."""
    if output_dim is None:
        return nn.Identity()
    return MLP(input_dim, list(hidden_dims), output_dim, dropout)


def spectral_indices(pixels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """[..., 12 bands] -> [..., 3]: NDVI (greenness), NDMI (moisture), NDWI (water)."""
    band = {name: pixels[..., i] for i, name in enumerate(S2_BANDS)}
    red, green, nir, swir = band["B04"], band["B03"], band["B08"], band["B11"]

    def normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) / (a + b).clamp_min(eps)

    return torch.stack(
        (
            normalized_difference(nir, red),
            normalized_difference(nir, swir),
            normalized_difference(green, nir),
        ),
        dim=-1,
    )


def masked_statistic(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    """Statistic over time of [B, T, D] values, ignoring padded steps. Empty series -> zeros."""
    if values.shape[1] == 0:
        return values.new_zeros(values.shape[0], values.shape[2])
    valid = mask[:, :, None]
    count = valid.sum(dim=1).clamp_min(1)
    if name == "mean":
        return values.masked_fill(~valid, 0.0).sum(dim=1) / count
    if name == "std":
        mean = masked_statistic(values, mask, "mean")
        centered = (values - mean[:, None]).masked_fill(~valid, 0.0)
        return torch.sqrt(centered.square().sum(dim=1) / count + torch.finfo(values.dtype).eps)
    if name == "max":
        return torch.nan_to_num(
            values.masked_fill(~valid, -torch.inf).max(dim=1).values, neginf=0.0
        )
    if name == "min":
        return torch.nan_to_num(values.masked_fill(~valid, torch.inf).min(dim=1).values, posinf=0.0)
    raise ValueError(f"Unknown statistic {name!r}: use mean, std, min or max.")
