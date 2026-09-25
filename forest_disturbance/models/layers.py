"""Small building blocks shared by models."""

from torch import nn


class MLP(nn.Sequential):
    """Linear -> LayerNorm -> ReLU (-> Dropout) for each hidden layer, then a final Linear.

    Works on the last tensor dimension, so input can be [..., features].
    `input_dim=None` lets PyTorch infer the input size from the first batch (LazyLinear).
    """

    def __init__(
        self, input_dim: int | None, hidden_dims: list[int], output_dim: int, dropout: float = 0.0
    ):
        layers: list[nn.Module] = []
        previous = input_dim
        for dim in hidden_dims:
            layers += [_linear(previous, dim), nn.LayerNorm(dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = dim
        layers.append(_linear(previous, output_dim))
        super().__init__(*layers)


def _linear(input_dim: int | None, output_dim: int) -> nn.Module:
    return nn.LazyLinear(output_dim) if input_dim is None else nn.Linear(input_dim, output_dim)
