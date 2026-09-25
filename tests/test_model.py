"""Both baseline configs train on a fake batch, on CPU and (if available) GPU."""

import pytest
import torch

from forest_disturbance.build import build_module
from forest_disturbance.config import load_config
from forest_disturbance.data.batch import collate
from tests.conftest import DEVICES, REPO, random_example


@pytest.mark.parametrize("config_name", ["statistical_mlp", "statistical_mlp_notemporal"])
@pytest.mark.parametrize("device", DEVICES)
def test_forward_backward(device, config_name, mapping, monkeypatch):
    monkeypatch.setenv("DISFOR_DATA_ROOT", "unused")
    config = load_config(REPO / f"configs/{config_name}.yaml")
    module = build_module(config, mapping).to(device)
    batch = collate([random_example(label=label, recent=2 + label) for label in [0, 3, 6, 6]])
    batch = module.transfer_batch_to_device(batch, torch.device(device), 0)

    optimizer = module.configure_optimizers()
    losses = []
    for _ in range(5):
        logits = module(batch)
        loss = module.loss(logits, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert logits.shape == (4, mapping.num_classes)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
