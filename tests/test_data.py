"""Tests on the real dataset. Skipped unless DISFOR_DATA_ROOT is set.

DISFOR_DATA_ROOT=/path/to/data pytest tests/test_data.py
"""

import random

import pytest
import torch

from forest_disturbance.build import build_datamodule, build_module
from forest_disturbance.config import load_config
from forest_disturbance.data.readers import PatchReader
from tests.conftest import DEVICES, REPO


def _config(data_root, *overrides):
    return load_config(
        REPO / "configs/statistical_mlp.yaml", [f"data.root={data_root}", *overrides]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sensors", ["[s2]", "[s1,s2]"])
def test_dataloader_batches_run_through_model(data_root, mapping, device, sensors):
    config = _config(
        data_root, f"data.sensors={sensors}", "data.num_workers=2", "data.batch_size=8"
    )
    datamodule = build_datamodule(config, mapping)
    datamodule.setup()
    assert len(datamodule.train_set) > 0 and len(datamodule.val_set) > 0
    module = build_module(config, mapping).to(device)

    for loader in (datamodule.train_dataloader(), datamodule.val_dataloader()):
        for i, batch in enumerate(loader):
            batch = module.transfer_batch_to_device(batch, torch.device(device), 0)
            for sensor in config["data"]["sensors"]:
                recent = batch["inputs"][sensor]["recent"]
                assert recent["values"].device.type == device
                assert recent["values"].shape[:2] == recent["mask"].shape
                is_target_sensor = torch.tensor(
                    [s == sensor for s in batch["target_sensor"]], device=device
                )
                assert recent["mask"].any(dim=1)[is_target_sensor].all(), (
                    "the target acquisition is in its recent series"
                )
            with torch.no_grad():
                logits = module(batch)
            assert torch.isfinite(logits).all()
            if i == 2:
                break


def test_zarr_reader_matches_pixel_cache(data_root, mapping):
    if not (data_root / "patches").exists():
        pytest.skip("no zarr patches in this data folder")
    config = _config(data_root, "data.sensors=[s1,s2]")
    datamodule = build_datamodule(config, mapping)
    datamodule.setup()
    dataset = datamodule.val_set
    zarr_reader = PatchReader(data_root / "patches", image_size=1)
    for index in random.Random(0).sample(range(len(dataset)), 20):
        example = dataset.examples.row(index, named=True)
        for sensor in ("s1", "s2"):
            _, zarr_ids = dataset.timelines[(example["sample_id"], sensor)]
            ids = [int(i) for i in zarr_ids[:10]]
            assert torch.equal(
                dataset.reader.read(example["sample_id"], sensor, ids),
                zarr_reader.read(example["sample_id"], sensor, ids),
            )


def test_full_patch_loading(data_root, mapping):
    if not (data_root / "patches").exists():
        pytest.skip("no zarr patches in this data folder")
    config = _config(data_root, "data.reader=zarr", "data.image_size=null", "data.sensors=[s1,s2]")
    datamodule = build_datamodule(config, mapping)
    datamodule.setup()
    example = datamodule.val_set[0]
    for sensor, bands in (("s1", 2), ("s2", 12)):
        values = example["inputs"][sensor]["recent"].values
        assert values.shape[1:] == (bands, 252, 252)
        assert values.abs().sum() > 0
