"""Create the data module and the model from a config dict (see configs/statistical_mlp.yaml).

This is the only place where config keys are turned into Python objects.

To add your own model: write it in forest_disturbance/models/, add it to MODELS below,
and set `model.name` (plus its own keyword arguments) in your config.
"""

from pathlib import Path

from forest_disturbance.data.datamodule import DisforDataModule
from forest_disturbance.data.labels import LabelMapping
from forest_disturbance.models.lit_module import DisturbanceModule
from forest_disturbance.models.loss import RecentDisturbanceLoss
from forest_disturbance.models.statistical_mlp import StatisticalMLP

REPO_ROOT = Path(__file__).resolve().parents[1]

# model.name in the config -> model class. Every other key under `model` is passed to the class.
MODELS = {
    "statistical_mlp": StatisticalMLP,
}


def build_mapping(config: dict) -> LabelMapping:
    path = Path(config["data"]["label_mapping"])
    return LabelMapping.from_yaml(path if path.is_absolute() else REPO_ROOT / path)


def build_datamodule(config: dict, mapping: LabelMapping) -> DisforDataModule:
    data = {key: value for key, value in config["data"].items() if key != "label_mapping"}
    return DisforDataModule(mapping=mapping, **data)


def build_module(
    config: dict, mapping: LabelMapping, output_dir: Path | None = None
) -> DisturbanceModule:
    model_args = {key: value for key, value in config["model"].items() if key != "name"}
    model = MODELS[config["model"]["name"]](num_classes=mapping.num_classes, **model_args)
    loss = RecentDisturbanceLoss(no_disturbance_id=mapping.no_disturbance_id, **config["loss"])
    return DisturbanceModule(
        model,
        loss,
        mapping,
        **config["optimizer"],
        output_dir=output_dir,
        evaluation=config["evaluation"],
    )
