"""Train a model and evaluate it on the validation split after every epoch.

    python scripts/train.py --config configs/statistical_mlp.yaml [key=value ...]

Examples:
    python scripts/train.py --config configs/statistical_mlp.yaml trainer.max_epochs=1 logger.kind=csv
    python scripts/train.py --config configs/statistical_mlp.yaml data.fold=2 run_name=fold2

Outputs in runs/<run_name>_<date-time>/:
    config.yaml                       the exact config used
    predictions/epoch_XXX.parquet     validation predictions (one row per acquisition)
    predictions/epoch_XXX_metrics.json
    checkpoints/last.ckpt
"""

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from forest_disturbance.build import REPO_ROOT, build_datamodule, build_mapping, build_module
from forest_disturbance.config import load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=REPO_ROOT / "configs/statistical_mlp.yaml")
    parser.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)

    output_dir = (
        Path(config["output_root"]) / f"{config['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True)
    save_config(config, output_dir / "config.yaml")
    print(f"Run folder: {output_dir}")

    # Same seed = same initial weights and same order of training batches.
    L.seed_everything(config["seed"], workers=True)
    mapping = build_mapping(config)
    datamodule = build_datamodule(config, mapping)
    module = build_module(config, mapping, output_dir)

    trainer_config = config["trainer"]
    callbacks = [ModelCheckpoint(dirpath=output_dir / "checkpoints", save_top_k=0, save_last=True)]
    if trainer_config["early_stopping_patience"] is not None:
        callbacks.append(
            EarlyStopping(
                monitor=trainer_config["early_stopping_monitor"],
                mode="max",
                patience=trainer_config["early_stopping_patience"],
            )
        )
    trainer = L.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        max_epochs=trainer_config["max_epochs"],
        limit_train_batches=trainer_config["limit_train_batches"],
        default_root_dir=output_dir,
        logger=make_logger(config, output_dir),
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)


def make_logger(config: dict, output_dir: Path):
    logger = config["logger"]
    if logger["kind"] == "wandb":
        return WandbLogger(
            name=output_dir.name,
            project=logger["project"],
            entity=logger["entity"],
            offline=logger["offline"],
            save_dir=output_dir,
            config=config,
        )
    if logger["kind"] == "csv":
        return CSVLogger(save_dir=output_dir, name="", version="")
    if logger["kind"] == "none":
        return False
    raise ValueError(f"Unknown logger.kind {logger['kind']!r}: use wandb, csv or none.")


if __name__ == "__main__":
    main()
