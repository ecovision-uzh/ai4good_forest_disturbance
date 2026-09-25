"""LightningModule: the training/validation loop around a model.

Lightning calls these methods for us:
    training_step         forward + loss on one batch (Lightning does backward and optimizer step)
    validation_step       forward + loss + collect predictions
    on_validation_epoch_end
                          save predictions to <output_dir>/predictions/epoch_XXX.parquet and
                          compute frame, non-operational and operational metrics
To try a new model, write an nn.Module whose forward(batch) returns the class
logits [B, classes] and pass it here.
"""

import json
import warnings
from datetime import date
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from forest_disturbance.data.labels import LabelMapping
from forest_disturbance.metrics.events import probability_column, s2_dates
from forest_disturbance.metrics.frame_metrics import FrameMetrics
from forest_disturbance.metrics.non_operational import evaluate_non_operational
from forest_disturbance.metrics.operational import evaluate_operational, flatten


class DisturbanceModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        mapping: LabelMapping,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        output_dir: str | Path | None = None,
        evaluation: dict | None = None,
    ) -> None:
        """Args:
        model: Network; forward(batch) -> class logits [B, classes].
        loss: Callable(logits, target) -> scalar loss.
        mapping: Label mapping (class names for prediction files and metrics).
        lr, weight_decay: AdamW settings.
        output_dir: Where validation predictions and metrics are written. None = do not write.
        evaluation: The `evaluation` section of the config (event metrics settings).
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.mapping = mapping
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.evaluation = evaluation or {}
        self.frame_metrics = FrameMetrics(
            mapping.num_classes, mapping.no_disturbance_id, forget_days=loss.forget_days
        )
        self.class_names = [mapping.class_names[i] for i in range(mapping.num_classes)]
        self._val_rows: list[dict[str, np.ndarray]] = []

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def _shared_step(self, batch: dict, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(batch)
        loss = self.loss(logits, batch["target"])
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sample_id"]),
        )
        return logits, loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        _, loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits, _ = self._shared_step(batch, "val")
        self.frame_metrics.update(logits, batch["target"])
        if not self.trainer.sanity_checking:
            self._val_rows.append(
                {
                    "sample_id": batch["sample_id"].cpu().numpy(),
                    "date": batch["target_date"].cpu().numpy(),
                    "sensor": np.array(batch["target_sensor"]),
                    "label": batch["target"]["label"].cpu().numpy(),
                    "days_since_event": batch["target"]["days_since_event"].cpu().numpy(),
                    "probabilities": logits.detach().float().softmax(dim=1).cpu().numpy(),
                }
            )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.frame_metrics.compute("val"))
        if not self.trainer.sanity_checking:
            _log_confusion_matrix(
                self.loggers,
                self.frame_metrics.class_confusion.compute(),
                self.class_names,
                self.current_epoch,
            )
        self.frame_metrics.reset()
        if self.trainer.sanity_checking or not self._val_rows:
            return

        predictions = self._collect_predictions()
        metrics = self.event_metrics(predictions)
        self.log_dict(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, int | float) and v is not None
            }
        )
        if self.output_dir is not None:
            folder = self.output_dir / "predictions"
            folder.mkdir(parents=True, exist_ok=True)
            predictions.write_parquet(folder / f"epoch_{self.current_epoch:03d}.parquet")
            (folder / f"epoch_{self.current_epoch:03d}_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )

    def _collect_predictions(self) -> pl.DataFrame:
        parts = {
            key: np.concatenate([rows[key] for rows in self._val_rows]) for key in self._val_rows[0]
        }
        self._val_rows.clear()
        probabilities = parts.pop("probabilities").astype(np.float64)
        table = pl.DataFrame(
            {
                "sample_id": parts["sample_id"].astype(np.int64),
                "date": [date.fromordinal(int(d)) for d in parts["date"]],
                "sensor": parts["sensor"],
                "label": parts["label"].astype(np.int64),
                "days_since_event": parts["days_since_event"].astype(np.float64),
            }
        )
        return table.with_columns(
            pl.Series(probability_column(name), probabilities[:, i])
            for i, name in enumerate(self.class_names)
        )

    def event_metrics(self, predictions: pl.DataFrame) -> dict:
        """Non-operational metrics (and operational ones if enabled) on validation predictions."""
        datamodule = self.trainer.datamodule
        horizons = tuple(self.evaluation.get("horizons_days", (14, 30, 60, 365)))
        metrics = {
            f"non_operational/{name}": value
            for name, value in evaluate_non_operational(
                predictions,
                labels=datamodule.labels,
                s2_dates=s2_dates(datamodule.frames),
                mapping=self.mapping,
                horizons=horizons,
                threshold=self.evaluation.get("threshold", 0.5),
            ).items()
        }
        operational = self.evaluation.get("operational")
        if operational:
            try:
                for horizon in horizons:  # how long after an event an alert still counts
                    result = evaluate_operational(
                        predictions,
                        labels=datamodule.labels,
                        timeline=datamodule.frames,
                        mapping=self.mapping,
                        t_after_days=horizon,
                        **operational,
                    )
                    metrics.update(flatten(result, suffix=f"_{horizon}d"))
            except AssertionError as error:
                # Fold 4 holds an event in the first year of its sample, before monitoring can start.
                warnings.warn(f"Operational metrics skipped: {error}", stacklevel=1)
        return metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def _log_confusion_matrix(
    loggers, confusion: torch.Tensor, class_names: list[str], epoch: int
) -> None:
    """Send a confusion matrix picture to Weights & Biases (skipped for other loggers)."""
    wandb_logger = next((logger for logger in loggers if isinstance(logger, WandbLogger)), None)
    if wandb_logger is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    matrix = confusion.cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    wandb_logger.experiment.log({"val/confusion_matrix": wandb.Image(figure), "epoch": epoch})
    plt.close(figure)
