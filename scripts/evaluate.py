"""Compute the event metrics of a saved prediction file, e.g. with other alert settings.

    python scripts/evaluate.py runs/statistical_mlp_20260101-120000/predictions/epoch_009.parquet
    python scripts/evaluate.py <file> --config configs/statistical_mlp.yaml evaluation.threshold=0.3

The data folder is $DISFOR_DATA_ROOT (or data.root of the config). Fold and metric settings come from the config (default: the run's
config.yaml next to the predictions folder). Every validation sample of the fold is
evaluated: samples missing from the file count as missed events.
Writes <file stem>_evaluation.json next to the predictions.
"""

import argparse
import json
import os
from pathlib import Path

import polars as pl

from forest_disturbance.build import build_mapping
from forest_disturbance.config import load_config
from forest_disturbance.data.labels import load_labels
from forest_disturbance.metrics.events import load_split_sample_ids, s2_dates
from forest_disturbance.metrics.non_operational import evaluate_non_operational
from forest_disturbance.metrics.operational import evaluate_operational


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--config", type=Path, default=None, help="Default: <run folder>/config.yaml"
    )
    parser.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = parser.parse_args()
    config = load_config(
        args.config or args.predictions.parent.parent / "config.yaml", args.overrides
    )

    # Prefer the current DISFOR_DATA_ROOT: the saved config holds the path used at training time.
    root = Path(os.environ.get("DISFOR_DATA_ROOT") or config["data"]["root"])
    mapping = build_mapping(config)
    labels = load_labels(root / "labels.parquet")
    frames = pl.read_parquet(root / "zarr_frames.parquet")
    sample_ids = load_split_sample_ids(root, "val", config["data"]["fold"])
    predictions = pl.read_parquet(args.predictions)
    evaluation = config["evaluation"]

    non_operational = evaluate_non_operational(
        predictions,
        labels=labels,
        s2_dates=s2_dates(frames),
        mapping=mapping,
        horizons=tuple(evaluation["horizons_days"]),
        threshold=evaluation["threshold"],
        sample_ids=sample_ids,
    )
    operational = {
        f"{h}d": evaluate_operational(
            predictions,
            labels=labels,
            timeline=frames,
            mapping=mapping,
            sample_ids=sample_ids,
            t_after_days=h,
            **evaluation["operational"],
        )
        for h in evaluation["horizons_days"]
    }

    print("\nOperational (filtered alerts): the main scores")
    print(f"{'':>8} {'--- binary: found, any class ---':>32} {'--- macro: right class ---':>30}")
    print(f"{'horizon':>8}" + f"{'precision':>11}{'recall':>9}{'F1':>8}" * 2)
    for h in evaluation["horizons_days"]:
        result = operational[f"{h}d"]
        row = [
            result[kind][name]
            for kind in ("binary", "macro")
            for name in ("precision", "recall", "f1")
        ]
        print(
            f"{h:>7}d" + "".join(f"{v:>{w}.3f}" for v, w in zip(row, (11, 9, 8) * 2, strict=True))
        )

    print("\nNon-operational (raw predictions, a diagnostic)")
    print(f"{'horizon':>8} {'recall':>8} {'precision':>10} {'PR-AUC':>8} {'type F1':>8}")
    for h in evaluation["horizons_days"]:
        row = [
            non_operational[f"{name}_{h}d"]
            for name in ("binary_recall", "alert_precision", "PR_AUC", "window_agent_f1_macro")
        ]
        print(f"{h:>7}d " + " ".join(f"{'-' if v is None else f'{v:.3f}':>9}" for v in row))

    output = args.predictions.with_name(args.predictions.stem + "_evaluation.json")
    output.write_text(
        json.dumps({"non_operational": non_operational, "operational": operational}, indent=2)
    )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
