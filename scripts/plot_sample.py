"""Plot one sample: time series, labels and (optionally) predictions and alerts.

    python scripts/plot_sample.py 889
    python scripts/plot_sample.py 889 --patches s2 s1 --series NDVI VV,VH
    python scripts/plot_sample.py 889 --predictions runs/<run>/predictions/epoch_009.parquet
    python scripts/plot_sample.py --random 5 --event-class Wind --predictions ...

Saves PNG files in plots/ (or --output-dir). The data folder is $DISFOR_DATA_ROOT or --data-root.
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import polars as pl

from forest_disturbance.build import build_mapping
from forest_disturbance.config import load_config
from forest_disturbance.data.labels import load_labels
from forest_disturbance.metrics.operational import evaluate_operational
from forest_disturbance.viz import plot_sample, style


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample_ids", type=int, nargs="*")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DISFOR_DATA_ROOT"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="prediction parquet of a run; also shows operational alerts",
    )
    parser.add_argument(
        "--patches",
        nargs="*",
        default=[],
        metavar="STRIP",
        help="image strips to draw: s2, s1, index, pca (needs the zarr patches)",
    )
    parser.add_argument("--n-patches", type=int, default=5, help="pictures per strip")
    parser.add_argument("--crop", type=int, default=96, help="picture width, in 10 m pixels")
    parser.add_argument(
        "--series",
        nargs="*",
        default=["NDVI"],
        metavar="NAMES",
        help="one panel per entry; comma-separated names share a panel, e.g. NDVI VV,VH",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="plot N random samples (from the prediction file if given)",
    )
    parser.add_argument(
        "--event-class",
        default=None,
        help="with --random: only samples with this disturbance class, e.g. Wind",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    assert args.data_root, "Set DISFOR_DATA_ROOT or pass --data-root."
    args.data_root = Path(args.data_root)

    predictions = pl.read_parquet(args.predictions) if args.predictions else None
    alerts = None
    sample_ids = list(args.sample_ids)
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/statistical_mlp.yaml",
        [f"data.root={args.data_root}"],
    )
    mapping = build_mapping(config)
    labels = load_labels(args.data_root / "labels.parquet")
    if predictions is not None:
        frames = pl.read_parquet(args.data_root / "zarr_frames.parquet")
        alerts = evaluate_operational(
            predictions,
            labels=labels,
            timeline=frames,
            mapping=mapping,
            **config["evaluation"]["operational"],
        )["alerts"]

    if args.random:
        candidates = labels
        if args.event_class:
            codes = [
                code
                for code, class_id in mapping.code_to_class.items()
                if mapping.class_names[class_id] == args.event_class
            ]
            assert codes, (
                f"Unknown class {args.event_class!r}; choose from {mapping.disturbance_names}."
            )
            candidates = labels.filter(pl.col("label").is_in(codes))
        pool = set(candidates["sample_id"].to_list())
        if predictions is not None:
            pool &= set(predictions["sample_id"].to_list())
        sample_ids += random.sample(sorted(pool), min(args.random, len(pool)))

    style.use()
    series = tuple(tuple(entry.split(",")) for entry in args.series)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        figure = plot_sample(
            sample_id,
            args.data_root,
            series=series,
            patches=tuple(args.patches),
            n_patches=args.n_patches,
            crop=args.crop,
            predictions=predictions,
            alerts=alerts,
        )
        print(f"wrote {style.save(figure, args.output_dir / f'sample_{sample_id}.png')}")


if __name__ == "__main__":
    main()
