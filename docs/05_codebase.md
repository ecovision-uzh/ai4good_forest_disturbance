# 🧭 The code

[← Metrics](04_metrics.md) · [Docs home](README.md) · next: [Setup →](06_setup.md)

```text
configs/statistical_mlp.yaml   the baseline: every setting of a run
configs/statistical_mlp_notemporal.yaml   the same model with learned per-image features
configs/label_mapping.yaml     label codes -> the 7 classes
scripts/train.py               train + validate; writes predictions and metrics every epoch
scripts/evaluate.py            recompute metrics from a saved prediction file
scripts/plot_sample.py         plot one sample: time series, labels, predictions, alerts
scripts/build_pixel_cache.py   one-off: extract the centre pixels from the zarr patches
scripts/slurm/train.sbatch     example cluster job
forest_disturbance/
  config.py    build.py        read the YAML; turn it into objects
  data/        frames.py labels.py readers.py dataset.py batch.py datamodule.py
  models/      statistical_mlp.py loss.py layers.py lit_module.py
  metrics/     frame_metrics.py non_operational.py operational.py
  viz/         style.py icons.py sample.py raster.py timeline.py maps.py counts.py
notebooks/explore.ipynb        start here: the tables, the figures, the loader, the predictions
tests/                         pytest (the data tests need DISFOR_DATA_ROOT)
```

Every file has a docstring that says what it does and how — start there, not here.

## 🎨 Figures

Every figure in these pages comes from `forest_disturbance.viz`, which ships with the
repository — use it for your own report instead of writing matplotlib from scratch.
[`notebooks/explore.ipynb`](../notebooks/explore.ipynb) runs through all of it.

```python
from forest_disturbance.viz import plot_sample, plot_hexmap, plot_class_counts, style
style.use()

figure = plot_sample(1005, root, patches=("s2", "s1"), series=("NDVI", ("VV", "VH")))
style.save(figure, "sample.png")
```

| Function | Draws |
|---|---|
| `plot_sample` | one sample: picture strips, the annotation lane, any 1-D series, targets, model output |
| `plot_hexmap`, `plot_hexmap_grid` | where samples are, one map or a grid of them |
| `plot_code_counts`, `plot_class_counts` | how many annotations of each kind, optionally split by fold |

`plot_sample` is the one you will use. Every panel is optional:
`patches=("s2", "s1", "index", "pca")` chooses the picture strips, `series=` takes one name
per panel or a tuple to overlay several (any of the 12 bands, `VV`/`VH`, `NDVI NDMI NDWI NBR
NDRE`, or `PC1 PC2 PC3`), `crop=` sets how much ground a picture covers, `predictions=`
adds the model panel, `span=` zooms on a date range. Sentinel-2 composites use fixed
reflectance ranges (`raster.S2_CLIP`), so two pictures of two samples are directly
comparable and a cloud cannot wash the forest out.

Colours, icons and sizes live in `viz/style.py` and are shared by every figure: change
`style.SIZES["label"]` and the bubbles, badges and ribbons all follow.

## One config per run

```bash
python scripts/train.py data.fold=1 model.dropout=0.1 trainer.max_epochs=20 run_name=my_test
```

`${DISFOR_DATA_ROOT}` in the YAML comes from your environment. A misspelled key raises an
error instead of doing nothing. The final config is saved in the run folder and sent to
Weights & Biases. Copy `configs/statistical_mlp.yaml` for each experiment — don't edit the baseline.

Every config key becomes an object in `build.py`. Read that one file and you know what each
key does.

## I want to…

| … | Do this |
|---|---|
| try a new model | new file in `models/`, add it to `MODELS` in `build.py`, set `model.name` ([details](03_baseline_model.md#swap-in-your-own-model)) |
| use Sentinel-1 too | `data.sensors=[s1,s2]`: the batch gets `inputs["s1"]` (the baseline ignores it, yours does not have to). `data.target_sensors=null` also turns every S1 date into an example — 4× more examples |
| use the image, not one pixel | `data.reader=zarr data.image_size=32` (slow: it reads the 1.7 TB patches; raise `data.num_workers`) |
| change the time windows | `data.days_before`, `data.years_context`, `data.days_context` |
| change the classes | copy `configs/label_mapping.yaml`, edit it, point `data.label_mapping` at it |
| change the loss | `models/loss.py`, or a new class wired in `build.py` |
| try other alert settings | `python scripts/evaluate.py <predictions.parquet> evaluation.threshold=0.3` |
| look at predictions | `python scripts/plot_sample.py <id> --predictions runs/<run>/predictions/epoch_009.parquet` |
| make a figure for the report | `forest_disturbance.viz`, see [Figures](#-figures) |
| run on another fold | `data.fold=2` |
| stop training early | `trainer.early_stopping_patience=3` |
| load a trained model | build the model, run one batch through it (the lazy layers need it), then `load_state_dict` on `checkpoints/last.ckpt` |

[← Metrics](04_metrics.md) · [Docs home](README.md) · next: [Setup →](06_setup.md)
