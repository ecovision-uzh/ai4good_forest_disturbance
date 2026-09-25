# 🔧 Setup

[← The code](05_codebase.md) · [Docs home](README.md)

Run everything from the repository folder.

## 1. Install

`uv` installs the right Python and every package into `.venv/`. No conda.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once per machine
uv sync
source .venv/bin/activate                           # or prefix commands with `uv run`
```

On a cluster, install on the login node — it has internet. The PyTorch build (CUDA 12.8)
also runs on CPU-only machines.

<details>
<summary>Without uv</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
pip install pytest jupyterlab
```
</details>

## 2. Point at the data

```bash
export DISFOR_DATA_ROOT=/path/to/disfor_data     # put this in your ~/.bashrc
```

The path on the course machines is not written down here — you will be given it separately.

The baseline reads `center_pixels.parquet` from that folder. If it is ever missing:
`python scripts/build_pixel_cache.py --data-root $DISFOR_DATA_ROOT --workers 32` (~5 min).

## 3. Check

```bash
pytest        # ~2 min; the data tests skip themselves if DISFOR_DATA_ROOT is unset
```

## 4. Train

```bash
# quick check: 500 batches, CSV logging, no account needed
python scripts/train.py trainer.max_epochs=1 trainer.limit_train_batches=500 logger.kind=csv

# the baseline: fold 0, 10 epochs, a quarter of the batches per epoch
python scripts/train.py trainer.limit_train_batches=0.25
```

One epoch takes ~10 minutes on a GPU with 8 workers: half training, half predicting the
248,370 validation dates. On a laptop use `data.num_workers=2`. Results land in
`runs/<run_name>_<date>/`. On a cluster, adapt
[`scripts/slurm/train.sbatch`](../scripts/slurm/train.sbatch) and `sbatch` it — never train
on a login node.

## 5. Track your runs

Every run gets its own folder, whatever the logger: `runs/<run_name>_<date-time>/`, with the
exact `config.yaml`, the predictions and metrics of every epoch (`predictions/`) and the last
checkpoint. Give each experiment a name you will recognise later: `run_name=cloud_mask_v1`.

Where the metric curves go is set by `logger.kind`:

| `logger.kind` | Where to look |
|---|---|
| `wandb` (default) | the [Weights & Biases](https://wandb.ai) dashboard: every run of your group side by side, with its config |
| `csv` | `runs/<run>/metrics.csv`, one row per logged step — no account needed |
| `none` | nothing extra; `predictions/epoch_XXX_metrics.json` is still written |

For W&B: create a free account and a team, run `wandb login` once, then set
`logger.entity=<your-team> logger.project=<name>`. The W&B run is named like the folder, and
its config is uploaded, so you can filter and compare runs by any setting. No internet on
the compute nodes? `logger.offline=true`, then `wandb sync runs/<run>/wandb/offline-run-*`
from the login node.

## 6. Look at things

```bash
jupyter lab notebooks/explore.ipynb
python scripts/plot_sample.py 889 --chips 5 --predictions runs/<run>/predictions/epoch_009.parquet
python scripts/evaluate.py runs/<run>/predictions/epoch_009.parquet
```

## When something breaks

| Message | Fix |
|---|---|
| `Environment variable DISFOR_DATA_ROOT is not set` | `export DISFOR_DATA_ROOT=/path/to/data` |
| bfloat16 not supported, or training crawls on an old GPU (V100, T4, P100) | `trainer.precision=32` |
| `NVIDIA driver … too old` | run on CPU: `CUDA_VISIBLE_DEVICES="" python ... trainer.accelerator=cpu` |
| data loading is slow | raise `data.num_workers`; keep `data.reader=pixel_cache` unless you need the patches |
| killed, out of memory | fewer workers, or ask for more memory |
| `unknown key` on an override | check the spelling in `configs/statistical_mlp.yaml` |
| `evaluate.py` cannot find the data of an old run | it uses `$DISFOR_DATA_ROOT` first, then the path saved in the run's `config.yaml` |
| `Operational metrics skipped` on `data.fold=4` | known: three 2015 clear cuts start before monitoring can begin. Everything else is still computed |

[← The code](05_codebase.md) · [Docs home](README.md)
