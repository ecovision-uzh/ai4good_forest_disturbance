# Forest disturbance detection from space <img align="right" width="110" height="110" src="docs/assets/logo.svg">

A storm flattens a stand in one night. Bark beetles take a season. A harvester comes and
goes in a week. Your job: spot all three from satellite images, as early as you can, and
say which one it was.

![The task](docs/figures/task.png)

## 🚀 Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # install uv, once per machine
uv sync && source .venv/bin/activate               # install everything into .venv/
export DISFOR_DATA_ROOT=/path/to/disfor_data       # where the data lives
pytest                                             # check it all works (~2 min)
python scripts/train.py trainer.max_epochs=1 trainer.limit_train_batches=500 logger.kind=csv
```

Run everything from this folder. Stuck? → [Setup](docs/06_setup.md).

## 🎯 The baseline to beat

A small MLP on simple statistics of one pixel ([`configs/statistical_mlp.yaml`](configs/statistical_mlp.yaml)),
trained on fold 0. Its alerts, scored at four horizons ([what these mean](docs/04_metrics.md)):

| horizon | binary precision | binary recall | binary F1 | macro precision | macro recall | macro F1 |
|---|---|---|---|---|---|---|
| 14 days | 0.10 | 0.08 | 0.09 | 0.16 | 0.04 | 0.05 |
| 30 days | 0.14 | 0.12 | 0.13 | 0.17 | 0.05 | 0.07 |
| 60 days | 0.30 | 0.25 | 0.28 | 0.23 | 0.11 | 0.12 |
| 365 days | 0.65 | 0.55 | 0.60 | 0.52 | 0.35 | 0.36 |

Go beat it.

## 📚 Where to go next

| | |
|---|---|
| [The problem](docs/01_project.md) | why forests, why satellites, what exactly you predict |
| [The data](docs/02_dataset.md) | samples, labels, images, folds |
| [Data to predictions](docs/03_baseline_model.md) | what the loader builds and what the baseline does with it |
| [Metrics](docs/04_metrics.md) | how results are scored, and which number to quote |
| [The code](docs/05_codebase.md) | file map, configs, adding your own model |
| [Setup](docs/06_setup.md) | install, data, cluster, Weights & Biases |
