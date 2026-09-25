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
trained on fold 0. Two numbers matter:

- **binary F1** — detection: did an alert land on the disturbance, whatever its class?
- **macro F1** — attribution: did an alert of the *right* class land on it? Averaged over the 6 classes.

Each is scored at four horizons: how long after a disturbance an alert still counts
([details](docs/04_metrics.md)).

| horizon | binary F1 | macro F1 |
|---|---|---|
| 14 days | 0.12 | 0.10 |
| 30 days | 0.19 | 0.15 |
| 60 days | 0.34 | 0.28 |
| 365 days | 0.60 | 0.48 |

Go beat it.

## 📚 Where to go next

| | |
|---|---|
| [The problem](docs/01_project.md) | why forests, why satellites, what exactly you predict, what to read |
| [The data](docs/02_dataset.md) | samples, labels, images, folds |
| [Data to predictions](docs/03_baseline_model.md) | what the loader builds and what the baseline does with it |
| [Metrics](docs/04_metrics.md) | how results are scored, and which number to quote |
| [The code](docs/05_codebase.md) | file map, configs, adding your own model |
| [Setup](docs/06_setup.md) | install, data, cluster, Weights & Biases |
