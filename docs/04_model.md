# 🎯 The model

[← Metrics](03_metrics.md) · [Docs home](README.md) · next: [The code →](05_codebase.md)

![The pipeline](figures/pipeline.png)

Four steps: what the model reads, the baseline model, the loss that trains it, and the
filter that turns its probabilities into alerts.

> **🧪 This page is your playground.** Every choice below — which images the loader reads,
> the input features, the model, the loss, the alert filter and its thresholds — is one
> reasonable starting point, not the answer. Challenge them, change them, replace them.

## 1. What the model reads

Every Sentinel-2 date inside a sample's annotated period is one example — 1,017,043 of them
in fold 0, plus 248,370 for validation. For that **target date**, the loader reads two short
series of the labelled pixel: the current context and the past context.

![Recent and yearly windows](figures/windows.png)

| Series | Window | What it is for |
|---|---|---|
| recent | the last 30 days, target date included | what the forest looks like now |
| yearly | the same date one year earlier, ± 15 days | what it looked like in the same season |

A date is only used if both windows hold an image. Nothing after the target date is ever
loaded. Window sizes: `data.days_before`, `data.years_context`, `data.days_context`.
The batch layout is in the docstring of [`data/batch.py`](../forest_disturbance/data/batch.py).

## 2. The baseline

![The baseline model](figures/model.png)

`StatisticalMLP` ([`configs/statistical_mlp.yaml`](../configs/statistical_mlp.yaml)): about
58 k parameters, Sentinel-2 centre pixel only. An MLP turns each image into features; each
window becomes their mean and standard deviation; an MLP compares the two windows and gives
one probability per class.

![A clear cut, with the baseline's class probabilities](figures/example_clear_cut.png)

The bottom panel is what your model writes to
`runs/<run>/predictions/epoch_XXX.parquet`: one probability per class, on every date.

## 3. The loss

What should the model answer on a given date? The class of the most recent disturbance —
but only while it is recent. Right after a storm the answer is Wind. Three years later the
forest has grown back, and nothing in the last 30 days shows the storm: asking for "Wind"
then would only teach the model to guess. So the target fades with the age of the event
(`days_since_event`, taken from the labels):

- up to ~215 days: the true class;
- from ~215 to 335 days: the true class *or* No Disturbance, whichever the model already
  prefers — any other class is still wrong;
- after 335 days: No Disturbance.

The loss is a cross-entropy against that target.

![The loss and how it forgets](figures/loss.png)

The bottom panel shows what two confident answers cost on a clear cut. Saying "Clear Cut" is
cheap while the event is recent and expensive once it is forgotten; saying "No Disturbance"
does the reverse. Rare classes are drawn more often during training (`data.sampling_alpha`).
Code: [`models/loss.py`](../forest_disturbance/models/loss.py).

## 4. From probabilities to alerts

The metrics score **alerts**, not probabilities. By default, the evaluation turns the
probabilities into alerts with this filter, per sample and per class:

```text
vote  = +1 if probability > alpha else −1       on every image date
score = max(0, score + vote)                    a running sum, never below zero
alert when score ≥ threshold, then score = 0
after an alert: that class is silent for 365 days
```

Thresholding turns each date into a yes or a no, so one spectacular image cannot fire an
alert on its own. The running sum then asks for that yes to repeat: a class needs a run of
confident dates, and a quiet spell wipes out the evidence it had built. The per-class `alpha`
and `threshold` live in `evaluation.operational`. They are a starting point: tune them, or
replace the filter with your own. The 365-day silence after an alert is part of the metrics
and stays. Always report the numbers with the default values too, so they compare with the
baseline.

With this filter, the baseline scores the numbers of the [metrics page](03_metrics.md):
binary F1 0.12 / 0.19 / 0.34 / 0.60 and macro F1 0.10 / 0.15 / 0.28 / 0.48 at 14 / 30 / 60 /
365 days, on fold 0.

## What a run produces

```
runs/statistical_mlp_20260921-1015/
├── config.yaml                          exactly what you ran
├── predictions/epoch_000.parquet        one row per validation date: labels + class probabilities
├── predictions/epoch_000_metrics.json   all metrics of that epoch
└── checkpoints/last.ckpt
```

The same metrics go to Weights & Biases (or `metrics.csv` with `logger.kind=csv`): see
[Track your runs](06_setup.md#5-track-your-runs).

## Swap in your own model

1. Write an `nn.Module` whose `forward(batch)` returns the class scores (logits) `[B, 7]`,
   in `forest_disturbance/models/`.
2. Add it to `MODELS` in `forest_disturbance/build.py`.
3. Copy `configs/statistical_mlp.yaml`, set `model.name` and your own keys, run it.

Everything else stays the same, so your numbers stay comparable.

[← Metrics](03_metrics.md) · [Docs home](README.md) · next: [The code →](05_codebase.md)
