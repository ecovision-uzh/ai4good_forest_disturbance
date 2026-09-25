# Documentation

Six short pages. Read [The problem](01_project.md) and [Setup](06_setup.md) today;
come back for the rest when you need them.

| | |
|---|---|
| [🌲 The problem](01_project.md) | why forests, why satellites, what you predict, what to read |
| [💾 The data](02_dataset.md) | samples, labels, images, folds |
| [🎯 Data to predictions](03_baseline_model.md) | the loader, the baseline, the loss |
| [📊 Metrics](04_metrics.md) | three families, and which number to quote |
| [🧭 The code](05_codebase.md) | file map, configs, adding your own model |
| [🔧 Setup](06_setup.md) | install, data, cluster, Weights & Biases |

## Words you will meet

| | |
|---|---|
| **sample** | one location and its whole time series (`sample_id`) |
| **frame / acquisition** | one satellite image of one sample on one date |
| **target frame** | a date used as a training or validation example |
| **patch** | the 252 × 252 pixel (2.5 km) image around a sample |
| **zarr** | array format stored as a folder; `zarr_id` = position on the time axis |
| **NDVI** | (B08 − B04) / (B08 + B04); high for healthy green vegetation |
| **agent / attribution** | the cause of a disturbance (wind, fire, insects…) |
| **event** | one annotated disturbance the model should find |
| **alert** | a date on which the model declares a disturbance |
| **CUSUM** | running sum of the per-date votes of one class; it ignores single blips and fires once confident dates pile up |
| **rest period** | after an alert, the days during which that class stays silent |
| **B window** | up to 14 days before an event where an alert still counts as on time |
| **non-operational / operational** | scoring the raw probabilities / scoring the filtered alerts |
| **days_since_event** | days between the last disturbance and this date; the loss uses it to forget old events |
