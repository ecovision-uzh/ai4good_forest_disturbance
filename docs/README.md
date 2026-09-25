# Documentation

Six short pages. Read [The problem](01_project.md) and [Setup](06_setup.md) today;
come back for the rest when you need them.

| | |
|---|---|
| [🌲 The problem](01_project.md) | why forests, why satellites, what you predict, what to read |
| [💾 The data](02_dataset.md) | samples, labels, images, folds |
| [📊 Metrics](03_metrics.md) | annotations, windows, alerts, and the numbers to beat |
| [🎯 The model](04_model.md) | what the model reads, the baseline, its loss, from probabilities to alerts |
| [🧭 The code](05_codebase.md) | file map, configs, adding your own model |
| [🔧 Setup](06_setup.md) | install, data, cluster, Weights & Biases |

## Words you will meet

Sorted by topic, then alphabetically. No remote sensing or forestry background is assumed.

### 🛰️ Satellites and images

| | |
|---|---|
| **band** | one channel of a satellite image: the light measured in one range of wavelengths. Sentinel-2 has 12 (`B01` … `B12`), Sentinel-1 has 2 (`VV`, `VH`) |
| **backscatter** | what a radar measures: the share of its own signal that bounces back. Given in decibels (dB); depends on the structure and water content of what it hits, not on its colour |
| **cloud mask (SCL)** | Sentinel-2's own per-pixel label of each image: cloud, cloud shadow, snow, water, vegetation, bare soil… Stored as `s2_scl`. Images more than half cloudy are already removed |
| **multispectral** | measuring more colours than a camera: besides red, green and blue, also near-infrared and short-wave infrared, which the eye cannot see but which tell healthy leaves, dry wood and bare soil apart |
| **NDVI / NDMI / NDWI** | spectral indices, each a normalised difference of two bands: greenness (B08 − B04) / (B08 + B04), moisture (B08 − B11) / (B08 + B11), water (B03 − B08) / (B03 + B08). Between −1 and 1 |
| **near-infrared (NIR) / short-wave infrared (SWIR)** | wavelengths just beyond red (B08, B8A) and further out (B11, B12). Healthy leaves reflect NIR strongly; SWIR drops when plants hold water |
| **optical** | a sensor that measures reflected sunlight, like a camera: needs daylight and a clear sky |
| **patch** | the 252 × 252 pixel image (2.5 km across) around a sample, for every date |
| **pixel, resolution** | one cell of the image grid. The 10 m bands of Sentinel-2 (and Sentinel-1) have 10 m × 10 m pixels; other Sentinel-2 bands are coarser (20 m, 60 m) |
| **polarisation (VV, VH)** | the orientation of the radar wave sent and received: vertical–vertical, vertical–horizontal. VH responds more to the volume of branches and leaves |
| **reflectance** | the share of sunlight a surface reflects in one band, between 0 and 1 |
| **revisit** | how often a satellite sees the same place: every few days for Sentinel-2 (clouds allowing), every 6–12 days for Sentinel-1 |
| **SAR (synthetic aperture radar)** | a radar that sends microwaves and records the echo, building a sharp image as it flies. Works at night and through clouds, but the images are grainy ("speckle") and do not look like photos |
| **Sentinel-1** | the European Space Agency's radar satellites (SAR). Two bands, `VV` and `VH`, at 10 m |
| **Sentinel-2** | the European Space Agency's optical multispectral satellites. 12 bands from blue to short-wave infrared, at 10, 20 or 60 m, free and open |
| **spectral index** | a simple formula over a few bands that highlights one property (greenness, moisture, water): NDVI, NDMI, NDWI |

### 🌲 Forests and disturbances

| | |
|---|---|
| **agent / attribution** | the cause of a disturbance (wind, fire, insects, harvest…) / naming it |
| **bark beetle** | an insect whose outbreaks kill spruce stands over months to years: the main **Biotic** disturbance here |
| **biotic / abiotic** | caused by living organisms (insects, disease) / by physical events (wind, fire, drought) |
| **clear cut** | harvesting (almost) all trees of a stand at once |
| **disturbance** | a sudden or gradual loss of trees or canopy: harvest, storm, fire, insects… |
| **revegetation** | the forest growing back after a disturbance; counted as "No Disturbance" here |
| **salvage logging** | cutting and removing trees already damaged by a storm, fire or insects |
| **thinning** | removing some trees of a stand so that the others grow better; a partial cut |
| **windthrow** | trees blown down by a storm: the **Wind** class |

### 💾 The dataset

| | |
|---|---|
| **campaign** | one of the three labelling efforts that built the dataset, each aimed at different disturbances |
| **event / period** | an annotation on one date (most disturbances) / over a stretch of time (healthy forest, Biotic) |
| **fold** | one of the 5 train/validation splits. Fold 0 for exploring, all five for a result |
| **frame / acquisition** | one satellite image of one sample on one date |
| **ignored code** | a label too rare or vague to learn (drought, flood…): the loss skips its dates, and scoring of that sample stops there |
| **label code / label mapping** | the expert's raw label (e.g. 243 Wind) / how codes are grouped into the 7 classes ([`configs/label_mapping.yaml`](../configs/label_mapping.yaml)) |
| **sample** | one location — the pixel at the centre of its patch — and its whole time series (`sample_id`) |
| **zarr** | array format stored as a folder; `zarr_id` = position on the time axis |

### 🎯 The model

| | |
|---|---|
| **days_since_event** | days between the last disturbance and a date; the loss uses it to forget old events. The model does not predict it |
| **logits / class probabilities** | the model's 7 raw scores / the same after a softmax, written to the prediction files |
| **recent / yearly window** | what the loader reads for a date: the last 30 days / the same season one year earlier |
| **target date / target frame** | a date used as a training or validation example |

### 📊 The metrics

| | |
|---|---|
| **alert** | a date on which the model declares a disturbance of one class |
| **alert filter** | turns daily probabilities into alerts: a vote per date, a running sum (**CUSUM**, cumulative sum) that fires at a threshold |
| **binary F1 / macro F1** | the two scores: detection (an alert of any class) / attribution (right class, averaged over the 6 classes) |
| **buffer** | the part of a window before the annotated date: up to 14 days, because annotated dates can be late |
| **horizon** | how long after the annotation an alert still counts: 14, 30, 60 or 365 days |
| **near real-time (NRT)** | detecting a disturbance soon after it happens, from the images so far, rather than years later |
| **precision / recall / F1** | the share of alerts that were right / the share of disturbances found / their harmonic mean |
| **rest period (cooldown)** | after an alert, the 365 days during which that class stays silent at that location |
| **TP / FP / FN / void** | an alert matching a window / an alert matching none / a window without alert / an extra alert in a Biotic window, not counted |
| **window** | the time around an annotation in which an alert counts: buffer + horizon |
