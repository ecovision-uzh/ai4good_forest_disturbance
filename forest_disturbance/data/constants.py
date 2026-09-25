"""Fixed facts about the DISFOR dataset: sensors, bands and label codes."""

SENSORS = ("s1", "s2")

# Sentinel-1 radar: two polarisations, stored as dB x 100 (int16).
S1_BANDS = ("VV", "VH")

# Sentinel-2 optical: the 12 L2A bands in wavelength order. This is the channel
# order of every S2 tensor the data loader returns.
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")

BANDS = {"s1": S1_BANDS, "s2": S2_BANDS}

# Zarr arrays inside patches/{sample_id}.zarr. S2 is split by ground resolution.
# The S2 date list lives on `s2_scl`, the S1 date list on `s1`.
S2_ARRAYS = {"s2_10m": 10, "s2_20m": 20, "s2_60m": 60}  # name -> metres per pixel
PATCH_SIZE = 252  # pixels at 10 m. The annotated location is pixel [126, 126].

# All DISFOR label codes. Hundreds = healthy (1xx) or disturbed (2xx);
# tens and units refine the class. See docs/02_dataset.md.
LABEL_NAMES = {
    100: "Healthy Vegetation",
    110: "Undisturbed Forest",
    120: "Revegetation",
    121: "With Trees (after clear cut)",
    122: "Canopy closing (after thinning/defoliation)",
    123: "Without Trees (shrubs and grasses, no reforestation visible)",
    200: "Disturbed",
    210: "Planned",
    211: "Clear Cut",
    212: "Thinning",
    213: "Forestry Mulching (Non Forest Vegetation Removal)",
    220: "Salvage",
    221: "After Biotic Disturbance",
    222: "After Abiotic Disturbance",
    230: "Biotic",
    231: "Bark Beetle (with decline)",
    232: "Gypsy Moth (temporary)",
    240: "Abiotic",
    241: "Drought",
    242: "Wildfire",
    243: "Wind",
    244: "Avalanche",
    245: "Flood",
}

# Labels describing a slow, ongoing disturbance rather than a one-day event.
CONTINUOUS_LABELS = (230, 231, 232)

# Target value ignored by the loss and the frame metrics.
IGNORE_INDEX = -100
