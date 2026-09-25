"""Turn satellite arrays into images you can look at.

Everything here takes plain arrays (time, channel, height, width) as the data
loader returns them, and gives back RGB images in [0, 1]:

    s2_rgb(cube)                      natural colour (B04, B03, B02)
    s2_rgb(cube, ("B08", "B04", "B03"))   false colour, vegetation is red
    s1_rgb(cube)                      radar: R = VV, G = VH, B = VV - VH
    index_gray(cube, "NDVI")          one spectral index, grey or coloured
    pca_rgb(cube)                     any number of channels -> 3 PCA components

The stretch (which value becomes black, which becomes white) is computed once
over the whole time series, so the same forest keeps the same colour on every
date and changes are real changes.
"""

import numpy as np

from forest_disturbance.data.constants import S1_BANDS, S2_BANDS

#: Spectral indices, as (positive band, negative band): index = (a - b) / (a + b).
INDICES: dict[str, tuple[str, str]] = {
    "NDVI": ("B08", "B04"),  # greenness / leaf area
    "NDMI": ("B08", "B11"),  # water in the leaves
    "NDWI": ("B03", "B08"),  # open water
    "NBR": ("B08", "B12"),  # burn severity
    "NDRE": ("B08", "B05"),  # red edge, chlorophyll stress
}

S2_OFFSET, S2_SCALE = 1000.0, 10000.0  # stored value -> reflectance
S1_SCALE = 100.0  # stored value -> dB
S1_NODATA = -32768
#: Fixed dB ranges of the Sentinel-1 composite (R = VV, G = VH, B = VV - VH).
S1_CLIP_DB = ((-25.0, 0.0), (-30.0, -5.0), (0.0, 15.0))

#: Fixed reflectance range of every Sentinel-2 band in a colour image.
#:
#: The top of each range sits a little above the brightest ordinary ground (the 92nd
#: to 95th percentile of the dataset's own pixels), so cloud and snow are clipped to
#: white instead of darkening everything else, and so two pictures of two different
#: samples, or of two different dates, can be compared directly. Bands that usually
#: appear together share a range, which keeps the colour balance of a natural-colour
#: image. Override an entry to change one band.
S2_CLIP: dict[str, tuple[float, float]] = {
    "B01": (0.0, 0.22), "B02": (0.0, 0.22), "B03": (0.0, 0.22), "B04": (0.0, 0.22),
    "B05": (0.0, 0.30), "B06": (0.0, 0.45), "B07": (0.0, 0.45), "B08": (0.0, 0.45),
    "B8A": (0.0, 0.45), "B09": (0.0, 0.45), "B11": (0.0, 0.35), "B12": (0.0, 0.35),
}  # fmt: skip

#: Brightens the mid-tones of a colour image; forest reflects little light, so the
#: raw range would look almost black.
S2_GAMMA = 0.6


def reflectance(cube: np.ndarray) -> np.ndarray:
    """Sentinel-2 stored values -> surface reflectance in [0, 1] (may be slightly negative)."""
    return (np.asarray(cube, dtype=np.float32) - S2_OFFSET) / S2_SCALE


def decibels(cube: np.ndarray) -> np.ndarray:
    """Sentinel-1 stored values -> dB, with no-data as NaN."""
    values = np.asarray(cube, dtype=np.float32)
    return np.where(values == S1_NODATA, np.nan, values / S1_SCALE)


def center_crop(array: np.ndarray, size: int | None) -> np.ndarray:
    """Crop the last two dimensions to `size` around the centre (None = no crop)."""
    if size is None:
        return array
    height, width = array.shape[-2:]
    if size > min(height, width):
        raise ValueError(f"crop {size} is larger than the array ({height} x {width})")
    top, left = height // 2 - size // 2, width // 2 - size // 2
    return array[..., top : top + size, left : left + size]


def center_index(array: np.ndarray) -> tuple[int, int]:
    """Row and column of the annotated pixel inside an array (its centre)."""
    height, width = array.shape[-2:]
    return height // 2, width // 2


def spectral_index(
    values: np.ndarray, name: str, *, bands: tuple[str, ...] = S2_BANDS
) -> np.ndarray:
    """Normalised difference index from Sentinel-2 values with channels on axis 1.

    Works for a cube (time, channel, height, width) and for a table (time, channel).
    """
    if name not in INDICES:
        raise ValueError(f"unknown index {name!r}; known: {', '.join(INDICES)}")
    first, second = INDICES[name]
    a = _channel(values, first, bands)
    b = _channel(values, second, bands)
    return (a - b) / np.where(np.abs(a + b) < 1e-6, 1e-6, a + b)


def _channel(values: np.ndarray, band: str, bands: tuple[str, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[:, bands.index(band)]


def stretch(
    values: np.ndarray,
    *,
    percentiles: tuple[float, float] = (2.0, 98.0),
    mask: np.ndarray | None = None,
    joint: bool = False,
    gamma: float = 1.0,
) -> np.ndarray:
    """Scale to [0, 1] with one percentile range per channel, shared over the time axis.

    Args:
        values: (time, channel, height, width).
        percentiles: Values below the first become 0, above the second become 1.
        mask: (time, height, width) boolean; only those pixels set the range
            (e.g. drop clouds and snow, which would wash the rest out).
        joint: One range for all channels instead of one each. Use it for colour
            images: a range per channel would shift the white balance.
        gamma: Values below 1 brighten the mid-tones.
    """
    values = np.asarray(values, dtype=np.float32)
    scaled = np.empty_like(values)
    if joint:
        sample = (
            values[:, :, mask[0] if False else slice(None)] if mask is None else values[:, :, :, :]
        )
        pool = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
        if mask is not None and mask.any():
            flat_mask = (
                np.broadcast_to(mask[:, None], values.shape)
                .transpose(1, 0, 2, 3)
                .reshape(values.shape[1], -1)
            )
            pool = np.where(flat_mask, pool, np.nan)
        low, high = (
            np.nanpercentile(pool[np.isfinite(pool)], percentiles)
            if np.isfinite(pool).any()
            else (0.0, 1.0)
        )
        scaled = (values - low) / max(float(high - low), 1e-6)
    else:
        for channel in range(values.shape[1]):
            band = values[:, channel]
            sample = band[mask] if mask is not None and mask.any() else band
            sample = sample[np.isfinite(sample)]
            low, high = np.percentile(sample, percentiles) if sample.size else (0.0, 1.0)
            scaled[:, channel] = (band - low) / max(float(high - low), 1e-6)
    scaled = np.clip(np.nan_to_num(scaled), 0.0, 1.0)
    return scaled**gamma if gamma != 1.0 else scaled


def s2_rgb(
    cube: np.ndarray,
    bands: tuple[str, str, str] = ("B04", "B03", "B02"),
    *,
    clip: bool = True,
    gamma: float | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sentinel-2 colour images: (time, 12, H, W) -> (time, H, W, 3) in [0, 1].

    Default is natural colour. ("B08", "B04", "B03") gives the classic false colour
    in which healthy vegetation is bright red. Any three bands work.

    Args:
        clip: True scales every band with its fixed range from `S2_CLIP`, so the same
            reflectance always gets the same colour: pictures stay comparable between
            dates and between samples, and a cloud or a snow field is clipped to white
            instead of washing the forest out. False instead stretches the percentiles
            of this cube, which adapts to a dull scene but makes nothing comparable.
        gamma: Below 1 brightens the mid-tones (default `S2_GAMMA`).
        percentiles, mask: Only used when `clip` is False.
    """
    picked = np.stack([_channel(cube, band, S2_BANDS) for band in bands], axis=1)
    if not clip:
        # One range for the three bands: a range per band would break the colour balance.
        stretched = stretch(picked, percentiles=percentiles, mask=mask, joint=True, gamma=0.85)
        return np.moveaxis(stretched, 1, -1)

    values = reflectance(picked)
    scaled = np.empty_like(values)
    for i, band in enumerate(bands):
        low, high = S2_CLIP.get(band, (0.0, 0.35))
        scaled[:, i] = (values[:, i] - low) / max(high - low, 1e-6)
    scaled = np.clip(np.nan_to_num(scaled), 0.0, 1.0) ** (S2_GAMMA if gamma is None else gamma)
    return np.moveaxis(scaled, 1, -1)


def s1_rgb(cube: np.ndarray, clip_db: tuple[tuple[float, float], ...] = S1_CLIP_DB) -> np.ndarray:
    """Sentinel-1 colour images: (time, 2, H, W) -> (time, H, W, 3) in [0, 1].

    R = VV, G = VH, B = VV - VH, each with a fixed dB range so that two dates
    are directly comparable. Forest is usually pale green-grey; clear cuts darken VH.
    """
    values = decibels(cube)
    vv, vh = values[:, S1_BANDS.index("VV")], values[:, S1_BANDS.index("VH")]
    channels = [vv, vh, vv - vh]
    scaled = [
        np.clip((c - low) / max(high - low, 1e-6), 0, 1)
        for c, (low, high) in zip(channels, clip_db, strict=True)
    ]
    return np.nan_to_num(np.stack(scaled, axis=-1))


def index_gray(
    cube: np.ndarray, name: str = "NDVI", *, vmin: float | None = None, vmax: float | None = None
) -> tuple[np.ndarray, float, float]:
    """One spectral index per date: (time, 12, H, W) -> ((time, H, W), vmin, vmax).

    Returns the index and the value range used, so a colour bar can be drawn.
    """
    values = spectral_index(reflectance(cube), name)
    low = float(np.nanpercentile(values, 2)) if vmin is None else vmin
    high = float(np.nanpercentile(values, 98)) if vmax is None else vmax
    return values, low, high


def index_rgb(
    cube: np.ndarray, names: tuple[str, str, str] = ("NDVI", "NDMI", "NDWI")
) -> np.ndarray:
    """Three spectral indices stacked as one false-colour image, stretched over time."""
    values = reflectance(cube)
    stacked = np.stack([spectral_index(values, name) for name in names], axis=1)
    return np.moveaxis(stretch(stacked), 1, -1)


def fit_pca(
    values: np.ndarray, n_components: int = 3, *, percentiles: tuple[float, float] = (2.0, 98.0)
) -> dict:
    """Fit a PCA on feature vectors, keeping the stretch of the projected values.

    Args:
        values: (..., channel) — any number of leading dimensions (time, pixels, ...).
        n_components: How many components to keep (3 to make an RGB image).

    Returns:
        A dict to pass to `apply_pca`. Uses numpy's SVD, so no extra dependency.
    """
    flat = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    mean = flat.mean(axis=0)
    # Economy SVD of the centred data: rows of Vt are the principal directions.
    _, variance, directions = np.linalg.svd(flat - mean, full_matrices=False)
    components = directions[:n_components]
    projected = (flat - mean) @ components.T
    low, high = np.percentile(projected, percentiles, axis=0)
    explained = (variance**2 / max((variance**2).sum(), 1e-12))[:n_components]
    return {
        "mean": mean,
        "components": components,
        "low": low,
        "high": high,
        "explained": explained,
    }


def apply_pca(values: np.ndarray, pca: dict) -> np.ndarray:
    """Project onto a fitted PCA and scale to [0, 1]: (..., channel) -> (..., n_components)."""
    projected = (np.asarray(values, dtype=np.float32) - pca["mean"]) @ pca["components"].T
    span = np.maximum(pca["high"] - pca["low"], 1e-6)
    return np.clip((projected - pca["low"]) / span, 0.0, 1.0)


def pca_rgb(cube: np.ndarray, *, pca: dict | None = None) -> tuple[np.ndarray, dict]:
    """False-colour image of any multi-channel cube: (time, C, H, W) -> (time, H, W, 3).

    The three strongest components of the whole series become red, green and blue,
    so pixels that behave alike get alike colours. Use it on model features or
    embeddings, or on the 12 Sentinel-2 bands.
    """
    channels_last = np.moveaxis(np.asarray(cube, dtype=np.float32), 1, -1)
    pca = pca or fit_pca(channels_last)
    return apply_pca(channels_last, pca), pca
