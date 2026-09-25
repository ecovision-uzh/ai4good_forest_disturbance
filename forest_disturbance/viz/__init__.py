"""Visualisation toolkit: one function per kind of figure, one shared style.

    from forest_disturbance.viz import style, plot_sample, plot_hexmap
    style.use()

    fig = plot_sample(889, root, patches=("s2", "s1"), series=("NDVI", ("VV", "VH")))
    fig, ax = plot_hexmap(samples)
    style.save(fig, "figure.png")

Modules:
    style     colours, fonts, annotation cards, save()
    sample    load one sample (labels, pixel values, patches) — `Sample.load(id, root)`
    raster    arrays -> images: colour composites, spectral indices, PCA
    timeline  `plot_sample`: pictures, annotations, series and model output through time
    maps      `plot_hexmap`, `plot_hexmap_grid`: where the samples are
    counts    `plot_code_counts`, `plot_class_counts`: how many annotations of each kind
    icons     class icons (colour emoji, or monochrome glyphs when there is no emoji font)
"""

from forest_disturbance.viz import icons, maps, raster, style
from forest_disturbance.viz.counts import plot_class_counts, plot_code_counts
from forest_disturbance.viz.maps import plot_hexmap, plot_hexmap_grid
from forest_disturbance.viz.sample import Sample
from forest_disturbance.viz.timeline import plot_sample

__all__ = [
    "Sample",
    "icons",
    "maps",
    "plot_class_counts",
    "plot_code_counts",
    "plot_hexmap",
    "plot_hexmap_grid",
    "plot_sample",
    "raster",
    "style",
]
