"""Maps of where the samples are, as hexagon bins over Europe.

Thousands of dots on a map hide each other. Hexagon bins do not: each cell shows
how many samples fall in it (or the average of any column), so dense regions read
as colour instead of as a blob.

    fig, ax = plot_hexmap(samples)                        # samples per cell
    fig, ax = plot_hexmap(samples, cell_km=40)            # finer mosaic
    fig = plot_hexmap_grid({f"fold {k}": rows_k, ...})    # one small map per group

Two projections, both computed in numpy (no extra dependency):

    "lcc"  (default) Lambert conformal conic, the standard European map. Shapes and
           hexagons keep their form; areas are slightly distorted away from 52°N.
    "laea" Lambert azimuthal equal-area (EPSG:3035). Every hexagon covers exactly the
           same ground area, at the price of visibly stretched shapes at the edges.
"""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm

from forest_disturbance.viz import style

ASSETS = Path(__file__).resolve().parent / "assets"

# GRS80 ellipsoid, as used by the official European projections.
_A, _F = 6378137.0, 1 / 298.257222101
#: Lambert azimuthal equal-area, ETRS89-LAEA (EPSG:3035).
LAEA = {"lat0": 52.0, "lon0": 10.0, "false_easting": 4321000.0, "false_northing": 3210000.0}
#: Lambert conformal conic, ETRS89-LCC (EPSG:3034): two standard parallels across Europe.
LCC = {
    "lat1": 35.0,
    "lat2": 65.0,
    "lat0": 52.0,
    "lon0": 10.0,
    "false_easting": 4000000.0,
    "false_northing": 2800000.0,
}
#: Corner of the drawn window, in degrees (west, east, south, north).
EUROPE_BOUNDS = (-11.0, 33.0, 34.0, 70.5)


def project(lon, lat, projection: str = "lcc") -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude in degrees -> projected metres."""
    lon = np.deg2rad(np.asarray(lon, dtype=float))
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    if projection == "laea":
        return _laea(lon, lat)
    if projection == "lcc":
        return _lcc(lon, lat)
    raise ValueError(f"unknown projection {projection!r}: use 'lcc' or 'laea'")


def _laea(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e2 = 2 * _F - _F**2
    e = np.sqrt(e2)
    lat0, lon0 = np.deg2rad(LAEA["lat0"]), np.deg2rad(LAEA["lon0"])

    def authalic(phi):
        sin_phi = np.sin(phi)
        return (1 - e2) * (
            sin_phi / (1 - e2 * sin_phi**2)
            - (1 / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi))
        )

    q_pole, q0, q = authalic(np.pi / 2), authalic(lat0), authalic(lat)
    beta0, beta = np.arcsin(q0 / q_pole), np.arcsin(q / q_pole)
    rq = _A * np.sqrt(q_pole / 2)
    d = _A * np.cos(lat0) / (np.sqrt(1 - e2 * np.sin(lat0) ** 2) * rq * np.cos(beta0))
    delta = lon - lon0
    b = rq * np.sqrt(
        2 / (1 + np.sin(beta0) * np.sin(beta) + np.cos(beta0) * np.cos(beta) * np.cos(delta))
    )
    east = LAEA["false_easting"] + b * d * np.cos(beta) * np.sin(delta)
    north = LAEA["false_northing"] + (b / d) * (
        np.cos(beta0) * np.sin(beta) - np.sin(beta0) * np.cos(beta) * np.cos(delta)
    )
    return east, north


def _lcc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.sqrt(2 * _F - _F**2)
    lat1, lat2 = np.deg2rad(LCC["lat1"]), np.deg2rad(LCC["lat2"])
    lat0, lon0 = np.deg2rad(LCC["lat0"]), np.deg2rad(LCC["lon0"])

    def m(phi):  # radius of the parallel
        return np.cos(phi) / np.sqrt(1 - (e * np.sin(phi)) ** 2)

    def t(phi):  # isometric latitude term
        sin_phi = np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * sin_phi) / (1 + e * sin_phi)) ** (e / 2)

    n = np.log(m(lat1) / m(lat2)) / np.log(t(lat1) / t(lat2))
    f = m(lat1) / (n * t(lat1) ** n)
    r0 = _A * f * t(lat0) ** n
    r = _A * f * t(lat) ** n
    theta = n * (lon - lon0)
    return LCC["false_easting"] + r * np.sin(theta), LCC["false_northing"] + r0 - r * np.cos(theta)


def europe_polygons(projection: str = "lcc") -> list[np.ndarray]:
    """Coastline of Europe as projected polygons (Natural Earth, simplified)."""
    with np.load(ASSETS / "europe_outline.npz") as data:
        return [
            np.stack(project(data[key][:, 0], data[key][:, 1], projection), axis=1)
            for key in data.files
        ]


def europe_extent(projection: str = "lcc") -> tuple[float, float, float, float]:
    """Drawn window (left, right, bottom, top) in projected metres."""
    west, east, south, north = EUROPE_BOUNDS
    lon = np.linspace(west, east, 60)
    lat = np.linspace(south, north, 60)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    x, y = project(grid_lon.ravel(), grid_lat.ravel(), projection)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def draw_land(
    ax: plt.Axes, projection: str = "lcc", *, facecolor: str = "#DCE2DD", edgecolor: str = "#7E8B84"
) -> None:
    """Paint the land as a backdrop; the sea stays the page colour."""
    for polygon in europe_polygons(projection):
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
            zorder=1,
        )


def plot_hexmap(
    points: pl.DataFrame,
    value: str | None = None,
    *,
    agg: str = "count",
    cell_km: float = 90,
    gridsize: int | None = None,
    projection: str = "lcc",
    cmap: str = style.SEQUENTIAL,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    min_count: int = 1,
    ax: plt.Axes | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Hexagon-binned map of points with `lon` and `lat` columns.

    Args:
        points: Table with `lon`, `lat` and, when `value` is given, that column.
        value: Column to aggregate; None counts the points.
        agg: "count", "sum", "mean" or "max" of `value` inside each hexagon.
        cell_km: Width of a hexagon in kilometres — the resolution knob.
            40 km gives a fine mosaic, 150 km a coarse one.
        gridsize: Number of hexagons across the map; overrides `cell_km` when given.
        projection: "lcc" (shapes look right) or "laea" (equal area).
        cmap: Any matplotlib colormap name.
        log_scale: Logarithmic colour scale, useful for counts that span decades.
        min_count: Hide hexagons holding fewer points than this.
        ax: Draw into an existing axes (for small multiples).
    """
    points = points.drop_nulls(["lon", "lat"] + ([value] if value else []))
    east, north = project(points["lon"].to_numpy(), points["lat"].to_numpy(), projection)
    extent = extent or europe_extent(projection)

    figure = ax.figure if ax is not None else plt.figure(figsize=(6.8, 7.0))
    ax = ax or figure.add_subplot(1, 1, 1)
    style.bare(ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    draw_land(ax, projection)

    cells = gridsize or max(6, int(round((extent[1] - extent[0]) / 1000 / cell_km)))
    options = {
        "gridsize": cells,
        "extent": extent,
        "cmap": cmap,
        "mincnt": min_count,
        "linewidths": 0.25,
        "edgecolors": "white",
        "zorder": 3,
    }
    if log_scale:
        options["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    else:
        options |= {"vmin": vmin, "vmax": vmax}
    if value is None:
        mesh = ax.hexbin(east, north, **options)
    else:
        reduce = {"count": len, "sum": np.sum, "mean": np.mean, "max": np.max}[agg]
        mesh = ax.hexbin(
            east, north, C=points[value].to_numpy(), reduce_C_function=reduce, **options
        )

    if title:
        ax.set_title(title, fontsize=12, pad=8)
    if colorbar:
        _colorbar(figure, mesh, ax, value, agg, colorbar_label)
    return figure, ax


def _colorbar(figure, mesh, ax, value, agg, label) -> None:
    bar = figure.colorbar(mesh, ax=ax, shrink=0.6, pad=0.015, aspect=22)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=10, length=3, color=style.INK, labelcolor=style.INK)
    bar.set_label(
        label or ("samples per cell" if value is None else f"{agg} of {value}"),
        fontsize=10.5,
        color=style.INK,
    )


def plot_hexmap_grid(
    groups: Mapping[str, pl.DataFrame],
    *,
    value: str | None = None,
    agg: str = "count",
    ncols: int = 3,
    cell_km: float = 120,
    gridsize: int | None = None,
    projection: str = "lcc",
    share_scale: bool = True,
    chips: bool = False,
    cmap: str = style.SEQUENTIAL,
    log_scale: bool = False,
    min_count: int = 1,
    title: str | None = None,
    colorbar_label: str | None = None,
    panel_size: float = 3.5,
) -> plt.Figure:
    """One small hexmap per group, all on the same colour scale.

    Use it to compare folds ("fold 0", "fold 1", …) or disturbance classes.

    Args:
        chips: Title each panel with the class chip (colour bubble and icon) instead
            of plain text. Only makes sense when the groups are model classes.
    """
    names = list(groups)
    nrows = int(np.ceil(len(names) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows * 1.12)
    )
    axes = np.atleast_1d(axes).ravel()
    extent = europe_extent(projection)
    cells = gridsize or max(6, int(round((extent[1] - extent[0]) / 1000 / cell_km)))

    top = _shared_maximum(groups, cells, extent, projection, min_count) if share_scale else None
    meshes = []
    for ax, name in zip(axes, names, strict=False):
        rows = groups[name]
        if rows.is_empty():
            style.bare(ax)
            ax.set_title(f"{name} (empty)", fontsize=style.SIZES["label"])
            continue
        _, ax = plot_hexmap(
            rows,
            value,
            agg=agg,
            gridsize=cells,
            projection=projection,
            cmap=cmap,
            log_scale=log_scale,
            vmin=1 if log_scale else 0,
            vmax=top,
            min_count=min_count,
            ax=ax,
            title=None if chips else f"{name}  ·  {rows.height}",
            colorbar=False,
            extent=extent,
        )
        meshes.append(ax.collections[-1])
        if chips:
            style.class_chip(ax, (0.02, 1.04), name, label=f"{name}  ·  {rows.height}")
    for ax in axes[len(names) :]:
        ax.set_visible(False)

    if meshes:
        bar = figure.colorbar(meshes[0], ax=axes.tolist(), shrink=0.45, pad=0.012, aspect=26)
        bar.outline.set_visible(False)
        bar.ax.tick_params(
            labelsize=style.SIZES["tick"], length=3, color=style.INK, labelcolor=style.INK
        )
        bar.set_label(
            colorbar_label or ("samples per cell" if value is None else f"{agg} of {value}"),
            fontsize=style.SIZES["label"],
            color=style.INK,
        )
    if title:
        figure.suptitle(
            title, x=0.012, ha="left", fontsize=style.SIZES["title"], fontweight="semibold"
        )
    return figure


def _shared_maximum(groups, cells, extent, projection, min_count) -> float | None:
    """Largest cell count over all groups, so every panel uses the same colour scale."""
    counts = []
    for rows in groups.values():
        if rows.is_empty():
            continue
        east, north = project(rows["lon"].to_numpy(), rows["lat"].to_numpy(), projection)
        probe = plt.figure()
        mesh = probe.add_subplot().hexbin(
            east, north, gridsize=cells, extent=extent, mincnt=min_count
        )
        values = mesh.get_array()
        counts.append(float(np.nanmax(values)) if values.size else 0.0)
        plt.close(probe)
    return max(counts) if counts else None
