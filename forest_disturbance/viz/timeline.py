"""The main figure of this project: one sample through time.

    from forest_disturbance.viz import plot_sample
    fig = plot_sample(889, root)                                   # NDVI + annotations
    fig = plot_sample(889, root, patches=("s2", "s1"), n_patches=6)
    fig = plot_sample(889, root, series=("NDVI", ("VV", "VH")))    # two panels
    fig = plot_sample(889, root, predictions=table, alerts=alerts) # what a model says

Panels, top to bottom (all optional):

    image strips     one row per modality: Sentinel-2, Sentinel-1, indices, features.
                     Each picture carries a number; the same number sits on the time
                     axis below, so you can see when it was taken.
    annotations      the expert's periods and events, plus every acquisition date.
    series           the value of the annotated pixel over time: bands, indices,
                     radar, or the components of your own features.
    model            class probabilities of a model, and the alerts they produce.
"""

from collections.abc import Sequence
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from forest_disturbance.data.constants import S1_BANDS
from forest_disturbance.viz import icons, raster, style
from forest_disturbance.viz.sample import Sample

#: Which strips can be drawn, and their accent colour.
STRIP_COLORS = {
    "s2": style.S2_COLOR,
    "s1": style.S1_COLOR,
    "index": "#2E7D32",
    "pca": style.FEATURE_COLOR,
    "features": style.FEATURE_COLOR,
}


def plot_sample(
    sample: Sample | int,
    root: str | Path | None = None,
    *,
    series: Sequence[str | tuple[str, ...]] = ("NDVI",),
    series_marks: str = "auto",
    patches: Sequence[str] = (),
    n_patches: int = 5,
    crop: int | None = 64,
    patch_dates: list[date] | None = None,
    s2_bands: tuple[str, str, str] = ("B04", "B03", "B02"),
    index_names: tuple[str, str, str] = ("NDVI", "NDMI", "NDWI"),
    features: tuple[list[date], np.ndarray] | None = None,
    feature_label: str = "features",
    annotations: bool = True,
    observations: bool | Sequence[str] = True,
    targets: pl.DataFrame | None = None,
    forget_days: float | None = None,
    predictions: pl.DataFrame | None = None,
    alerts: list[dict] | None = None,
    threshold: float = 0.5,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    span: tuple[date, date] | None = None,
) -> plt.Figure:
    """Draw one sample. Every panel is optional; see the module docstring for the layout.

    Returns:
        The figure. `figure.panels` maps a panel name ("strip:s2", "marks", "annotations",
        "series:NDVI", "model") to its axes, so you can annotate a specific panel.

    Args:
        sample: A `Sample`, or a sample id together with `root`.
        root: Dataset folder, when `sample` is an id.
        series: Panels of 1-D values at the annotated pixel. An entry is one name
            ("NDVI") or several names drawn together (("VV", "VH")). Mixed units are
            z-scored and the panel says so. A name can also be a principal component
            of all the bands of a sensor ("PC1"). Use () for no series panel.
        series_marks: How a series is drawn: "auto" (dots, plus a line when the series
            is not too dense), "dots" or "line". The same function draws every kind of
            series, so this applies to bands, indices, radar and components alike.
        patches: Image strips to draw: any of "s2", "s1", "index", "pca", "features".
        n_patches: Number of pictures per strip.
        crop: Side of the pictures in 10 m pixels (None = the full 252 px patch).
        patch_dates: Force the dates of the pictures instead of choosing them.
        s2_bands: The three bands making the Sentinel-2 colour image.
        index_names: The three indices making the "index" strip.
        features: (dates, array) of your own features: (time, channel) for the pixel
            series, or (time, channel, height, width) to also allow a "features" strip.
            More than three channels are reduced to three by PCA.
        feature_label: Row label for the feature strip and panel.
        annotations: Draw the expert periods and events.
        targets: The examples a `DisforDataset` built (`dataset.examples`). Draws what
            the model is actually asked to answer on every acquisition: its target
            class, and how long ago the event was.
        forget_days: Marks the age at which the loss stops asking for the true class,
            on the `targets` panel.
        observations: Draw a tick per acquisition under the annotation lane: True for
            both sensors, or the sensors to show, e.g. ("s2",).
        predictions: Prediction table of a run (runs/<run>/predictions/epoch_XXX.parquet).
        alerts: Operational alerts, e.g. `evaluate_operational(...)["alerts"]`.
        threshold: Alert threshold drawn as a dotted line in the model panel.
        title: Figure title; by default a one-line description of the sample.
        span: Restrict the time axis, e.g. to zoom on one event.
    """
    if not isinstance(sample, Sample):
        if root is None:
            raise ValueError("pass a Sample, or a sample id together with root=")
        sample = Sample.load(int(sample), root)

    panels = _panel_plan(series, patches, annotations, predictions, features, targets)
    events = [row for row in sample.labels.iter_rows(named=True) if _is_disturbance(row)]
    figure, axes = _build_axes(panels, figsize, n_patches, len(events))
    time_axes = [ax for kind, ax in zip(panels, axes, strict=True) if not kind.startswith("strip:")]

    start, end = span or sample.window
    for ax in time_axes:
        ax.set_xlim(start, end)

    marks: list[list[tuple[int, date, str, str]]] = []  # one list per strip
    for kind, ax in zip(panels, axes, strict=True):
        if kind == "marks":
            continue
        if kind.startswith("strip:"):
            marks.append(
                _draw_strip(
                    figure,
                    ax,
                    sample,
                    kind.split(":")[1],
                    n=n_patches,
                    crop=crop,
                    dates=patch_dates,
                    s2_bands=s2_bands,
                    index_names=index_names,
                    features=features,
                    feature_label=feature_label,
                )
            )
        elif kind == "annotations":
            _draw_annotations(
                ax,
                sample,
                observations=observations,
                height_points=annotation_height(len(events)) * 72,
            )
        elif kind.startswith("series:"):
            _draw_series(ax, sample, _names(kind.split(":", 1)[1]), marks=series_marks)
        elif kind == "features":
            _draw_feature_series(ax, features, feature_label)
        elif kind == "targets":
            _draw_targets(ax, sample, targets, forget_days)
        elif kind == "model":
            _draw_model(ax, sample, predictions, alerts, threshold)

    _mark_events(
        [
            ax
            for kind, ax in zip(panels, axes, strict=True)
            if kind.startswith(("series", "model", "features", "targets"))
        ],
        sample,
    )
    if "marks" in panels:
        _draw_marks(axes[panels.index("marks")], marks)
    _format_time_axis(time_axes)

    caption = title or sample.caption()
    if any(kind.startswith("strip:") for kind in panels):
        # Say how much ground a picture covers: the patches are 252 px (2.52 km) and
        # everything below that is a centre crop, which is easy to mistake for the whole thing.
        side = 252 if crop is None else crop
        caption += f"  ·  pictures {side} x {side} px  =  {side * 10 / 1000:g} km across"
    figure.suptitle(
        caption, x=0.012, ha="left", fontsize=style.SIZES["title"], fontweight="semibold"
    )
    figure.align_ylabels(time_axes)
    # Name every panel so callers can annotate one: figure.panels["series:NDVI"], ...
    figure.panels = dict(zip(panels, axes, strict=True))
    figure.sample = sample
    return figure


# --- layout ------------------------------------------------------------------------------


def _names(spec: str) -> tuple[str, ...]:
    return tuple(part for part in spec.split(",") if part)


def _panel_plan(series, patches, annotations, predictions, features, targets) -> list[str]:
    plan = [f"strip:{mode}" for mode in patches]
    if plan:
        plan.append("marks")
    if annotations:
        plan.append("annotations")
    if targets is not None:
        plan.append("targets")
    for entry in series:
        names = (entry,) if isinstance(entry, str) else tuple(entry)
        plan.append("series:" + ",".join(names))
    if features is not None:
        plan.append("features")
    if predictions is not None:
        plan.append("model")
    if not plan:
        raise ValueError("nothing to draw: give series, patches, targets or predictions")
    return plan


#: Height in inches of the panels whose size is a matter of taste rather than of
#: content (a picture strip is sized so the pictures stay square, and the ruler and
#: the annotation lane are derived from `style.SIZES`). Edit to make series taller.
PANEL_HEIGHTS = {"series": 1.7, "features": 1.7, "model": 1.8, "targets": 1.7}
_WIDTH, _LEFT, _RIGHT, _TOP_PAD, _BOTTOM_PAD = 13.5, 0.085, 0.99, 0.62, 0.62

#: Blank space kept above each kind of panel, in inches. A picture strip needs room
#: for its date titles; the ruler sits right under the pictures it labels, and the
#: annotation lane right under the ruler, so those two get almost none.
_GAPS = {"strip": 0.44, "marks": 0.08, "annotations": 0.10}
_DEFAULT_GAP = 0.34


#: Diameter of a numbered bubble, in points (defined in `style`, re-exported here
#: because the layout maths below is full of it).
bubble_size = style.bubble_size


def marks_height(n_rows: int) -> float:
    """Height of the ruler panel, in inches: one row per picture strip."""
    return n_rows * bubble_size() * 1.75 / 72


def _stems() -> tuple[float, float]:
    """The two heights a marker stem can have, in points.

    The second one clears the first one's bubble *and* its label, so a marker that
    has to move up really does escape the one below it.
    """
    size = style.SIZES
    marker = max(bubble_size(), size["event_icon"] * 1.5)
    low = size["event_stem"]
    return low, low + marker + size["label"] * 1.6 + 6


def annotation_height(n_events: int = 1) -> float:
    """Height of the annotation lane, in inches, derived from the text sizes.

    Computed rather than fixed so the lane is exactly as tall as what it holds:
    the acquisition rows, the period ribbon, the event stems (twice as tall when
    several events have to be staggered), the icons and their labels.
    """
    size = style.SIZES
    points = (
        2 * (size["tick_height"] + 5)  # the two acquisition rows
        + 3
        + size["label"] * 1.7  # the period ribbon
        + _stems()[1 if n_events > 1 else 0]  # the tallest stem
        + max(bubble_size(), size["event_icon"] * 1.5)  # the bubble on top of it
        + size["label"] * 1.5  # the event label above the icon
    )
    return points / 72


def _panel_height(kind: str, *, n_events: int, n_rows: int, picture: float) -> float:
    """Height of one panel, in inches."""
    if kind == "annotations":
        return annotation_height(n_events)
    if kind == "marks":
        return marks_height(n_rows)
    if kind.startswith("strip:"):
        return min(picture + 0.34, 2.9)
    return PANEL_HEIGHTS[kind.split(":")[0]]


def _build_axes(
    panels: list[str], figsize, n_patches: int, n_events: int
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Stack the panels from top to bottom, each with the blank space it needs.

    The panels are placed at absolute positions rather than through a gridspec, so
    that a tall picture strip does not push a big gap under the ruler as well.
    """
    picture = (_WIDTH * (_RIGHT - _LEFT) - 1.1) / max(n_patches, 1)  # one picture, inches
    n_rows = sum(1 for kind in panels if kind.startswith("strip:"))
    heights = [
        _panel_height(kind, n_events=n_events, n_rows=n_rows, picture=picture) for kind in panels
    ]
    gaps = [_GAPS.get(kind.split(":")[0], _DEFAULT_GAP) for kind in panels]

    width = figsize[0] if figsize else _WIDTH
    height = figsize[1] if figsize else sum(heights) + sum(gaps[1:]) + _TOP_PAD + _BOTTOM_PAD
    figure = plt.figure(figsize=(width, height))

    axes: list[plt.Axes] = []
    shared = None
    top = height - _TOP_PAD
    for kind, panel_height, gap in zip(panels, heights, gaps, strict=True):
        top -= gap if axes else 0.0
        rect = (_LEFT, (top - panel_height) / height, _RIGHT - _LEFT, panel_height / height)
        if kind.startswith("strip:"):
            axes.append(style.bare(figure.add_axes(rect)))
        else:
            ax = figure.add_axes(rect, sharex=shared)
            shared = shared or ax
            axes.append(ax)
        top -= panel_height
    return figure, axes


def _format_time_axis(time_axes: list[plt.Axes]) -> None:
    for ax in time_axes[:-1]:
        ax.tick_params(labelbottom=False)
    last = time_axes[-1]
    last.xaxis.set_major_locator(mdates.YearLocator())
    last.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    last.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))


# --- panels ------------------------------------------------------------------------------


def _draw_annotations(
    ax: plt.Axes, sample: Sample, *, observations: bool | Sequence[str], height_points: float
) -> None:
    """The expert's reading of this pixel: periods as ribbons, events as glyphs.

    Everything is placed in points inside the axes (`ylim` is the axes height in
    points), so sizes taken from `style.SIZES` land exactly where they are meant to
    and nothing can spill into the panel above or below.
    """
    size = style.SIZES
    ax.set_ylim(0, height_points)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    tick_row_s1 = 0.0
    tick_row_s2 = size["tick_height"] + 5
    ribbon_bottom = 2 * size["tick_height"] + 13
    ribbon_height = size["label"] * 1.7
    ribbon_top = ribbon_bottom + ribbon_height

    # A bubble matches the ruler bubbles, unless a bigger icon has been asked for.
    marker = max(bubble_size(), size["event_icon"] * 1.5)
    lo, hi = ax.get_xlim()
    rows = [row for row in sample.labels.iter_rows(named=True) if _benchmark_name(row["label"])]
    periods = [row for row in rows if not _is_event(row)]
    # Everything the model has to report gets a marker: a date for a sudden event,
    # a stretch of the ribbon for a slow one such as a beetle outbreak.
    marked = [row for row in rows if _is_disturbance(row)]

    # The corners are rounded in both directions at once, which takes care: this axes
    # measures x in days and y in points, so the radius is given in days and the
    # patch is told how to stretch it back (`mutation_aspect`).
    scale = per_point(ax)
    radius = ribbon_height * min(size["ribbon_rounding"], 0.5)
    for row in periods:
        end_day = max(row["end_evidence"], row["start"])
        color = style.class_color(_benchmark_name(row["label"]) or "No Disturbance")
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (mdates.date2num(row["start"]), ribbon_bottom),
                mdates.date2num(end_day) - mdates.date2num(row["start"]),
                ribbon_height,
                boxstyle=f"round,pad=0,rounding_size={radius * scale}",
                mutation_aspect=1 / scale,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        # Centre the name on the part of the ribbon that is actually on screen, and
        # leave it out when the ribbon is too narrow to hold it.
        name = _benchmark_name(row["label"])
        left = max(mdates.date2num(row["start"]), lo)
        right = min(mdates.date2num(end_day), hi)
        if right - left > len(name) * size["label"] * 0.62 * per_point(ax):
            ax.text(
                (left + right) / 2,
                ribbon_bottom + ribbon_height / 2,
                name,
                ha="center",
                va="center",
                fontsize=size["label"],
                color="white",
                fontweight="semibold",
                zorder=3,
            )

    visible = [row for row in marked if lo <= mdates.date2num(row["start"]) <= hi]

    # Work out where every marker goes before drawing any of it: a stem rising to the
    # upper row would otherwise be drawn straight through the label of the marker
    # beside it, which no amount of z-order makes readable.
    placed = []
    previous_right = None  # right edge of the last label, in days
    for row in visible:
        name = _class_of(sample, row["label"])
        start_day, end_day = row["start"], max(row["end_evidence"], row["start"])
        lasts = (end_day - start_day).days > 0
        text = style.event_label(name, start_day)
        if lasts:
            text += f" \u2192 {end_day}  ({(end_day - start_day).days} d)"
        anchor = mdates.date2num(start_day + (end_day - start_day) / 2 if lasts else start_day)
        half = len(text) * size["label"] * 0.55 / 2 * scale
        raised = previous_right is not None and anchor - half < previous_right
        previous_right = max(anchor + half, previous_right or anchor + half)
        placed.append(
            {
                "name": name,
                "anchor": anchor,
                "text": text,
                "half": half,
                "raised": raised,
                "lasts": lasts,
            }
        )

    gap = size["event_rule"] * 1.5 * scale
    for item in placed:
        # Slide a label sideways when someone else's stem would cross it.
        crossing = [
            other["anchor"]
            for other in placed
            if other is not item
            and other["raised"] != item["raised"]
            and abs(other["anchor"] - item["anchor"]) < item["half"]
        ]
        alignment, label_x = "center", item["anchor"]
        if crossing:
            nearest = min(crossing, key=lambda x: abs(x - item["anchor"]))
            if nearest > item["anchor"]:
                alignment, label_x = "right", nearest - gap
            else:
                alignment, label_x = "left", nearest + gap
        item.update(align=alignment, label_x=label_x)

    for item in placed:
        color = style.class_color(item["name"])
        icon_y = ribbon_top + _stems()[1 if item["raised"] else 0]
        # The stem starts at the top of the ribbon, not inside it: the ribbon carries
        # its own name and a line through it is hard to read.
        ax.plot(
            [item["anchor"], item["anchor"]],
            [ribbon_top, icon_y],
            color=color,
            linewidth=size["event_rule"],
            solid_capstyle="round",
            zorder=4,
        )
        # The icon sits in a white bubble, as wide as the ruler bubbles below the
        # pictures, so a colour emoji stays readable over the stem and the ribbon.
        ax.scatter(
            [item["anchor"]],
            [icon_y],
            s=marker**2,
            facecolor="white",
            edgecolors=color,
            linewidths=size["event_rule"],
            zorder=5,
        )
        icons.draw_icon(
            ax, (item["anchor"], icon_y), item["name"], size=size["event_icon"], color=color
        )
        label = ax.text(
            item["label_x"],
            icon_y + marker * 0.62,
            item["text"],
            ha=item["align"],
            va="bottom",
            fontsize=size["label"],
            fontfamily=style.ICON_FONT if "icon" in style.EVENT_LABEL else None,
            color=color,
            fontweight="semibold",
            zorder=6,
            clip_on=True,
        )
        style.outlined(label, width=3.5)

    if observations:
        shown = ("s2", "s1") if observations is True else tuple(observations)
        rows = (("s2", tick_row_s2, style.S2_COLOR), ("s1", tick_row_s1, style.S1_COLOR))
        for sensor, y, color in (row for row in rows if row[0] in shown):
            dates = sample.dates(sensor)
            if not dates:
                continue
            ax.vlines(
                dates,
                y,
                y + size["tick_height"],
                color=color,
                linewidth=size["tick_width"],
                alpha=0.9,
                zorder=2,
            )
            ax.annotate(
                f"{sensor.upper()}  {len(dates)}",
                xy=(0, y + size["tick_height"] / 2),
                xycoords=("axes fraction", "data"),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=size["label"],
                fontweight="semibold",
                color=color,
            )


def per_point(ax: plt.Axes) -> float:
    """Days of the time axis per point on the page — the scale of the figure.

    Lets a panel keep a gap, or check that a name fits, in page units rather than in
    days, so it looks the same whether the figure spans one year or ten.
    """
    x0, x1 = ax.get_xlim()
    width = ax.get_position().width * ax.figure.get_size_inches()[0] * 72
    return (x1 - x0) / max(width, 1)


def _is_event(row: dict) -> bool:
    """A disturbance that happened on one date: a cut, a storm, a fire.

    Not every disturbance is one. A bark beetle outbreak is annotated as a period
    that lasts (median 458 days over the dataset), and `is_event` is false for all
    of them, so it has to be drawn as a stretch of time rather than as a date.
    """
    return bool(row["is_event"])


def _is_disturbance(row: dict) -> bool:
    """True when the benchmark asks the model to report this row."""
    name = _benchmark_name(row["label"])
    return name is not None and name != "No Disturbance"


def _mark_events(axes: list[plt.Axes], sample: Sample) -> None:
    """Mark every disturbance on the other panels, so they can be read against each other.

    A sudden event is a thin rule; one that lasts, such as a beetle outbreak, is a
    shaded band over the days it covers.
    """
    marks = []
    for row in sample.labels.iter_rows(named=True):
        if not _is_disturbance(row):
            continue
        end_day = max(row["end_evidence"], row["start"])
        marks.append((row["start"], end_day, style.class_color(_class_of(sample, row["label"]))))
    for ax in axes:
        for start, end_day, color in marks:
            if (end_day - start).days > 0:
                ax.axvspan(start, end_day, color=color, alpha=0.10, linewidth=0, zorder=1)
            ax.axvline(
                start, color=style.MUTED, linewidth=0.7, linestyle=style.DASHED, alpha=0.5, zorder=1
            )


@lru_cache(maxsize=1)
def _default_mapping():
    """The project's label mapping, so event colours match the model classes."""
    from forest_disturbance.build import REPO_ROOT
    from forest_disturbance.data.labels import LabelMapping

    return LabelMapping.from_yaml(REPO_ROOT / "configs/label_mapping.yaml")


def _class_of(sample: Sample, code: int) -> str:
    """Model class of a label code ("Wind", "Clear Cut", ...)."""
    mapping = _default_mapping()
    class_id = mapping.code_to_class.get(int(code))
    return (
        mapping.class_names.get(class_id, "No Disturbance")
        if class_id is not None
        else "No Disturbance"
    )


def _benchmark_name(code: int) -> str | None:
    """The name the benchmark gives a raw label code, or None when it ignores it.

    The raw codes are finer than the classes the model predicts: "Bark beetle" and
    "Gypsy moth" are both Biotic, "Undisturbed Forest" and "Revegetation" are both
    No Disturbance, and a few codes ("Drought", "Forestry Mulching") are dropped
    altogether. The annotation lane shows what the benchmark uses, not the raw code,
    so a figure never promises a class the model is not asked for.
    """
    mapping = _default_mapping()
    code = int(code)
    if code in mapping.ignored_codes or code in getattr(mapping, "filtered_codes", ()):
        return None
    class_id = mapping.code_to_class.get(code)
    if class_id is not None:
        return mapping.class_names[class_id]
    return mapping.class_names[mapping.no_disturbance_id]


def _draw_series(ax: plt.Axes, sample: Sample, names: tuple[str, ...], marks: str = "auto") -> None:
    """One panel with one or several 1-D series of the annotated pixel.

    This is the only series drawing code: a Sentinel-2 band, a spectral index, a
    radar polarisation and a feature component all come through here. Give several
    names to draw them together (they are z-scored when their units differ), or one
    name per panel to keep them apart.

    Args:
        marks: "auto" draws dots and a line, and drops the line when the series has
            more than 600 points (radar); "dots" and "line" force one of the two.
    """
    units = {
        ("dB" if name in S1_BANDS else "index" if name in raster.INDICES else "reflectance")
        for name in names
    }
    normalize = len(units) > 1
    colors = _series_colors(names)
    for name in names:
        dates, values = sample.series(name)
        if not len(dates):
            continue
        shown = _zscore(values) if normalize else values
        dense = marks == "dots" or (marks == "auto" and len(dates) > 600)
        if not dense:
            ax.plot(
                dates,
                shown,
                "-",
                color=colors[name],
                linewidth=style.SIZES["line"],
                alpha=0.6,
                zorder=2,
            )
        ax.plot(
            dates,
            shown,
            ".",
            color=colors[name],
            markersize=style.SIZES["dot"],
            alpha=0.8 if dense else 1.0,
            zorder=3,
            label=name,
        )
    unit = "z-scored" if normalize else units.pop()
    ax.set_ylabel(names[0] if len(names) == 1 else unit)
    # Few horizontal lines: the panel is short and the grid should not compete with
    # the data. Matplotlib's default puts one every few hundredths of an index.
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    if len(names) > 1:
        handles = [
            plt.Line2D([], [], color=colors[name], linewidth=2.8, label=name) for name in names
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            ncol=min(len(names), 4),
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=1.1,
            borderaxespad=0.2,
        )


def _series_colors(names: Sequence[str]) -> dict[str, str]:
    base = {
        "NDVI": "#2E7D32",
        "NDMI": "#0E7C86",
        "NDWI": "#3B6FB6",
        "NBR": "#B5651D",
        "NDRE": "#7A5195",
        "VV": style.S1_COLOR,
        "VH": "#8C5A9E",
    }
    cmap = plt.get_cmap("magma")
    extra = [cmap(v) for v in np.linspace(0.15, 0.8, max(len(names), 1))]
    return {name: base.get(name, extra[i]) for i, name in enumerate(names)}


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    return (values - finite.mean()) / max(finite.std(), 1e-6)


def _draw_feature_series(ax: plt.Axes, features, label: str) -> None:
    """Three strongest PCA components of user-supplied features, over time."""
    dates, array = features
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 4:  # (time, channel, height, width) -> centre pixel
        row, col = raster.center_index(values)
        values = values[:, :, row, col]
    if values.shape[1] > 3:
        pca = raster.fit_pca(values)
        values = raster.apply_pca(values, pca)
        names = [f"PC{i + 1} ({pca['explained'][i]:.0%})" for i in range(values.shape[1])]
    else:
        names = [f"channel {i + 1}" for i in range(values.shape[1])]
    for i, name in enumerate(names):
        color = plt.get_cmap("magma")(0.2 + 0.3 * i)
        ax.plot(dates, values[:, i], "-", color=color, linewidth=0.8, alpha=0.6)
        ax.plot(dates, values[:, i], ".", color=color, markersize=3.2, label=name)
    ax.set_ylabel(label)
    ax.legend(
        loc="upper left",
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
        markerscale=2.2,
        borderaxespad=0.2,
    )


def _draw_targets(
    ax: plt.Axes, sample: Sample, targets: pl.DataFrame, forget_days: float | None
) -> None:
    """What the model is asked to answer on every acquisition.

    One dot per training example, at the height of its target class and in that
    class's colour, over a grey line saying how long ago the event was. Together
    they show what the dataset turns the expert's periods into: a class that the
    loss stops asking for once the event is old enough to have faded from the images.
    """
    rows = targets.filter(pl.col("sample_id") == sample.sample_id).sort("date")
    if rows.is_empty():
        style.callout(ax, "no training examples for this sample", xytext=(0.5, 0.5))
        return
    class_names = _default_mapping().class_names
    names = [class_names.get(int(c), "No Disturbance") for c in rows["label"]]
    order = list(dict.fromkeys(["No Disturbance", *style.CLASS_COLORS]))
    levels = [name for name in order if name in set(names)]

    age = ax.twinx()
    ages = rows["days_since_event"].to_numpy().astype(float)
    days = np.where(np.isfinite(ages) & (ages >= 0), ages, np.nan)
    age.plot(
        rows["date"], days, "-", color=style.FAINT, linewidth=style.SIZES["line"] * 1.4, zorder=1
    )
    age.set_ylabel("days since the event", color=style.MUTED)
    age.tick_params(axis="y", labelcolor=style.MUTED)
    age.grid(False)
    if forget_days is not None:
        # Keep the scale around the age that matters: an event from five years ago
        # would otherwise flatten the whole line against the bottom.
        age.set_ylim(0, forget_days * 2.2)
        age.axhline(forget_days, color=style.MUTED, linewidth=1.4, linestyle=style.DASHED, zorder=2)
        age.annotate(
            f"the loss forgets after {forget_days:.0f} days",
            xy=(0.995, forget_days),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=style.SIZES["label"],
            color=style.MUTED,
        )

    ax.scatter(
        rows["date"],
        names,
        s=style.SIZES["dot"] ** 2,
        c=[style.class_color(name) for name in names],
        zorder=3,
    )
    ax.set_yticks(levels)
    ax.set_ylabel("target class")
    ax.set_zorder(age.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.grid(axis="y", visible=False)


def _draw_model(
    ax: plt.Axes, sample: Sample, predictions: pl.DataFrame, alerts, threshold: float
) -> None:
    """Class probabilities of a model, and the alerts the operational filter emits."""
    from forest_disturbance.metrics.events import probability_column

    rows = predictions.filter(pl.col("sample_id") == sample.sample_id).sort("date")
    if rows.is_empty():
        ax.text(
            0.5,
            0.5,
            "no predictions for this sample",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=style.MUTED,
        )
        ax.set_ylabel("p(class)")
        return
    drawn = []
    for name, color in style.CLASS_COLORS.items():
        column = probability_column(name)
        if name != "No Disturbance" and column in rows.columns:
            ax.plot(rows["date"], rows[column], "-", color=color, linewidth=style.SIZES["line"])
            drawn.append(name)
    ax.axhline(threshold, color=style.MUTED, linewidth=1.6, linestyle=style.DASHED, zorder=1)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("p(class)")
    style.class_legend(ax, drawn, xy=(0.005, 0.90))

    for alert in alerts or []:
        if alert["sample_id"] != sample.sample_id:
            continue
        matched = alert["matched_event_id"] is not None
        ax.scatter(
            [date.fromisoformat(alert["date"])],
            [1.0],
            marker="v",
            s=110,
            zorder=6,
            clip_on=False,
            color=style.class_color(alert["predicted_class"] or ""),
            edgecolors="white" if matched else style.INK,
            linewidths=1.1,
        )


# --- image strips -------------------------------------------------------------------------


def _draw_strip(
    figure,
    ax,
    sample: Sample,
    mode: str,
    *,
    n,
    crop,
    dates,
    s2_bands,
    index_names,
    features,
    feature_label,
) -> list[tuple[int, date, str]]:
    """One row of pictures. Returns (number, date, colour) for the timeline marks."""
    color = STRIP_COLORS.get(mode, style.FEATURE_COLOR)
    if mode == "features":
        used, images, label = _feature_images(features, n)
    else:
        sensor = "s1" if mode == "s1" else "s2"
        used, cube = sample.patches(sensor, dates=dates, n=n, crop=crop)
        if not len(used):
            ax.text(
                0.5,
                0.5,
                f"no {sensor.upper()} patches found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=style.MUTED,
                fontsize=9,
            )
            return []
        if mode == "s2":
            images, label = raster.s2_rgb(cube, s2_bands), " ".join(s2_bands)
        elif mode == "s1":
            images, label = raster.s1_rgb(cube), "VV · VH · VV-VH"
        elif mode == "index":
            images, label = raster.index_rgb(cube, index_names), " · ".join(index_names)
        elif mode == "pca":
            images, _ = raster.pca_rgb(cube)
            label = "PCA of the bands"
        else:
            raise ValueError(f"unknown strip {mode!r}: use s2, s1, index, pca or features")

    marks = []
    for i, (day, image) in enumerate(zip(used, images, strict=True), start=1):
        cell = style.bare(figure.add_axes(_cell(ax, i - 1, len(used))))
        cell.imshow(np.clip(image, 0, 1))
        row, col = raster.center_index(image[..., 0])
        cell.add_patch(
            mpatches.Circle(
                (col, row),
                radius=max(image.shape[0] * 0.04, 2.2),
                fill=False,
                edgecolor="white",
                linewidth=1.8,
            )
        )
        offset = _offset_label(sample, day)
        cell.set_title(
            f"{day}{offset}", fontsize=style.SIZES["label"], color=style.INK, pad=5, loc="center"
        )
        style.badge(cell, i, color)
        marks.append((i, day, color, mode.upper()))
    ax.annotate(
        f"{mode.upper()}\n{label}",
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=style.SIZES["label"],
        color=style.MUTED,
    )
    ax.set_frame_on(False)
    return marks


def _cell(ax: plt.Axes, index: int, total: int, wspace: float = 0.03) -> tuple:
    """Position of one picture inside a strip panel, as a figure rectangle."""
    box = ax.get_position()
    step = box.width / total
    return (box.x0 + index * step + step * wspace / 2, box.y0, step * (1 - wspace), box.height)


def _offset_label(sample: Sample, day: date) -> str:
    """How far a picture is from the first disturbance, e.g. "  (-1 y)"."""
    events = sample.events()
    if events.is_empty():
        return ""
    days = (day - events["start"][0]).days
    if abs(days) < 45:
        return f"  ({days:+d} d)"
    if abs(days) < 400:
        return f"  ({days / 30.44:+.0f} mo)"
    return f"  ({days / 365.25:+.1f} y)"


def _feature_images(features, n: int) -> tuple[list[date], np.ndarray, str]:
    """Pictures of user-supplied feature maps: PCA of the whole series, three components."""
    if features is None:
        raise ValueError("patches=('features',) needs features=(dates, array)")
    dates, array = features
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("a feature strip needs features shaped (time, channel, height, width)")
    keep = np.linspace(0, len(dates) - 1, min(n, len(dates))).round().astype(int)
    images, _ = raster.pca_rgb(values[keep])
    return [dates[int(i)] for i in keep], images, "PCA of your features"


def _draw_marks(ax: plt.Axes, marks: list[list[tuple[int, date, str, str]]]) -> None:
    """The ruler tying every picture to its date: same number, same colour.

    Two pictures a few days apart would land on the same spot, so the bubbles are
    spread apart just enough not to touch and a leader line points back at the tick
    marking the real date.
    """
    style.bare(ax)
    rows = [row for row in marks if row]
    diameter = bubble_size()
    row_height = diameter * 1.75
    height = max(len(rows), 1) * row_height
    ax.set_ylim(0, height)
    ax.set_autoscaley_on(False)

    scale = per_point(ax)  # days of the axis per point on the page

    for index, row in enumerate(rows):
        color = row[0][2]
        top = height - index * row_height
        rule = top - diameter * 1.35
        centre = top - diameter * 0.55
        ax.axhline(rule, color=color, alpha=0.35, linewidth=style.SIZES["line"] * 1.4, zorder=1)
        ax.annotate(
            row[0][3],
            xy=(0, rule),
            xycoords=("axes fraction", "data"),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style.SIZES["label"],
            fontweight="semibold",
            color=color,
        )
        for (number, day, _, _), x in zip(row, _spread(row, scale, diameter), strict=True):
            true_x = mdates.date2num(day)
            ax.vlines(true_x, rule - diameter * 0.2, rule + diameter * 0.2, color=color, zorder=2)
            ax.plot(
                [true_x, x],
                [rule + diameter * 0.2, centre - diameter * 0.5],
                color=color,
                linewidth=style.SIZES["line"] * 0.8,
                zorder=2,
            )
            ax.scatter(
                [x],
                [centre],
                s=diameter**2,
                color=color,
                edgecolors="white",
                linewidths=1.6,
                zorder=3,
            )
            ax.text(
                x,
                centre,
                str(number),
                ha="center",
                va="center",
                fontsize=style.SIZES["label"],
                color="white",
                fontweight="bold",
                zorder=4,
            )


def _spread(row, scale: float, diameter: float) -> list[float]:
    """Positions for one row of bubbles: the true dates, pushed apart where they touch.

    `scale` converts points on the page into days, so the gap kept between two
    bubbles is a gap on the page whatever the span of the figure.
    """
    gap = diameter * 1.1 * scale
    true = [mdates.date2num(day) for _, day, _, _ in row]
    spread = list(true)
    first = 0
    while first < len(spread):
        last = first
        while last + 1 < len(spread) and true[last + 1] - spread[last] < gap:
            spread[last + 1] = spread[last] + gap
            last += 1
        if last > first:  # a cluster was pushed apart: centre it back on its own dates
            shift = (true[first] + true[last] - spread[first] - spread[last]) / 2
            spread[first : last + 1] = [x + shift for x in spread[first : last + 1]]
        first = last + 1
    return spread
