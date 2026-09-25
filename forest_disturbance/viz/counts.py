"""How many annotations of each kind, and how they spread over the folds.

fig = plot_code_counts(labels, mapping)             # every label code
fig = plot_class_counts(labels, mapping)            # the 6 model classes
fig = plot_class_counts(labels, mapping, splits)    # ... stacked per fold
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from forest_disturbance.data.constants import LABEL_NAMES
from forest_disturbance.data.labels import LabelMapping
from forest_disturbance.viz import style


def plot_code_counts(
    labels: pl.DataFrame,
    mapping: LabelMapping,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """All label codes of the dataset, grouped by what the benchmark makes of them.

    The raw codes are finer than the classes the model predicts. The bars are
    therefore stacked under the class they are merged into, so it is visible at a
    glance that "Bark beetle" and "Gypsy moth" both become Biotic, and which codes
    are dropped before training.
    """
    counts = dict(zip(labels["label"].to_list(), [0] * labels.height, strict=False))
    for code, n in labels.group_by("label").len().iter_rows():
        counts[int(code)] = int(n)

    groups = _code_groups(counts, mapping)
    figure = ax.figure if ax is not None else plt.figure(figsize=(11.5, 7.4))
    ax = ax or figure.add_subplot(1, 1, 1)

    rows, ticks, labels_, colors, headers = [], [], [], [], []
    y = 0.0
    for name, color, codes in groups:
        headers.append((y, name, color))
        y += 1.0
        for code in codes:
            rows.append(y)
            ticks.append(y)
            labels_.append(f"{code}  {LABEL_NAMES.get(code, code)}")
            colors.append(color)
            y += 1.0
        y += 0.5

    values = np.array([counts[c] for _, _, codes in groups for c in codes], dtype=float)
    ax.barh(rows, values, color=colors, edgecolor="white", linewidth=0.6, height=0.74)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_, fontsize=style.SIZES["label"], color=style.INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, values.max() * 3.2)
    ax.set_ylim(y - 0.5, -1.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for position, value in zip(rows, values, strict=True):
        ax.text(
            value * 1.12,
            position,
            str(int(value)),
            va="center",
            fontsize=style.SIZES["label"],
            color=style.INK,
        )
    for position, name, color in headers:
        known = name in style.CLASS_COLORS
        style.class_chip(
            ax,
            (0.012, position),
            name if known else "No Disturbance",
            xycoords=("axes fraction", "data"),
            label=f"→ {name}",
            color=color,
            icon=None if known else "\u00d7",  # Lato has no U+2715
        )
    ax.set_xlabel("annotated periods (log scale)")
    style.title(
        ax,
        title or "Every label code, and the class it becomes",
        "the model only ever predicts the classes on the left",
    )
    return figure


def _code_groups(counts: dict[int, int], mapping: LabelMapping) -> list[tuple[str, str, list[int]]]:
    """(class name, colour, codes) in the order they should be drawn."""
    seen = sorted(counts)
    groups = []
    for class_id in sorted(set(mapping.code_to_class.values())):
        name = mapping.class_names[class_id]
        codes = [c for c in seen if mapping.code_to_class.get(c) == class_id]
        if codes:
            groups.append((name, style.class_color(name), codes))
    healthy = [
        c
        for c in seen
        if c not in mapping.code_to_class
        and c not in mapping.ignored_codes
        and c not in mapping.filtered_codes
    ]
    if healthy:
        name = mapping.class_names[mapping.no_disturbance_id]
        groups.append((name, style.class_color(name), healthy))
    dropped = [c for c in seen if c in mapping.ignored_codes or c in mapping.filtered_codes]
    if dropped:
        groups.append(("ignored by the benchmark", "#B9BFC4", dropped))
    return groups


def plot_class_counts(
    labels: pl.DataFrame,
    mapping: LabelMapping,
    splits: pl.DataFrame | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Annotations per model class. With `splits`, each bar is split by validation fold."""
    events = labels.filter(pl.col("label").is_in(list(mapping.code_to_class))).with_columns(
        pl.col("label").replace_strict(mapping.code_to_class).alias("class_id")
    )
    order = [i for i in sorted(mapping.class_names) if i != mapping.no_disturbance_id]
    names = [mapping.class_names[i] for i in order]
    figure = ax.figure if ax is not None else plt.figure(figsize=(11.5, 5.2))
    ax = ax or figure.add_subplot(1, 1, 1)

    if splits is None:
        totals = np.array([events.filter(pl.col("class_id") == i).height for i in order])
        _barh(ax, names, totals, [style.class_color(n) for n in names])
        _chip_ticks(ax, names)
        ax.set_xlabel("annotated disturbances")
        style.title(ax, title or "Disturbances per class")
        return figure

    centres: list[tuple[float, int, int, str, float]] = []
    validation = splits.filter(pl.col("split") == "val").select("sample_id", "fold_id")
    per_fold = events.join(validation, on="sample_id").group_by("class_id", "fold_id").len()
    folds = sorted(validation["fold_id"].unique().to_list())
    left = np.zeros(len(order))
    shades = np.linspace(0.45, 1.0, len(folds))
    widest = 0.0
    for fold, shade in zip(folds, shades, strict=True):
        values = np.array(
            [
                per_fold.filter((pl.col("class_id") == i) & (pl.col("fold_id") == fold))[
                    "len"
                ].sum()
                for i in order
            ],
            dtype=float,
        )
        colors = [_lighten(style.class_color(name), 1 - shade) for name in names]
        ax.barh(
            names,
            values,
            left=left,
            color=colors,
            edgecolor="white",
            linewidth=1.0,
            label=f"fold {fold}",
            height=0.7,
        )
        # A bubble with the fold number inside each segment: the shades alone are
        # hard to tell apart. Segments too narrow to hold one are left plain.
        for y, (start, value) in enumerate(zip(left, values, strict=True)):
            if value > 0:
                centres.append((start + value / 2, y, fold, style.class_color(names[y]), value))
        left += values
        widest = max(widest, float(left.max()))
    ax.set_xlim(0, widest * 1.10)
    for y, total in enumerate(left):
        ax.text(
            total + widest * 0.012,
            y,
            str(int(total)),
            va="center",
            fontsize=style.SIZES["label"],
            color=style.INK,
        )
    # The bubble needs room for itself: about that many data units wide.
    room = widest / max(ax.get_position().width * figure.get_size_inches()[0] * 72, 1)
    for x, y, fold, color, value in centres:
        if value > style.bubble_size() * room * 1.25:
            style.bubble(ax, x, y, fold, color, filled=False)
    ax.invert_yaxis()
    _chip_ticks(ax, names)
    ax.set_xlabel("annotated disturbances")
    ax.grid(axis="y", visible=False)
    style.title(
        ax,
        title or "Disturbances per class",
        f"each bar is split into the {len(folds)} validation folds, "
        "numbered from fold 0 on the left; the shade says the same thing",
    )
    return figure


def _chip_ticks(ax: plt.Axes, names: list[str]) -> None:
    """Replace the y tick labels by the class chips, so every figure names a class
    the same way: its colour, its icon, its name."""
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for position, name in enumerate(names):
        style.class_chip(
            ax,
            (-0.012, position),
            name,
            xycoords=("axes fraction", "data"),
            label="",
        )
        ax.annotate(
            name,
            xy=(-0.012, position),
            xycoords=("axes fraction", "data"),
            xytext=(-style.bubble_size() / 2 - 6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style.SIZES["label"],
            color=style.INK,
            annotation_clip=False,
        )


def _barh(
    ax: plt.Axes, names: list[str], values: np.ndarray, colors: list[str], *, log: bool = False
) -> None:
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.6, height=0.72)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=style.INK)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(0.8, values.max() * 2.6)
    else:
        ax.set_xlim(0, values.max() * 1.12)
    for y, value in enumerate(values):
        ax.text(
            value * (1.10 if log else 1) + (0 if log else values.max() * 0.012),
            y,
            str(int(value)),
            va="center",
            fontsize=10,
            color=style.INK,
        )


def _lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with white; amount 0 = unchanged, 1 = white."""
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * np.clip(amount, 0, 1))
