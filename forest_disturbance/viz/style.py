"""The look of every figure in this project: colours, fonts, annotations, export.

Import this before plotting so all figures share one design:

    from forest_disturbance.viz import style
    style.use()

The palette is colour-blind friendly (Paul Tol's "muted" set). Class colours are
fixed here and used by every figure, so "Wind" is the same colour in a map, in a
time series and in a confusion matrix.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

# --- palette ---------------------------------------------------------------------------

INK = "#15181A"  # main text, axis labels, tick labels
MUTED = "#3E4A46"  # secondary text — still dark enough to read on white
FAINT = "#AFB8B3"  # grid, thin separators
PAPER = "#FFFFFF"  # figure background
PANEL = "#F4F5F3"  # a panel behind something (e.g. land in a map)

#: One colour per model class (configs/label_mapping.yaml). Used everywhere.
CLASS_COLORS: dict[str, str] = {
    "Clear Cut": "#8F58FD",  # purple      - planned removal
    "Thinning": "#C7A5FD",  # light purple - partial removal
    "Salvage": "#FD81E6",  # pink         - clean-up after damage
    "Wildfire": "#E8A302",  # amber        - flames
    "Wind": "#7EC6EA",  # light blue   - storm
    "Biotic": "#D54C34",  # brick red    - insects
    "No Disturbance": "#34A236",  # green        - the forest is fine
}

#: Symbols drawn on the annotation lane instead of a plain marker. These ship with
#: every matplotlib install (DejaVu Sans), so a figure looks the same everywhere.
CLASS_GLYPHS: dict[str, str] = {
    "Clear Cut": "⚒",  # tools: a planned cut
    "Thinning": "✂",  # scissors: a partial cut
    "Salvage": "✚",  # cross: clean-up after damage
    "Wildfire": "✹",  # flame
    "Wind": "≈",  # gust
    "Biotic": "✻",  # insect
    "No Disturbance": "●",
}

#: The same idea in colour emoji. Matplotlib cannot draw colour emoji as text, so
#: `viz.icons` renders these into small pictures instead (needs a colour emoji font
#: on the machine, and falls back to the glyphs above when there is none).
CLASS_EMOJI: dict[str, str] = {
    "Clear Cut": "🪓",
    "Thinning": "✂️",
    "Salvage": "🚑",
    "Wildfire": "🔥",
    "Wind": "💨",
    "Biotic": "🪲",
    "No Disturbance": "🌳",
}

#: "emoji" (colour pictures) or "glyph" (monochrome text symbols, work everywhere).
ICON_SET = "emoji"
#: Font used for the glyphs; it carries the symbols above whatever the body font is.
ICON_FONT = "DejaVu Sans"

#: What is written next to an event on the annotation lane. The class symbol is
#: already drawn as the marker, so the default text is just the date. One of
#: "date", "name", "name-date", "icon-date", "icon-name", "icon-name-date", "none".
EVENT_LABEL = "date"

#: Every size of the sample figure, in points, derived from one text size.
#: Text that sits at the same level of the figure shares `label`, and the round
#: badges, the ribbons and the acquisition ticks are sized from it, so changing
#: `label` alone rescales the whole annotation lane consistently.
SIZES: dict[str, float] = {
    "label": 11.5,  # event labels, ribbon labels, picture dates, sensor names
    "axis": 12.5,  # axis labels
    "tick": 11.5,  # tick labels
    "title": 15.5,  # figure title
    "tick_height": 19.0,  # height of one acquisition mark
    "tick_width": 1.5,  # width of one acquisition mark
    "event_icon": 15.0,  # the class icon on the annotation lane (inside a bubble)
    "event_rule": 2.8,  # the vertical line under it
    "event_stem": 26.0,  # how far that line rises above the ribbon
    "dot": 5.0,  # series and radar dots (deliberately the same)
    "line": 1.8,  # series line
    "ribbon_rounding": 0.40,  # corner radius of a period ribbon, as a share of its height
}

#: Colour of an annotated period, by the tens digit of its label code. Healthy periods
#: (forest, revegetation) share the "No Disturbance" green; disturbed periods take their
#: class colour, so a ribbon and a marker of the same event always match.
PERIOD_COLORS: dict[int, str] = {
    110: CLASS_COLORS["No Disturbance"],  # undisturbed forest
    120: CLASS_COLORS["No Disturbance"],  # revegetation
    200: "#6E7B76",  # disturbed, agent unknown
    210: CLASS_COLORS["Clear Cut"],
    220: CLASS_COLORS["Salvage"],
    230: CLASS_COLORS["Biotic"],
    240: CLASS_COLORS["Wildfire"],
}

#: Sentinel-1 / Sentinel-2 accent colours, as in the DISFOR visualisation toolkit.
S1_COLOR = "#3B82A8"
S2_COLOR = "#E07A3A"
FEATURE_COLOR = "#6D28D9"

#: Colormap for maps and heatmaps.
SEQUENTIAL = "coolwarm"

DASHED = (0, (2.5, 2.5))


def class_color(name: str) -> str:
    """Colour of a model class, grey for anything unknown."""
    return CLASS_COLORS.get(name, MUTED)


def class_icon(name: str) -> str:
    """Symbol of a model class, from the active icon set."""
    icons = CLASS_EMOJI if ICON_SET == "emoji" else CLASS_GLYPHS
    return icons.get(name, "●")


def event_label(class_name: str, day, kind: str = EVENT_LABEL) -> str:
    """Text next to an event on the annotation lane, in the configured format."""
    icon, name, date = class_icon(class_name), class_name, str(day)
    return {
        "none": "",
        "date": date,
        "icon": icon,
        "icon-date": f"{icon} {date}",
        "icon-name": f"{icon} {name}",
        "icon-name-date": f"{icon} {name} · {date}",
        "name": name,
        "name-date": f"{name} · {date}",
    }[kind]


def use(font_scale: float = 1.0) -> None:
    """Apply the project style to matplotlib. Call once before plotting."""
    family = "Lato" if _has_font("Lato") else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "figure.dpi": 120,
            "savefig.facecolor": PAPER,
            "font.family": family,
            "font.size": SIZES["label"] * font_scale,
            "text.color": INK,
            "axes.facecolor": PAPER,
            "axes.edgecolor": "#8D9691",
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK,
            "axes.labelsize": SIZES["axis"] * font_scale,
            "axes.titlesize": SIZES["title"] * font_scale,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.titlepad": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": FAINT,
            "grid.alpha": 0.6,
            "grid.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": SIZES["tick"] * font_scale,
            "ytick.labelsize": SIZES["tick"] * font_scale,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
            "legend.fontsize": SIZES["label"] * font_scale,
            "image.interpolation": "nearest",
        }
    )


def _has_font(name: str) -> bool:
    from matplotlib import font_manager

    return name in {f.name for f in font_manager.fontManager.ttflist}


# --- annotations -----------------------------------------------------------------------


def callout(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float] | None = None,
    xytext: tuple[float, float] = (0.5, 0.5),
    *,
    color: str = INK,
    fontsize: float = 10,
    align: str = "center",
    va: str = "center",
    transform: str = "axes",
    zorder: float = 20,
) -> mpl.text.Annotation:
    """A short explanation placed *on top of* a figure, in a soft rounded card.

    Use it to teach a figure, not to label data (that is what axis labels are for).

    Args:
        ax: Axes to draw on.
        text: The note. Keep it to one or two short lines.
        xy: Point the note refers to; an arrow is drawn to it. None = no arrow.
        xytext: Where the card sits.
        transform: "axes" = xytext in axes fractions (0-1), "data" = data coordinates.
            `xy` is always in data coordinates.
    """
    box = {
        "boxstyle": "round,pad=0.45,rounding_size=0.5",
        "facecolor": "white",
        "edgecolor": FAINT,
        "linewidth": 0.8,
        "alpha": 0.96,
    }
    arrow = None
    if xy is not None:
        arrow = {
            "arrowstyle": "-",
            "color": MUTED,
            "linewidth": 0.9,
            "shrinkA": 6,
            "shrinkB": 3,
            "connectionstyle": "arc3,rad=0.12",
        }
    annotation = ax.annotate(
        text,
        xy=xy if xy is not None else xytext,
        xycoords="data" if xy is not None else (ax.transAxes if transform == "axes" else "data"),
        xytext=xytext,
        textcoords=ax.transAxes if transform == "axes" else "data",
        ha=align,
        va=va,
        fontsize=fontsize,
        color=color,
        bbox=box,
        arrowprops=arrow,
        zorder=zorder,
    )
    return annotation


def bubble_size() -> float:
    """Diameter of a numbered bubble, in points: a badge on a picture, a dot on the
    ruler, the bubble around an event icon, the fold number inside a bar.

    Derived from the size of the number inside it, so one text size drives them all.
    """
    return SIZES["label"] * 2.0


def bubble(ax: plt.Axes, x: float, y: float, number: int, color: str, *, filled: bool = True):
    """A numbered bubble at a point of the axes, in data coordinates.

    Filled: the number is white on `color`. Otherwise the bubble is white with a
    `color` border, which stays readable on top of a dark or a light background.
    """
    diameter = bubble_size()
    ax.scatter(
        [x],
        [y],
        s=diameter**2,
        facecolor=color if filled else "white",
        edgecolors="white" if filled else color,
        linewidths=1.6,
        zorder=6,
    )
    ax.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        fontsize=SIZES["label"],
        fontweight="bold",
        color="white" if filled else color,
        zorder=7,
    )


def badge(ax: plt.Axes, number: int, color: str, *, x: float = 0.11, y: float = 0.89) -> None:
    """Small numbered disc in the corner of an image, tying it to a date on a timeline.

    Its size follows `SIZES["label"]`, the size of the number inside it, so it always
    matches the bubbles of the ruler underneath.
    """
    ax.text(
        x,
        y,
        str(number),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=SIZES["label"],
        color="white",
        fontweight="bold",
        bbox={
            "boxstyle": "circle,pad=0.30",
            "facecolor": color,
            "edgecolor": "white",
            "linewidth": 1.6,
        },
        zorder=10,
    )


def outlined(text_artist, width: float = 2.0, color: str = "white") -> None:
    """Give a text a thin halo so it stays readable over an image."""
    text_artist.set_path_effects([withStroke(linewidth=width, foreground=color)])


def title(ax: plt.Axes, main: str, subtitle: str | None = None) -> None:
    """Title, with an optional second line underneath it. Neither overlaps the other."""
    if not subtitle:
        ax.set_title(main)
        return
    ax.set_title("")  # the title is drawn by hand so the two lines cannot collide
    ax.annotate(
        main,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 30),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=mpl.rcParams["axes.titlesize"],
        fontweight="semibold",
        color=INK,
        annotation_clip=False,
    )
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
        annotation_clip=False,
    )


def legend_swatches(
    ax: plt.Axes,
    entries: Sequence[tuple[str, str]],
    *,
    loc: str = "upper left",
    ncol: int = 1,
    **kwargs,
) -> mpl.legend.Legend:
    """Legend from (label, colour) pairs, drawn as small rounded patches."""
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in entries
    ]
    return ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.4,
        labelspacing=0.5,
        columnspacing=1.2,
        **kwargs,
    )


def class_chip(
    ax: plt.Axes,
    xy: tuple[float, float],
    name: str,
    *,
    xycoords: str | tuple[str, str] = "axes fraction",
    label: str | None = None,
    fontsize: float | None = None,
    gap: float = 5.0,
    color: str | None = None,
    icon: str | None = None,
) -> float:
    """A class bubble - its colour, with its icon inside - and its name beside it.

    The same marker everywhere: in a legend, next to a bar, on a map, under a
    picture. Returns the width it used, in points, so several chips can be laid
    out in a row.

    Args:
        ax: Where to draw.
        xy: Left edge of the chip, in `xycoords`.
        name: A model class, e.g. "Wildfire".
        xycoords: Coordinate system of `xy` ("axes fraction", "data", ...).
        label: Text beside the bubble; the class name by default, "" for none.
        fontsize: Text size; `SIZES["label"]` by default.
        gap: Space between the bubble and the text, in points.
        color: Bubble colour; the class colour by default.
        icon: Text to put inside the bubble instead of the class icon.
    """
    from forest_disturbance.viz import icons  # here: icons imports style

    size = SIZES["label"] if fontsize is None else fontsize
    diameter = size * 2.0
    text = name if label is None else label
    ax.scatter(
        [xy[0]],
        [xy[1]],
        transform=_transform(ax, xycoords),
        s=diameter**2,
        facecolor=class_color(name) if color is None else color,
        edgecolors="white",
        linewidths=1.6,
        zorder=6,
        clip_on=False,
    )
    if icon is None:
        icons.draw_icon(
            ax,
            xy,
            name,
            size=size * 1.05,
            color="white",
            zorder=7,
            xycoords=xycoords,
            clip_on=False,
        )
    else:
        ax.annotate(
            icon,
            xy=xy,
            xycoords=xycoords,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold",
            color="white",
            zorder=7,
            annotation_clip=False,
        )
    if text:
        ax.annotate(
            text,
            xy=xy,
            xycoords=xycoords,
            xytext=(diameter / 2 + gap, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=size,
            color=INK,
            zorder=7,
            annotation_clip=False,
        )
    return diameter + gap + len(text) * size * 0.55


def class_legend(
    ax: plt.Axes,
    names: Sequence[str],
    *,
    xy: tuple[float, float] = (0.0, 1.06),
    fontsize: float | None = None,
    spacing: float = 14.0,
) -> None:
    """A row of class chips above an axes, in place of a matplotlib legend.

    Matplotlib cannot put a colour emoji in a legend entry, and a row of chips
    reads faster than a row of line samples anyway.
    """
    size = SIZES["label"] if fontsize is None else fontsize
    width = ax.get_position().width * ax.figure.get_size_inches()[0] * 72
    x, y = xy
    for name in names:
        used = class_chip(ax, (x, y), name, fontsize=size)
        x += (used + spacing) / width


def _transform(ax: plt.Axes, xycoords):
    """Transform for a coordinate spec, which may mix the two axes.

    `"axes fraction"`, `"data"`, or a pair such as `("axes fraction", "data")` to
    pin something to the left edge of the axes at the height of one bar.
    """
    from matplotlib.transforms import blended_transform_factory

    if isinstance(xycoords, tuple):
        pick = {"axes fraction": ax.transAxes, "data": ax.transData}
        return blended_transform_factory(pick[xycoords[0]], pick[xycoords[1]])
    return ax.transAxes if xycoords == "axes fraction" else ax.transData


def bare(ax: plt.Axes) -> plt.Axes:
    """Strip an axes down to its content (no ticks, no frame, no grid)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def save(fig: plt.Figure, path: str | Path, *, dpi: int = 160, transparent: bool = False) -> Path:
    """Save a figure (creates parent folders). `transparent=True` for slides."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
        facecolor="none" if transparent else fig.get_facecolor(),
    )
    return path
