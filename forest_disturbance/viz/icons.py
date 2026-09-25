"""Class icons for the figures.

Matplotlib draws text with its own font engine, which cannot paint colour emoji:
`ax.text(..., "\N{FIRE}")` comes out as an empty box. So an emoji is rendered once
into a small picture with Pillow and placed on the axes as an image instead.

Two icon sets are available, chosen with `style.ICON_SET`:

- ``"emoji"``  colour emoji (needs a colour emoji font on the machine; falls back
  to the glyph set when there is none),
- ``"glyph"``  monochrome symbols drawn as text, which work everywhere.

The only function figures need is `draw_icon`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from . import style

#: Colour emoji fonts, in order of preference. The first one found is used.
EMOJI_FONTS = (
    str(Path.home() / ".fonts/NotoColorEmoji.ttf"),
    str(Path.home() / ".local/share/fonts/NotoColorEmoji.ttf"),
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
)

#: Size of the bitmap strike inside Noto Color Emoji. Pillow can only rasterise a
#: bitmap emoji font at its native size, so the picture is drawn big and scaled down.
_NATIVE = 109


@lru_cache(maxsize=1)
def emoji_font_path() -> str | None:
    """Path of the first colour emoji font on this machine, or None if there is none."""
    return next((path for path in EMOJI_FONTS if Path(path).exists()), None)


@lru_cache(maxsize=64)
def emoji_image(character: str) -> np.ndarray | None:
    """One emoji as an RGBA array, cropped to its ink. None if it cannot be drawn."""
    path = emoji_font_path()
    if path is None:
        return None
    from PIL import Image, ImageDraw, ImageFont  # imported here: only emoji need Pillow

    font = ImageFont.truetype(path, _NATIVE)
    canvas = Image.new("RGBA", (_NATIVE * 2, _NATIVE * 2), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text(
        (_NATIVE, _NATIVE), character, font=font, embedded_color=True, anchor="mm"
    )
    box = canvas.getbbox()
    if box is None:  # the font has no picture for this character
        return None
    return np.asarray(canvas.crop(box)) / 255.0


def draw_icon(
    ax,
    xy,
    name: str,
    *,
    size: float,
    color: str,
    zorder: float = 6,
    xycoords: str = "data",
    clip_on: bool = True,
    **text_kwargs,
):
    """Draw the icon of class `name` centred on `xy`, `size` points tall.

    Uses the colour emoji when `style.ICON_SET` is "emoji" and one is available, and
    the monochrome glyph (tinted with `color`) otherwise, so a figure always renders.
    """
    if style.ICON_SET == "emoji":
        image = emoji_image(style.CLASS_EMOJI.get(name, ""))
        if image is not None:
            # OffsetImage scales by dpi/72 itself, so zoom is simply points per pixel.
            box = OffsetImage(image, zoom=size / image.shape[0])
            ax.add_artist(
                AnnotationBbox(
                    box,
                    xy,
                    xycoords=xycoords,
                    frameon=False,
                    pad=0,
                    zorder=zorder,
                    annotation_clip=clip_on,
                    clip_on=clip_on,
                )
            )
            return
    ax.annotate(
        style.CLASS_GLYPHS.get(name, "\N{BLACK CIRCLE}"),
        xy=xy,
        xycoords=xycoords,
        ha="center",
        va="center",
        fontsize=size,
        fontfamily=style.ICON_FONT,
        color=color,
        zorder=zorder,
        annotation_clip=clip_on,
        **text_kwargs,
    )
