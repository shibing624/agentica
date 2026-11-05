"""Regenerate the app icon used by the desktop app and the web UI.

The mark is the project cat on a rounded-square (squircle) badge — the shape
macOS / iOS app icons use: a square with continuously curved corners, not a
circle and not a hard-cornered square. One source drives every output:

  desktop/build/icon.png        1024  electron-builder master (it refuses < 512)
  desktop/build/icon-mac.png    1024  macOS only: full-bleed square, the OS rounds it
  web/src/assets/app-icon.png    256  sidebar / welcome / login brand mark
  web/public/favicon.png          64  browser tab
  web/public/favicon.ico           multi-resolution ICO, same artwork

The masters are committed because CI packages without a Python step; this
script is how they get rebuilt when the logo changes, so the icon has a source
instead of being a binary nobody can reproduce.

The full logo is a cat ringed by "AGENTICA / Build AI Agent". Only the cat is
taken: at 32px in a taskbar the ring is illegible noise.

    python desktop/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "docs" / "assets" / "logo.png"

SIZE = 1024
# The cat's box inside logo.png (918x976). Hand-measured rather than derived
# from the alpha channel: the surrounding text is opaque too, so a bounding box
# of non-transparent pixels is the whole logo.
CROP = (245, 243, 676, 752)
# Transparent margin around the badge, as a share of the canvas. macOS and
# Windows both assume some; without it the shape touches the icon's edge.
MARGIN = 0.055
# Superellipse exponent. n=2 is an ellipse, n->infinity a square; iOS uses
# roughly 4-5, which keeps the straight edges but rounds the corners smoothly.
SQUIRCLE_N = 4.6
# The cat's longest side as a share of the badge side. Reference app icons keep
# roughly 20-30% air around the glyph; 0.66 leaves that without shrinking the
# cat illegible at 16px.
CAT_FILL = 0.66
# Same, for the full-bleed macOS icon, as a share of the whole canvas. The
# system mask trims the corners, so the cat sits a little smaller than
# CAT_FILL * (1 - 2 * MARGIN) would suggest to keep it clear of the rounded edge.
MAC_CAT_FILL = 0.62
BADGE = (234, 243, 255, 255)  # the logo's own pale blue
BORDER = (196, 219, 250, 255)  # a hairline so the pale badge reads on a light sidebar
BORDER_W = 0.006  # share of the canvas


def cut_out(cat: Image.Image) -> Image.Image:
    """Make the white box behind the cat transparent.

    logo.png is opaque white around the cat. Flood fill from the crop's edges so
    only the background connected to the border goes; the white belly stays,
    because the black outline closes it off. Light, nearly-white pixels hugging
    the outline fade out too, otherwise they show as a halo on the badge.
    """
    rgb = cat.convert("RGB")
    marker = (255, 0, 255)
    probe = rgb.copy()
    w, h = probe.size
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if probe.getpixel(seed) != marker:
            ImageDraw.floodfill(probe, seed, marker, thresh=48)
    out = cat.convert("RGBA")
    px, src, flood = out.load(), rgb.load(), probe.load()
    for y in range(h):
        for x in range(w):
            if flood[x, y] == marker:
                px[x, y] = (255, 255, 255, 0)
                continue
            # One pixel in from the cut: ease the antialiased rim into transparency.
            near = any(
                0 <= x + dx < w and 0 <= y + dy < h and flood[x + dx, y + dy] == marker
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if near:
                lum = sum(src[x, y]) / 3
                if lum > 150:
                    px[x, y] = (*src[x, y], max(0, int((255 - lum) * 255 / 105)))
    return out


def squircle(cx: float, cy: float, half: float, n: float, steps: int = 720):
    """Points on |x/half|^n + |y/half|^n = 1, sampled for a polygon fill."""
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = math.copysign(abs(c) ** (2 / n), c)
        y = math.copysign(abs(s) ** (2 / n), s)
        pts.append((cx + half * x, cy + half * y))
    return pts


def build(size: int = SIZE) -> Image.Image:
    # Draw at 4x and downsample: PIL's polygons and ellipse have no antialiasing.
    scale = 4
    work = size * scale
    side = round(work * (1 - 2 * MARGIN))
    half = side / 2
    cx = cy = work / 2

    canvas = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # Two filled squircles make a hairline border without a stroke width that
    # would have to be scaled by 4 differently on every output size.
    draw.polygon(squircle(cx, cy, half, SQUIRCLE_N), fill=BORDER)
    draw.polygon(squircle(cx, cy, half - work * BORDER_W, SQUIRCLE_N), fill=BADGE)

    cat = cut_out(Image.open(SOURCE).convert("RGBA").crop(CROP))
    factor = side * CAT_FILL / max(cat.size)
    cat = cat.resize((round(cat.width * factor), round(cat.height * factor)), Image.LANCZOS)
    canvas.paste(cat, ((work - cat.width) // 2, (work - cat.height) // 2 + round(work * 0.012)), cat)
    return canvas.resize((size, size), Image.LANCZOS)


def build_mac(size: int = SIZE) -> Image.Image:
    """Full-bleed square for macOS.

    macOS 26+ applies the icon mask itself. An icon that carries its own rounded
    shape and transparent margin (``build``) is treated as a legacy icon and
    shrunk into a grey rounded container, which is the "icon does not fill the
    tile" look. A flat opaque square that reaches every edge is masked directly,
    so the artwork fills the tile. No margin, no corners, no border: the system
    supplies all three.
    """
    scale = 4
    work = size * scale
    canvas = Image.new("RGBA", (work, work), BADGE)
    cat = cut_out(Image.open(SOURCE).convert("RGBA").crop(CROP))
    factor = work * MAC_CAT_FILL / max(cat.size)
    cat = cat.resize((round(cat.width * factor), round(cat.height * factor)), Image.LANCZOS)
    canvas.paste(cat, ((work - cat.width) // 2, (work - cat.height) // 2 + round(work * 0.012)), cat)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    mac = HERE / "build" / "icon-mac.png"
    build_mac(SIZE).save(mac)
    print(f"wrote {mac.relative_to(ROOT)} ({SIZE}x{SIZE}, full-bleed)")

    master = build(SIZE)
    outputs = [
        (HERE / "build" / "icon.png", SIZE),
        (ROOT / "web" / "src" / "assets" / "app-icon.png", 256),
        (ROOT / "web" / "public" / "favicon.png", 64),
    ]
    for out, size in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        img = master if size == SIZE else master.resize((size, size), Image.LANCZOS)
        img.save(out)
        print(f"wrote {out.relative_to(ROOT)} ({size}x{size})")

    # The tab asks for /favicon.ico too. Keep it the same artwork (the previous
    # one was a bare transparent cat). BMP inside ICO compresses poorly, so
    # stick to the sizes browsers actually pick — 16/32/48 lands ~7KB.
    ico = ROOT / "web" / "public" / "favicon.ico"
    master.resize((256, 256), Image.LANCZOS).save(
        ico, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print(f"wrote {ico.relative_to(ROOT)} (ICO, 16/32/48)")


if __name__ == "__main__":
    main()
