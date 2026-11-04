"""Regenerate the round logo used by the desktop app and the web UI.

The mark is the project cat on a soft circular badge, so it reads as a round
logo in the dock, the taskbar and the sidebar instead of a loose cat on a
transparent square. One source drives every output:

  desktop/build/icon.png        1024  electron-builder master (it refuses < 512)
  web/src/assets/logo-round.png  256  sidebar / login / assistant brand mark
  web/public/favicon.png          64  browser tab

The masters are committed because CI packages without a Python step; this
script is how they get rebuilt when the logo changes, so the icon has a source
instead of being a binary nobody can reproduce.

The full logo is a cat ringed by "AGENTICA / Build AI Agent". Only the cat is
taken: at 32px in a taskbar the ring is illegible noise.

    python desktop/make_icon.py
"""

from __future__ import annotations

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
# Windows both assume some; without it the circle touches the icon's edge.
MARGIN = 0.05
# The cat's longest side as a share of the badge diameter: big enough to read at
# 16px, small enough that the ears do not graze the circle.
CAT_FILL = 0.74
BADGE = (234, 243, 255, 255)  # the logo's own pale blue, so the cat's white belly still reads


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


def build(size: int = SIZE) -> Image.Image:
    # Draw at 4x and downsample: PIL's ellipse has no antialiasing.
    work = size * 4
    diameter = round(work * (1 - 2 * MARGIN))
    left = (work - diameter) // 2

    canvas = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).ellipse((left, left, left + diameter - 1, left + diameter - 1), fill=BADGE)

    cat = cut_out(Image.open(SOURCE).convert("RGBA").crop(CROP))
    scale = diameter * CAT_FILL / max(cat.size)
    cat = cat.resize((round(cat.width * scale), round(cat.height * scale)), Image.LANCZOS)
    canvas.paste(cat, ((work - cat.width) // 2, (work - cat.height) // 2 + round(work * 0.012)), cat)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    master = build(SIZE)
    targets = {
        HERE / "build" / "icon.png": SIZE,
        ROOT / "web" / "src" / "assets" / "logo-round.png": 256,
        ROOT / "web" / "public" / "favicon.png": 64,
    }
    for out, size in targets.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        img = master if size == SIZE else master.resize((size, size), Image.LANCZOS)
        img.save(out)
        print(f"wrote {out.relative_to(ROOT)} ({size}x{size})")


if __name__ == "__main__":
    main()
