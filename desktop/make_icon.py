"""Regenerate the packaging icon from the project logo.

electron-builder derives every platform's icon (icns, ico, Linux pngs) from one
square master, and it refuses anything under 512x512 — which is why this cannot
just point at `web/src/assets/cat.png` (256) or `docs/assets/favicon.ico` (48)
the way the running shell does.

The master is committed (`build/icon.png`) because CI packages without a Python
step; this script is how it gets rebuilt when the logo changes, so the icon has
a source instead of being a binary nobody can reproduce.

The full logo is a cat ringed by "AGENTICA / Build AI Agent". Only the cat is
taken: at 32px in a taskbar the ring is illegible noise, and the SPA sidebar
already made the same call.

    python desktop/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "docs" / "assets" / "logo.png"
OUT = HERE / "build" / "icon.png"

SIZE = 1024
# The cat's box inside logo.png (918x976). Hand-measured rather than derived
# from the alpha channel: the surrounding text is opaque too, so a bounding box
# of non-transparent pixels is the whole logo.
CROP = (245, 243, 676, 737)
# Transparent margin, as a share of the canvas. macOS draws app icons on the
# assumption there is some; without it the cat's ears touch the dock's edge.
MARGIN = 0.06


def main() -> None:
    cat = Image.open(SOURCE).convert("RGBA").crop(CROP)
    inner = round(SIZE * (1 - 2 * MARGIN))
    scale = inner / max(cat.size)
    cat = cat.resize((round(cat.width * scale), round(cat.height * scale)), Image.LANCZOS)

    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(cat, ((SIZE - cat.width) // 2, (SIZE - cat.height) // 2), cat)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT} ({SIZE}x{SIZE}) from {SOURCE.name}{CROP}")


if __name__ == "__main__":
    main()
