"""Inline-image preparation: oversize edge cap, PNG preference, byte cap.

The token cost of an image is set by its decoded geometry, so these tests
assert on the *bytes we send*, not on token counts: an image already inside
the cap must travel byte-for-byte, and one past it must come back smaller,
as PNG, and still decodable.
"""
import base64
import io
import os
import random

import pytest
from PIL import Image

from agentica.model.base import (
    _VISION_PNG_BUDGET_BYTES,
    VISION_MAX_IMAGE_BYTES,
    VISION_MAX_IMAGE_EDGE,
    _prepare_image_bytes,
)


def _png_bytes(size, mode="RGB", color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _decode(data_url):
    """Pull the decoded bytes + mime out of a ``data:`` URL."""
    header, payload = data_url.split(",", 1)
    mime = header[len("data:"):].split(";")[0]
    return base64.b64decode(payload), mime


def _assert_same_bytes(out, raw, msg=""):
    """Compare large payloads without letting pytest diff megabytes on failure.

    A bare ``out == raw`` makes pytest render both blobs (and hang) when the
    assertion is the very thing under test.
    """
    same = out is raw or out == raw
    assert same, f"expected bytes to be unchanged{': ' + msg if msg else ''} " \
                 f"(got {len(out)} bytes, wanted {len(raw)})"


def test_small_image_passes_through_untouched():
    """Under the cap: identical bytes, identical mime — no re-encode."""
    raw = _png_bytes((800, 600))

    out, mime = _prepare_image_bytes(raw, "image/png")

    _assert_same_bytes(out, raw)
    assert mime == "image/png"


def test_image_at_exactly_the_cap_is_untouched():
    """Boundary: longest edge == cap is inside, not outside."""
    raw = _png_bytes((VISION_MAX_IMAGE_EDGE, 40))

    out, mime = _prepare_image_bytes(raw, "image/png")

    _assert_same_bytes(out, raw)
    assert mime == "image/png"


def test_oversized_image_is_downscaled_to_the_cap():
    raw = _png_bytes((4000, 3000))

    out, mime = _prepare_image_bytes(raw, "image/png")

    with Image.open(io.BytesIO(out)) as img:
        assert max(img.size) == VISION_MAX_IMAGE_EDGE
        # Aspect ratio preserved (4:3 stays 4:3).
        assert round(img.size[0] / img.size[1], 2) == pytest.approx(1.33, abs=0.02)
    assert mime == "image/png"


def test_downscaled_output_is_smaller_than_input():
    raw = _png_bytes((5000, 4000))

    out, _ = _prepare_image_bytes(raw, "image/png")

    assert len(out) < len(raw)


def test_png_is_preferred_for_screenshots():
    """A text-like image must stay lossless PNG, not silently become JPEG."""
    img = Image.new("RGB", (4000, 200), (255, 255, 255))
    for x in range(0, 4000, 3):  # thin strokes => JPEG would smear them
        for y in range(0, 200, 7):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png"


def test_a_compact_png_keeps_png_even_when_jpeg_is_smaller():
    """JPEG is a fallback, not a size optimiser.

    Image tokens come from geometry, not bytes, so trading text legibility for
    a smaller JPEG buys nothing. Only a PNG too big to send justifies it.
    """
    raw = _screenshot_like_rgba()

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png"
    assert len(out) <= _VISION_PNG_BUDGET_BYTES


def test_alpha_is_preserved_when_downscaling():
    """RGBA stays RGBA — the PNG fallback must not drop the alpha channel."""
    raw = _png_bytes((3000, 3000), mode="RGBA", color=(10, 20, 30, 128))

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png"
    with Image.open(io.BytesIO(out)) as img:
        assert img.mode in ("RGBA", "LA")
        assert img.getpixel((10, 10))[3] == 128


def test_palette_transparency_survives_downscale():
    """A P-mode PNG with a transparency key must not be flattened to RGB."""
    img = Image.new("P", (3000, 100))
    img.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    img.info["transparency"] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG", transparency=0)
    raw = buf.getvalue()

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png"
    with Image.open(io.BytesIO(out)) as out_img:
        assert out_img.mode in ("RGBA", "LA", "P")


def test_narrow_tall_image_caps_long_edge():
    """A full-page screenshot is tall, not wide — the cap is on the long edge."""
    raw = _png_bytes((600, 9000))

    out, _ = _prepare_image_bytes(raw, "image/png")

    with Image.open(io.BytesIO(out)) as img:
        assert max(img.size) == VISION_MAX_IMAGE_EDGE
        assert img.size[1] > img.size[0]


def _noisy_jpeg(size, mode="RGB", quality=40):
    """Noise compresses badly in PNG, so the PNG output exceeds the input."""
    channels = 3 if mode == "RGB" else 4
    img = Image.frombytes(mode, size, os.urandom(size[0] * size[1] * channels))
    if mode == "RGBA":
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def test_jpeg_fallback_when_png_would_inflate():
    """A photo must not grow: PNG that beats the input wins, oversized PNG falls back.

    Downscaling is only worth doing if the payload actually shrinks — an
    alpha-free image whose PNG re-encode is *larger* than the original is a
    regression, so JPEG takes over there.
    """
    raw = _noisy_jpeg((2200, 2200), mode="RGB", quality=40)

    out, mime = _prepare_image_bytes(raw, "image/jpeg")

    assert mime == "image/jpeg"
    assert len(out) < len(raw)
    with Image.open(io.BytesIO(out)) as img:
        assert max(img.size) == VISION_MAX_IMAGE_EDGE


def test_alpha_blocks_the_jpeg_fallback():
    """JPEG has no alpha: an RGBA image must stay PNG even though JPEG is smaller."""
    raw = _noisy_jpeg((2001, 2001), mode="RGBA")

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png"
    with Image.open(io.BytesIO(out)) as img:
        assert img.mode in ("RGBA", "LA")


def test_screenshot_png_stays_png_even_though_jpeg_is_smaller():
    """The PNG preference is deliberate, not just a smaller-wins rule.

    A downscaled screenshot gets smaller as JPEG, but silently lossy text is
    the wrong trade — the MIME must stay image/png.
    """
    img = Image.new("RGB", (3000, 2000), (255, 255, 255))
    for x in range(0, 3000, 5):
        for y in range(0, 2000, 9):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png", "screenshot must not be re-encoded to lossy JPEG"
    with Image.open(io.BytesIO(out)) as resized:
        assert max(resized.size) == VISION_MAX_IMAGE_EDGE


def test_original_wins_when_nothing_else_shrinks():
    """Nothing beats the original: keep it rather than emit a larger capped copy.

    Providers clamp the long edge server-side before tokenising, so the cap is
    mainly about bytes — and capping must not *cost* bytes. This image is
    already compact for its size (an alpha image JPEG cannot represent), so the
    original is the right answer.
    """
    img = Image.frombytes("RGBA", (2200, 2200), os.urandom(2200 * 2200 * 4))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=30, method=0)
    raw = buf.getvalue()

    out, mime = _prepare_image_bytes(raw, "image/webp")

    _assert_same_bytes(out, raw, "oversized alpha image that cannot shrink")
    assert mime == "image/webp"


def test_opaque_oversized_image_always_shrinks():
    """An ordinary screenshot bigger than the cap must come back smaller.

    Regression guard for the realistic case: macOS clipboard screenshots are
    RGBA even when fully opaque, so treating the channel as transparency would
    skip the JPEG fallback and inflate the payload.
    """
    # Opaque alpha, the way macOS writes a clipboard PNG.
    img = Image.frombytes("RGB", (2880, 1800), os.urandom(2880 * 1800 * 3)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    out, mime = _prepare_image_bytes(raw, "image/png")

    with Image.open(io.BytesIO(out)) as resized:
        assert max(resized.size) == VISION_MAX_IMAGE_EDGE
        assert resized.mode == "RGB", "fully-opaque alpha should be dropped"
    assert len(out) < len(raw) / 2, "a dense retina screenshot should shrink a lot"


def test_opaque_alpha_channel_is_dropped():
    """RGBA with every pixel opaque becomes RGB — the 4th channel is dead weight."""
    from agentica.model.base import _has_transparency

    opaque = Image.new("RGBA", (50, 50), (1, 2, 3, 255))
    assert _has_transparency(opaque) is False

    transparent = Image.new("RGBA", (50, 50), (1, 2, 3, 255))
    transparent.putpixel((0, 0), (1, 2, 3, 0))
    assert _has_transparency(transparent) is True


def _screenshot_like_rgba(size=(2400, 1200), seed=11):
    """An opaque RGBA image resembling a screenshot: flat areas, hard edges.

    A 1px checker texture is averaged away by downscaling, so the capped PNG is
    smaller than the original — the ordinary case, unlike a tiny few-colour
    PNG whose resampled form needs more shades than the source did.
    """
    random.seed(seed)
    width, height = size
    img = Image.new("RGB", size, (255, 255, 255))
    px = img.load()
    for y in range(height):
        for x in range(width):
            value = 255 if (x + y) % 2 else 250
            px[x, y] = (value, value, value)
    for _ in range(60):
        x0 = random.randrange(0, width - 200)
        y0 = random.randrange(0, height - 100)
        for y in range(y0, y0 + 60):
            for x in range(x0, x0 + 120):
                px[x, y] = (30, 30, 40)
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def test_opaque_alpha_stripped_on_the_png_path_too():
    """The dead channel is dropped even when the result stays PNG.

    A few-colour oversized screenshot keeps PNG (JPEG is no smaller there), so
    the alpha stripping has to happen before the PNG encode, not only inside
    the JPEG branch.
    """
    raw = _screenshot_like_rgba()

    out, mime = _prepare_image_bytes(raw, "image/png")

    assert mime == "image/png", "a few-colour screenshot should stay PNG"
    with Image.open(io.BytesIO(out)) as resized:
        assert max(resized.size) == VISION_MAX_IMAGE_EDGE
        assert resized.mode == "RGB", "opaque alpha should have been dropped"
    assert len(out) < len(raw)


def test_palette_transparency_detected():
    from agentica.model.base import _has_transparency

    img = Image.new("P", (10, 10))
    img.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    img.info["transparency"] = 0
    assert _has_transparency(img) is True

    plain = Image.new("P", (10, 10))
    plain.info.pop("transparency", None)
    assert _has_transparency(plain) is False


def test_over_byte_cap_is_refused():
    huge = b"\x00" * (VISION_MAX_IMAGE_BYTES + 1)

    with pytest.raises(ValueError, match="too large"):
        _prepare_image_bytes(huge, "image/png", source="/tmp/huge.png")


def test_undecodable_bytes_pass_through():
    """An SVG or corrupt file is returned as-is, not fatal to the request."""
    raw = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"

    out, mime = _prepare_image_bytes(raw, "image/svg+xml")

    assert out == raw
    assert mime == "image/svg+xml"


def test_data_url_round_trips_through_model():
    """End to end: a 4000px data URI is capped, not passed through raw."""
    from agentica.model.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
    raw = _png_bytes((4000, 500))
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()

    block = model.process_image(data_url)
    payload, mime = _decode(block["image_url"]["url"])

    assert block["type"] == "image_url"
    assert mime == "image/png"
    assert len(payload) <= len(raw)
    with Image.open(io.BytesIO(payload)) as img:
        assert max(img.size) == VISION_MAX_IMAGE_EDGE


def test_gateway_data_url_dict_is_capped():
    from agentica.model.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
    raw = _png_bytes((4000, 400))
    block = model.process_image({
        "url": "data:image/png;base64," + base64.b64encode(raw).decode(),
        "detail": "high",
    })
    payload, _ = _decode(block["image_url"]["url"])
    assert block["image_url"]["detail"] == "high"
    with Image.open(io.BytesIO(payload)) as img:
        assert max(img.size) == VISION_MAX_IMAGE_EDGE


def test_decompression_bomb_is_not_swallowed(monkeypatch):
    import agentica.model.base as model_base

    raw = _png_bytes((32, 32))

    def boom(*_a, **_k):
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(model_base.Image, "open", boom)
    with pytest.raises(Image.DecompressionBombError):
        _prepare_image_bytes(raw, "image/png")


def test_oversized_pil_image_is_resized_before_encode():
    from agentica.model.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
    img = Image.new("RGB", (4000, 800), (10, 20, 30))
    block = model.process_image(img)
    payload, mime = _decode(block["image_url"]["url"])
    assert mime == "image/png"
    with Image.open(io.BytesIO(payload)) as out:
        assert max(out.size) == VISION_MAX_IMAGE_EDGE
