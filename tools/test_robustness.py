"""
Input robustness — which real-world image problems actually break conversion?

Takes a design that is known to convert cleanly, applies the degradations that
real uploads suffer, and measures what survives. Anything that fails here is a
concrete gap, not a hypothetical one.

Each degradation is applied to the SOURCE, then the result is scored against
the CLEAN original — so the question asked is "would a weaver get the right
cloth from this photo?", not "did the pipeline do something with it".

Run:  python tools/test_robustness.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fidelity import fidelity_report, _binary          # noqa: E402
from vision_engine import detect_colors_smart          # noqa: E402

SOURCE = '/mnt/user-data/uploads/WhatsApp_Image_2026-07-18_at_10_05_44_PM__1_.jpeg'
PINS = 859


# ── Degradations ────────────────────────────────────────────────────────────

def d_clean(img):
    return img


def d_rotate(img, deg=3.0):
    """Photographed slightly off-square — extremely common with phone photos."""
    return img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def d_perspective(img, k=0.06):
    """Photographed at an angle: the far edge is narrower than the near edge."""
    w, h = img.size
    dx = int(w * k)
    coeffs = _perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(dx, 0), (w - dx, 0), (w, h), (0, h)])
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         Image.BICUBIC, fillcolor=(255, 255, 255))


def d_shadow(img, strength=0.45):
    """A hand or phone shadow falling across part of the design."""
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    field = 1.0 - strength * np.clip((xx / w) * 1.6 - 0.4, 0, 1)
    return Image.fromarray(np.clip(a * field[:, :, None], 0, 255).astype(np.uint8))


def d_blur(img, r=1.6):
    """Slightly out of focus."""
    return img.filter(ImageFilter.GaussianBlur(r))


def d_jpeg(img, q=25):
    """Heavily recompressed, e.g. forwarded through chat apps repeatedly."""
    import io as _io
    buf = _io.BytesIO()
    img.save(buf, 'JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def d_noise(img, sigma=14):
    """Photographed in poor light."""
    rng = np.random.default_rng(11)
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def d_lowcontrast(img, k=0.45):
    """Faint pencil, or a washed-out scan."""
    a = np.asarray(img, dtype=np.float32)
    return Image.fromarray(np.clip(255 - (255 - a) * k, 0, 255).astype(np.uint8))


def d_colourcast(img):
    """Shot under tungsten or fluorescent light."""
    a = np.asarray(img, dtype=np.float32) * np.array([1.10, 0.98, 0.82])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def d_downscale(img, f=0.5):
    """Sent as a chat thumbnail."""
    w, h = img.size
    small = img.resize((int(w * f), int(h * f)), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def _perspective_coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.asarray(m, dtype=np.float64)
    B = np.asarray(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B).tolist()


DEGRADATIONS = [
    ('clean (control)', d_clean),
    ('rotated 3 degrees', d_rotate),
    ('perspective skew', d_perspective),
    ('shadow across design', d_shadow),
    ('slightly out of focus', d_blur),
    ('heavy JPEG (q25)', d_jpeg),
    ('sensor noise', d_noise),
    ('low contrast / faint', d_lowcontrast),
    ('warm colour cast', d_colourcast),
    ('chat-thumbnail downscale', d_downscale),
]


def main():
    if not os.path.exists(SOURCE):
        print(f'Source not available: {SOURCE}')
        return 0

    clean = Image.open(SOURCE).convert('RGB')
    cards = int(round(PINS * clean.size[1] / clean.size[0]))
    ref = _binary(clean)
    ref_ink = ref.mean()

    print(f'\nSource {clean.size} at {PINS} pins — each degradation scored '
          f'against the CLEAN original\n')
    print(f'  {"input condition":<28} {"ink drift":>10} {"gaps":>16} {"verdict":>8}')
    print('  ' + '-' * 66)

    rows = []
    for name, fn in DEGRADATIONS:
        try:
            degraded = fn(clean)
            _, _, lm, _ = detect_colors_smart(degraded, 2, PINS, cards)
            mask = np.asarray(lm) > 0
            # Score against the CLEAN source: the weaver wants the real design,
            # not a faithful reproduction of a bad photograph.
            rep = fidelity_report(clean, mask)
            drift = 100 * (mask.mean() / ref_ink - 1)
            gaps = f"{rep['source_white_regions']}→{rep['output_white_regions']}"
            print(f'  {name:<28} {drift:>+9.0f}% {gaps:>16} {rep["verdict"].upper():>8}')
            rows.append((name, abs(drift), rep['verdict']))
        except Exception as e:
            print(f'  {name:<28} {"CRASHED":>10}   {type(e).__name__}: {e}')
            rows.append((name, 999, 'crash'))

    print()
    bad = [r for r in rows if r[2] in ('fail', 'crash') or r[1] > 40]
    if bad:
        print('  Conditions that break conversion:')
        for n, d, v in bad:
            print(f'    - {n} ({v}, {d:.0f}% drift)')
    else:
        print('  All tested conditions stay within tolerance.')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
