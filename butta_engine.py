"""
Butta Engine — high-quality reduction of a dense motif (butta) down to a small
pin width (typically 150-240) for jacquard weaving.

The Generator and line-art paths preserve thin BLACK lines (they lean toward
adding ink). A butta needs the opposite: the detail lives in the thin WHITE
gaps between petals, inside dot-rings and around swirls. At 150-240 pins those
gaps are thinner than one thread, so a normal downscale (average + threshold,
plus the engine's edge dilation) fills them in and the motif becomes a black
blob.

This module reduces the motif with a GAP-PRESERVING method instead:
binarise at full source resolution, then reduce to the target pin width by
area-sampling and keeping a cell white unless it is clearly mostly black.
A coverage threshold controls the black/white balance; "auto" picks it so the
reduced ink-% matches the source, preserving visual weight. A detail slider
nudges it open (favour gaps) or solid (favour ink).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import label as _label

from bmp_engine import write_1bit_bmp


# ──────────────────────────────────────────────────────────────────────────
# Core reduction
# ──────────────────────────────────────────────────────────────────────────
def _binarize_full_res(image: Image.Image, thresh: float | None = None):
    """
    Binarise the source at full resolution. True/1 = ink (the motif).

    Robust to polarity and degenerate inputs: the ground tone is read from the
    image border (a butta sits on a margin of ground), and ink is taken as the
    side that differs from the ground. So a dark motif on white, a light motif
    on a dark ground, and constant/near-constant images are all handled without
    blanking the design.
    """
    g = np.asarray(image.convert('L')).astype(np.float32)
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    bg = float(np.median(border))                 # ground tone around the motif
    if thresh is None:
        try:
            from skimage.filters import threshold_otsu
            if float(g.min()) == float(g.max()):  # constant image -> no otsu
                raise ValueError('constant image')
            thresh = float(threshold_otsu(g))
        except Exception:
            thresh = bg - 40.0 if bg >= 128 else bg + 40.0
    if bg >= 128:                                  # light ground -> ink is darker
        ink = g < min(thresh, bg - 1.0)
    else:                                          # dark ground -> ink is lighter
        ink = g > max(thresh, bg + 1.0)
    return ink.astype(np.float32), float(thresh)


def _coverage_map(hi: np.ndarray, target_w: int):
    """
    Per-output-cell black fraction using area (BOX) resampling. Works for any
    reduction ratio (no integer-block requirement) and is exactly the fraction
    of each output cell that is black in the full-res source.
    Returns (frac_map float in [0,1], target_h).
    """
    H, W = hi.shape
    target_h = max(1, round(H * target_w / W))
    im = Image.fromarray((hi * 255).astype(np.uint8), 'L').resize(
        (target_w, target_h), Image.BOX)
    return np.asarray(im, np.float32) / 255.0, target_h


def reduce_gap_preserving(hi: np.ndarray, target_w: int, coverage: float):
    """Reduce: an output cell is ink only if >= `coverage` of it is black."""
    frac, target_h = _coverage_map(hi, target_w)
    return (frac >= coverage), target_h


def auto_coverage(hi: np.ndarray, target_w: int, want_ink: float | None = None):
    """Pick the coverage threshold so reduced ink-% ~= source ink-%."""
    if want_ink is None:
        want_ink = float(hi.mean())
    best = None
    for cov in np.arange(0.30, 0.80, 0.01):
        ink = float(reduce_gap_preserving(hi, target_w, float(cov))[0].mean())
        d = abs(ink - want_ink)
        if best is None or d < best[0]:
            best = (d, float(cov))
    return best[1]


def _despeckle(mask: np.ndarray, min_px: int = 1):
    """Remove isolated ink specks <= min_px. Does NOT fill white pinholes —
    at low pin counts those single-pixel gaps ARE the filigree detail and must
    be preserved, not closed."""
    m = mask.astype(bool)
    lbl, n = _label(m)
    if n:
        sizes = np.bincount(lbl.ravel())
        keep = np.isin(lbl, np.where(sizes > min_px)[0]); keep[lbl == 0] = False
        m = keep
    return m


def _autocrop(image: Image.Image, pad_frac: float = 0.04):
    """Trim surrounding whitespace and add a small uniform margin, so the motif
    fills the pin width. Returns the cropped image (or original if nothing to do)."""
    g = np.asarray(image.convert('L'))
    ink = g < 200
    if not ink.any():
        return image
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = g.shape
    pad = int(round(max(H, W) * pad_frac))
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(H, y1 + pad); x1 = min(W, x1 + pad)
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def reduce_butta(image: Image.Image, target_pins: int, detail: float = 0.0,
                 despeckle_px: int = 1, autocrop: bool = True,
                 thresh: float | None = None):
    """
    Reduce a (mono / B&W) butta to `target_pins` width, preserving negative space.

    detail : -1.0 .. +1.0  — 0 = auto (matches the SOURCE ink weight per design);
                              + = more open (favour gaps); - = more solid.
    Auto adapts to each design: it searches the coverage threshold so the FINAL
    (post-despeckle) ink-% matches the source ink-%, i.e. it preserves the
    original's visual weight. The detail slider then nudges open/solid from there.
    Returns (mask, info) — mask bool (target_h x target_pins), True = ink/UP.
    """
    if autocrop:
        image = _autocrop(image)
    hi, used_thresh = _binarize_full_res(image, thresh)
    src_ink = float(hi.mean())

    # Compute the coverage map ONCE — it is identical for every threshold, so the
    # auto-search only needs to re-threshold it (cheap) rather than re-resize the
    # full-resolution source 50 times (which made previews slow on large images).
    frac, target_h = _coverage_map(hi, target_pins)

    def _final(cov):
        m = frac >= cov
        if despeckle_px > 0:
            m = _despeckle(m, despeckle_px)
        return m

    # Auto baseline: match the source ink weight using the real output pipeline.
    base_cov, best = 0.5, None
    for cov in np.arange(0.30, 0.80, 0.01):
        ink = float(_final(cov).mean())
        d = abs(ink - src_ink)
        if best is None or d < best:
            best, base_cov = d, float(cov)

    cov = float(min(0.85, max(0.20, base_cov + detail * 0.18)))
    mask = _final(cov)
    info = {
        'source_size': list(image.size),
        'target_w': target_pins,
        'target_h': target_h,
        'coverage': round(cov, 3),
        'source_ink': round(100 * src_ink, 1),
        'result_ink': round(100 * float(mask.mean()), 1),
        'threshold': round(used_thresh, 1),
        'compression': round(image.size[0] / target_pins, 1),
    }
    return mask, info


# ──────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────
def mask_to_bmp_bytes(mask: np.ndarray) -> bytes:
    """1-bit BMP from an ink mask (True = ink/black/UP)."""
    arr = np.where(mask, 0, 1).astype(np.uint8)   # 0 = black/UP, 1 = white/DOWN
    return write_1bit_bmp(arr)


def mask_to_preview_png(mask: np.ndarray, scale: int = 1) -> Image.Image:
    """Grayscale preview (black ink on white) for the UI."""
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), 'L')
    if scale > 1:
        img = img.resize((mask.shape[1] * scale, mask.shape[0] * scale), Image.NEAREST)
    return img.convert('RGB')


def mask_to_label_map(mask: np.ndarray):
    """
    Build a (label_map, colors, assignments) tuple for handing the reduced motif
    to generate_bmps (Full output mode). index 0 = ground, 1 = ink (-> zari).
    """
    label_map = mask.astype(np.uint8)                 # 0 = ground, 1 = ink
    colors = [(255, 255, 255), (0, 0, 0)]
    assignments = {0: 'background', 1: 'zari'}
    return label_map, colors, assignments
