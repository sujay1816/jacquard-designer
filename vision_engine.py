"""
Vision engine — high-accuracy colour detection for photographed designs.

This module is an OPTIONAL, drop-in replacement for bmp_engine.detect_colors.
It does not touch the deterministic BMP core (smart_fill, rani phase,
write_1bit_bmp); it only produces a better label_map for that core to consume.

Three problems it addresses, in the order they hurt:

  1. THIN LINES VANISH.
     The legacy path resizes the photo to (pins x cards) with LANCZOS and
     *then* clusters. A 2px vine in a 3000px photo becomes a grey smear at
     360 pins, and KMeans assigns the smear to background. The line is lost
     before clustering ever sees it.
     Fix: cluster at high resolution, then downsample the LABEL MAP by area
     coverage (majority vote) instead of interpolating the image. A thin
     rescue pass then reinstates design cells that majority-vote would drop.

  2. BAD LIGHTING SPLITS ONE THREAD INTO TWO COLOURS.
     A lighting gradient across a photographed saree pushes the same gold
     thread into two different clusters.
     Fix: divide by a heavily blurred copy of the image to flatten the
     illumination field before clustering.

  3. COLOURS MERGE OR CLUSTER WRONG.
     KMeans on raw RGB is perceptually wrong: RGB distance does not match
     visual difference, so gold-in-highlight and gold-in-shadow can sit
     further apart than gold does from maroon.
     Fix: cluster in CIELAB with luminance down-weighted (see LAB_WEIGHTS).
     Chroma is largely illumination-invariant, so this also cancels fabric
     weave texture, which is almost entirely a luminance effect.

     SLIC superpixel pre-averaging is available but OFF by default: it was
     measured to reduce thin-line recall at every segment size tried, because
     segment boundaries smear features narrower than a segment.

Determinism: every step here is seeded and deterministic. The same image with
the same settings always yields the same label_map.
"""
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

# Reuse the existing artefact-cluster test so 'is_genuine' semantics stay
# identical to the legacy path.
from bmp_engine import _is_genuine_colour

# Hi-res working factor. The image is processed at (pins*k, cards*k) so the
# label map can be pooled back down by exact integer blocks. k=3 recovers
# features down to ~1/3 of a loom cell; beyond k=4 the gain flattens and
# memory cost grows quadratically.
DEFAULT_HIRES_FACTOR = 0         # 0 = derive from the source image (recommended)
MAX_HIRES_FACTOR = 6
MAX_HIRES_PIXELS = 16_000_000    # safety cap on pins*k * cards*k

# A design class claims an output cell during rescue if it covers at least
# this fraction of the cell. 0.22 was tuned so a 1px line at k=3 (33%
# coverage) survives while single-pixel speckle (11%) does not.
THIN_RESCUE_COVERAGE = 0.22

# SLIC superpixels must be SMALLER than the thinnest feature we intend to
# keep, or averaging inside a segment erases that feature. Targeting ~5px
# segments keeps 1-2px lines intact while still cancelling weave texture.
TARGET_SUPERPIXEL_PX = 5.0
MAX_SUPERPIXELS = 120_000

# KMeans is fitted on a fixed-seed random subsample and then used to predict
# every pixel. Centroids converge to the same place from 200k samples as from
# several million, but fitting is ~100x faster. The subsample is drawn with a
# fixed seed, so results stay reproducible.
KMEANS_FIT_SAMPLES = 200_000

# Clustering weights for the (L, a, b) channels.
#
# A lighting gradient scales luminance hard but leaves chroma nearly intact,
# so down-weighting L makes clustering robust to uneven lighting WITHOUT the
# risk of subtracting an illumination estimate. Explicit field correction was
# tried and rejected: when a motif occupies a large share of the frame, no
# blur radius separates the lighting field from the design, and the
# correction cancels the motif's own contrast.
#
# L is kept at 0.45 rather than 0 because achromatic designs (grey/black/
# white on white) carry all their signal in luminance.
LAB_WEIGHTS = (0.45, 1.0, 1.0)


def flatten_illumination(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Remove a smooth lighting gradient while preserving design detail.

    Divides the image by a heavily blurred copy of itself (a homomorphic /
    rolling-ball style correction). The blur radius is tied to image size so
    it captures the illumination field but not the motifs themselves.

    strength: 0.0 = no correction, 1.0 = full correction.
    """
    if strength <= 0:
        return image

    rgb = image.convert('RGB')
    w, h = rgb.size
    # Radius must be much larger than any design feature, so scale with the
    # short edge. 1/8 of the short edge reliably exceeds motif size.
    radius = max(8, int(min(w, h) / 8))

    src = np.asarray(rgb, dtype=np.float32)
    bg = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    bg = np.maximum(bg, 1.0)

    # Divide out the field, then re-centre on the image's own mean brightness
    # so the result keeps a natural exposure rather than washing out.
    corrected = src / bg * float(src.mean())
    out = src * (1.0 - strength) + corrected * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _srgb_to_lab(arr_uint8: np.ndarray) -> np.ndarray:
    """Convert an (H,W,3) uint8 sRGB array to CIELAB float32."""
    from skimage.color import rgb2lab
    return rgb2lab(arr_uint8.astype(np.float32) / 255.0).astype(np.float32)


def _superpixel_means(lab: np.ndarray, n_segments: int, feat: np.ndarray = None):
    """
    Segment with SLIC and return (segment_ids, mean_feature_per_segment, sizes).

    SLIC runs on true LAB (its compactness term assumes real geometry), but the
    per-segment means are taken from `feat` — the weighted space that KMeans
    will cluster in.

    Falls back to per-pixel treatment if skimage.segmentation is unavailable,
    so the module degrades gracefully rather than failing.
    """
    try:
        from skimage.segmentation import slic
    except Exception:
        return None, None, None

    seg = slic(lab, n_segments=n_segments, compactness=6.0,
               channel_axis=2, convert2lab=False, start_label=0,
               enforce_connectivity=True)
    if feat is None:
        feat = lab
    n_seg = int(seg.max()) + 1
    flat = seg.ravel()
    sizes = np.bincount(flat, minlength=n_seg).astype(np.float32)
    sizes = np.maximum(sizes, 1.0)

    means = np.empty((n_seg, 3), dtype=np.float32)
    for c in range(3):
        means[:, c] = np.bincount(flat, weights=feat[:, :, c].ravel(),
                                  minlength=n_seg) / sizes
    return seg, means, sizes


def _pool_labels(label_hi: np.ndarray, n_classes: int, k: int):
    """
    Pool a hi-res label map down by an integer factor k.

    Returns (winner, coverage) where:
      winner   : (H,W) int  — majority-vote class per output cell
      coverage : (n_classes,H,W) float — fraction of each cell held by class c

    This is the core of the thin-line fix: it measures how much of each output
    cell each class actually occupies, rather than interpolating colours and
    re-guessing afterwards.
    """
    H, W = label_hi.shape[0] // k, label_hi.shape[1] // k
    coverage = np.empty((n_classes, H, W), dtype=np.float32)
    cell = float(k * k)
    for c in range(n_classes):
        m = (label_hi == c).astype(np.float32)
        # Sum over each k x k block.
        coverage[c] = m.reshape(H, k, W, k).sum(axis=(1, 3)) / cell
    winner = coverage.argmax(axis=0).astype(np.int32)
    return winner, coverage


def _thin_rescue(winner, coverage, bg_class, order):
    """
    Reinstate design cells that majority vote would have dropped.

    A 1px line at k=3 covers only ~33% of an output cell, so a background that
    holds the other 67% wins the vote and the line disappears. Here, any cell
    where a design class holds at least THIN_RESCUE_COVERAGE and the current
    winner is background is handed to that design class.

    `order` lists design classes smallest-area-first, so fine detail claims
    contested cells before large fills do.
    """
    out = winner.copy()
    for c in order:
        if c == bg_class:
            continue
        contested = (out == bg_class) & (coverage[c] >= THIN_RESCUE_COVERAGE)
        out[contested] = c
    return out


def detect_colors_smart(image: Image.Image,
                        n_colors: int,
                        pins: int,
                        cards: int,
                        flatten_light: bool = False,
                        superpixels: bool = False,
                        thin_rescue: bool = True,
                        hires_factor: int = DEFAULT_HIRES_FACTOR):
    """
    High-accuracy replacement for bmp_engine.detect_colors.

    Unlike the legacy function, this takes the FULL-RESOLUTION image plus the
    loom target size, because detecting before downsampling is the whole point.

    Returns the same 4-tuple as detect_colors, so it is a drop-in at the call
    site:
        colors        : list of (R,G,B) tuples, most dominant first
        counts        : list of int pixel counts at loom resolution
        label_map     : (cards x pins) uint8 array
        genuine_flags : list of bool — False marks a likely artefact cluster
    """
    n_colors = max(1, int(n_colors))

    # Choose the working factor from the SOURCE resolution. Downscaling below
    # native res before clustering is exactly the bug this module exists to
    # fix, so k is set to keep as much of the original detail as the memory
    # cap allows.
    if hires_factor and int(hires_factor) > 0:
        k = int(hires_factor)
    else:
        src_w, src_h = image.size
        k = int(max(2, min(src_w / max(pins, 1), src_h / max(cards, 1))))
    k = max(1, min(k, MAX_HIRES_FACTOR))
    while k > 1 and (pins * k) * (cards * k) > MAX_HIRES_PIXELS:
        k -= 1

    hi_w, hi_h = pins * k, cards * k
    work = image.convert('RGB').resize((hi_w, hi_h), Image.LANCZOS)

    if flatten_light:
        work = flatten_illumination(work)

    arr = np.asarray(work, dtype=np.uint8)
    lab = _srgb_to_lab(arr)

    # Weighted LAB: see LAB_WEIGHTS. Clustering runs in the weighted space;
    # the representative colours reported back to the UI are still computed
    # from the true RGB pixels of each cluster.
    lab_w = lab * np.asarray(LAB_WEIGHTS, dtype=np.float32)

    # ── Cluster ─────────────────────────────────────────────────────────────
    seg = means = sizes = None
    if superpixels and n_colors > 1:
        n_segments = int((hi_w * hi_h) / (TARGET_SUPERPIXEL_PX ** 2))
        n_segments = max(n_colors * 40, min(n_segments, MAX_SUPERPIXELS))
        seg, means, sizes = _superpixel_means(lab, n_segments, lab_w)

    if seg is not None:
        km = KMeans(n_clusters=min(n_colors, means.shape[0]),
                    random_state=42, n_init=10, max_iter=300)
        km.fit(means, sample_weight=sizes)
        seg_labels = km.predict(means)
        label_hi = seg_labels[seg].astype(np.int32)
    else:
        flat = lab_w.reshape(-1, 3)
        if flat.shape[0] > KMEANS_FIT_SAMPLES:
            idx = np.random.default_rng(42).choice(
                flat.shape[0], KMEANS_FIT_SAMPLES, replace=False)
            fit_data = flat[idx]
        else:
            fit_data = flat
        km = KMeans(n_clusters=n_colors, random_state=42, n_init=4, max_iter=100)
        km.fit(fit_data)
        label_hi = km.predict(flat).reshape(hi_h, hi_w).astype(np.int32)

    n_found = int(label_hi.max()) + 1

    # ── Representative RGB per cluster (mean of its actual pixels) ───────────
    rgb_means = np.zeros((n_found, 3), dtype=np.float32)
    flat_lbl = label_hi.ravel()
    px_counts = np.bincount(flat_lbl, minlength=n_found).astype(np.float32)
    px_counts = np.maximum(px_counts, 1.0)
    for c in range(3):
        rgb_means[:, c] = np.bincount(
            flat_lbl, weights=arr[:, :, c].ravel().astype(np.float32),
            minlength=n_found) / px_counts

    # ── Pool to loom resolution ─────────────────────────────────────────────
    winner, coverage = _pool_labels(label_hi, n_found, k)

    hi_area = np.bincount(flat_lbl, minlength=n_found)
    bg_class = int(hi_area.argmax())          # background = largest class
    if thin_rescue and k > 1 and n_found > 1:
        design_order = [c for c in np.argsort(hi_area) if c != bg_class]
        winner = _thin_rescue(winner, coverage, bg_class, design_order)

    # ── Sort by dominance at loom resolution, background first ──────────────
    final_counts = np.bincount(winner.ravel(), minlength=n_found)
    order = np.argsort(-final_counts)

    remap = np.zeros(n_found, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    label_map = remap[winner].astype(np.uint8)

    colors = [tuple(int(v) for v in rgb_means[i]) for i in order]
    counts = [int(final_counts[i]) for i in order]

    # ── Genuine-colour flags (same rule as the legacy path) ─────────────────
    genuine_flags, confirmed = [], []
    for rgb in colors:
        ok = _is_genuine_colour(rgb, confirmed) if confirmed else True
        genuine_flags.append(bool(ok))
        if ok:
            confirmed.append(rgb)

    return colors, counts, label_map, genuine_flags
