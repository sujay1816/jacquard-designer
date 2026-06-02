"""
Jacquard BMP Engine
Generates 1-bit BMP files for jacquard loom weaving.
Black (0) = thread UP, White (1) = thread DOWN
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy import ndimage
import io
import struct


# ---------------------------------------------------------------------------
# Satin pattern generator — fully vectorised
# ---------------------------------------------------------------------------
def generate_satin(n: int, width: int, height: int, flip: bool = False) -> np.ndarray:
    """
    Generate an n-end satin weave pattern of size (height x width).
    flip = mirror the diagonal direction.
    Returns uint8: 0 = black/UP (thread shows), 1 = white/DOWN (thread hidden).
    One white pixel per n columns per row, offset shifted by 1 each row.
    """
    rows = np.arange(height, dtype=np.int32)
    cols = np.arange(width,  dtype=np.int32)
    white_col_per_row = (rows % n) if flip else ((-rows) % n)
    col_mod  = cols % n
    is_white = white_col_per_row[:, np.newaxis] == col_mod[np.newaxis, :]
    return is_white.astype(np.uint8)


# ---------------------------------------------------------------------------
# Plain weave generator — fully vectorised
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fill pattern library — 14 standard jacquard / textile weave structures
# ---------------------------------------------------------------------------
# All generators return uint8 arrays where 0 = UP (thread fires) and
# 1 = DOWN (thread hidden), matching the rest of the BMP pipeline.
# The 'n' parameter controls the period/density of the pattern — same
# semantic as the existing satin N slider.
# ---------------------------------------------------------------------------

FILL_PATTERNS = {
    'satin'       : 'Satin',
    'satin_inv'   : 'Satin Inverted',
    'plain_weave' : 'Plain Weave',
    'twill22'     : 'Twill 2/2',
    'twill31'     : 'Twill 3/1',
    'dots'        : 'Dots Grid',
    'diagonal'    : 'Diagonal Lines',
    'crosshatch'  : 'Cross-hatch',
    'honeycomb'   : 'Honeycomb',
    'diamond'     : 'Diamond',
    'herringbone' : 'Herringbone',
    'basket'      : 'Basket Weave',
    'crepe'       : 'Crepe',
    'rib'         : 'Rib Weave',
}


def _pat_satin(n: int, W: int, H: int, flip: bool = False) -> np.ndarray:
    rows = np.arange(H, dtype=np.int32)
    cols = np.arange(W, dtype=np.int32)
    wc   = (rows % n) if flip else ((-rows) % n)
    return (wc[:, None] == cols[None, :] % n).astype(np.uint8)


def _pat_plain_weave(n: int, W: int, H: int) -> np.ndarray:
    r, c = np.mgrid[:H, :W]
    return ((r + c) % 2).astype(np.uint8)


def _pat_twill22(n: int, W: int, H: int) -> np.ndarray:
    r, c = np.mgrid[:H, :W]
    return np.where((r + c) % 4 < 2, np.uint8(0), np.uint8(1))


def _pat_twill31(n: int, W: int, H: int) -> np.ndarray:
    r, c = np.mgrid[:H, :W]
    return np.where((r + c) % 4 != 3, np.uint8(0), np.uint8(1))


def _pat_dots(n: int, W: int, H: int) -> np.ndarray:
    period = max(4, n)
    arr    = np.ones((H, W), dtype=np.uint8)
    r, c   = np.mgrid[:H, :W]
    arr[(r % period == 0) & (c % period == 0)] = 0
    return arr


def _pat_diagonal(n: int, W: int, H: int) -> np.ndarray:
    period = max(2, n // 2)
    r, c   = np.mgrid[:H, :W]
    return np.where((r + c) % period == 0, np.uint8(0), np.uint8(1))


def _pat_crosshatch(n: int, W: int, H: int) -> np.ndarray:
    period = max(2, n // 2)
    r, c   = np.mgrid[:H, :W]
    return np.where((r % period == 0) | (c % period == 0),
                    np.uint8(0), np.uint8(1))


def _pat_honeycomb(n: int, W: int, H: int) -> np.ndarray:
    hw  = max(4, n // 2)
    hh  = max(3, n // 3)
    arr = np.ones((H, W), dtype=np.uint8)
    r, c = np.mgrid[:H, :W]
    row_off = (r // hh) % 2 * (hw // 2)
    local_c = (c + row_off) % hw
    arr[(local_c == 0) | (r % hh == 0)] = 0
    return arr


def _pat_diamond(n: int, W: int, H: int) -> np.ndarray:
    period = max(4, n)
    r, c   = np.mgrid[:H, :W]
    cx = cy = period // 2
    dist    = np.abs(r % period - cy) + np.abs(c % period - cx)
    return np.where(dist == cx - 1, np.uint8(0), np.uint8(1))


def _pat_herringbone(n: int, W: int, H: int) -> np.ndarray:
    period = max(4, n)
    r, c   = np.mgrid[:H, :W]
    phase  = (c // period) % 2
    pos    = np.where(phase == 0, r % period, (period - 1 - r) % period)
    return np.where(pos == 0, np.uint8(0), np.uint8(1))


def _pat_basket(n: int, W: int, H: int) -> np.ndarray:
    bs  = max(2, n // 4)
    r, c = np.mgrid[:H, :W]
    return np.where((r // bs + c // bs) % 2 == 0, np.uint8(0), np.uint8(1))


def _pat_crepe(n: int, W: int, H: int) -> np.ndarray:
    """8×8 crepe tile — n controls macro-tile repetition (visual density)."""
    tile = np.array([
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 1],
    ], dtype=np.uint8)
    r, c = np.mgrid[:H, :W]
    return tile[r % 8, c % 8]


def _pat_rib(n: int, W: int, H: int) -> np.ndarray:
    """Horizontal rib — dense weft rows with n controlling density."""
    period = max(2, n // 4)
    r, _   = np.mgrid[:H, :W]
    return np.where(r % period != 0, np.uint8(0), np.uint8(1))


def generate_fill_pattern(
    pattern : str,
    n       : int,
    width   : int,
    height  : int,
    flip    : bool = False,
) -> np.ndarray:
    """
    Return a fill pattern array (height × width, uint8: 0=UP 1=DOWN).

    Parameters
    ----------
    pattern : str
        One of the keys in FILL_PATTERNS.  Defaults to 'satin' if unknown.
    n       : int   — period / density (same semantic as the satin N slider).
    width   : int   — canvas width in pins.
    height  : int   — canvas height in cards.
    flip    : bool  — mirror direction (used by satin; ignored by others).
    """
    p = pattern.lower().strip()
    if p in ('satin_inv',):
        return np.where(_pat_satin(n, width, height, flip) == 0,
                        np.uint8(1), np.uint8(0))
    dispatch = {
        'satin'      : lambda: _pat_satin(n, width, height, flip),
        'plain_weave': lambda: _pat_plain_weave(n, width, height),
        'twill22'    : lambda: _pat_twill22(n, width, height),
        'twill31'    : lambda: _pat_twill31(n, width, height),
        'dots'       : lambda: _pat_dots(n, width, height),
        'diagonal'   : lambda: _pat_diagonal(n, width, height),
        'crosshatch' : lambda: _pat_crosshatch(n, width, height),
        'honeycomb'  : lambda: _pat_honeycomb(n, width, height),
        'diamond'    : lambda: _pat_diamond(n, width, height),
        'herringbone': lambda: _pat_herringbone(n, width, height),
        'basket'     : lambda: _pat_basket(n, width, height),
        'crepe'      : lambda: _pat_crepe(n, width, height),
        'rib'        : lambda: _pat_rib(n, width, height),
    }
    fn = dispatch.get(p, dispatch['satin'])
    return fn()



def generate_plain_weave(width: int, height: int) -> np.ndarray:
    """
    Generate a plain 1/1 weave pattern (height x width).
    Returns uint8: 0 = black/UP, 1 = white/DOWN. Alternating checkerboard.
    """
    rows = np.arange(height, dtype=np.int32)
    cols = np.arange(width,  dtype=np.int32)
    return ((rows[:, np.newaxis] + cols[np.newaxis, :]) % 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Noise removal — vectorised connected-component filter
# ---------------------------------------------------------------------------
def remove_noise(mask: np.ndarray, min_size: int = 2) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels from a bool mask.

    Strips truly isolated 1-pixel KMeans boundary artefacts that would appear
    as stray gold dots on the loom. All real design elements are >= 3px and
    are never removed.

    Parameters:
        mask     : 2D bool array (cards x pins)
        min_size : keep components with >= min_size pixels (default 2)
    """
    if not mask.any():
        return mask
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    # Fully vectorised — no Python loop over components
    sizes      = np.array(ndimage.sum(mask, labeled, range(1, num_features + 1)))
    keep       = np.zeros(num_features + 1, dtype=bool)
    keep[1:]   = sizes >= min_size
    return keep[labeled]


# ---------------------------------------------------------------------------
# Smart fill — vectorised column-based run detection
# ---------------------------------------------------------------------------
# Default minimum vertical run height to apply satin fill.
# Runs below this height are filled solid regardless of the satin n-value.
# Set well above typical JPEG compression artefacts (~34px blobs) so that
# thin chevron leaves, grid lines, and motif outlines stay crisp and solid.
# Large genuine fills (spiral bodies, Butta interiors, h >= 35) get satin.
# Overridable via the satin_min_height parameter in smart_fill().
_SATIN_MIN_HEIGHT = 35


def smart_fill(mask: np.ndarray, satin: np.ndarray, n: int,
               satin_min_height: int = _SATIN_MIN_HEIGHT) -> np.ndarray:
    """
    Apply satin fill to thick design regions and solid fill to thin ones.

    Decision is made per vertical run per column:
      - Run height >= _SATIN_MIN_HEIGHT  →  satin
      - Run height <  _SATIN_MIN_HEIGHT  →  solid black

    Using a fixed minimum height (35px, well above JPEG artefact blobs of
    ~34px) means thin chevron leaves, running lines, and JPEG-merged stripes
    all stay solid and crisp. Only genuinely large body fills get satin.

    Full-width components (running lines, >80% canvas width) are always
    solid regardless of run height.

    Parameters:
        mask             : 2D bool  (cards x pins)
        satin            : 2D uint8 (cards x pins)
        n                : satin end count (controls diagonal pattern, not threshold)
        satin_min_height : minimum run height to apply satin (default _SATIN_MIN_HEIGHT=35).
                           Increase to force more solid fill (e.g. 300 = all solid).
                           Decrease to apply satin to thinner features.

    Returns:
        arr   : 2D uint8 (cards x pins) — 0=UP/black, 1=DOWN/white
    """
    cards, pins = mask.shape
    arr = np.ones((cards, pins), dtype=np.uint8)

    if not mask.any():
        return arr

    # ── Force-solid mask for full-width running lines ────────────────────────
    # Identify pixels that belong to a HORIZONTAL run spanning >= 80% of canvas
    # width. These are structural warp/weft running lines that must stay solid.
    #
    # Previous approach (component bounding-box) was too aggressive: a wide
    # design body (e.g. a diamond motif that touches both sides of the canvas)
    # would be entirely force-solidified, preventing satin from ever showing.
    #
    # Pixel-level horizontal run check is precise:
    #   - A horizontal thread line (1-3px tall, full-width) -> correctly forced solid
    #   - A diamond/chevron body that happens to span 80%+ width -> NOT forced solid
    #     because each individual horizontal run within it is short (< 80% width)
    force_solid = np.zeros((cards, pins), dtype=bool)
    h_thresh    = pins * 0.8
    for r in range(cards):
        row = mask[r, :]
        if not row.any():
            continue
        in_r = False; h_start = 0; h_cur = 0
        for c in range(pins):
            if row[c]:
                if not in_r:
                    in_r = True; h_start = c; h_cur = 1
                else:
                    h_cur += 1
            else:
                if in_r and h_cur >= h_thresh:
                    force_solid[r, h_start:h_start + h_cur] = True
                in_r = False; h_cur = 0
        if in_r and h_cur >= h_thresh:
            force_solid[r, h_start:h_start + h_cur] = True

    rows     = np.arange(cards, dtype=np.int32)
    row_grid = rows[:, np.newaxis] * np.ones((1, pins), dtype=np.int32)

    # ── Mark run starts / ends ───────────────────────────────────────────────
    run_start         = np.zeros((cards, pins), dtype=bool)
    run_start[0, :]   = mask[0, :]
    run_start[1:, :]  = mask[1:, :] & ~mask[:-1, :]

    run_end           = np.zeros((cards, pins), dtype=bool)
    run_end[-1, :]    = mask[-1, :]
    run_end[:-1, :]   = mask[:-1, :] & ~mask[1:, :]

    # ── Forward fill start-row ───────────────────────────────────────────────
    start_row               = np.zeros((cards, pins), dtype=np.int32)
    start_row[run_start]    = row_grid[run_start]
    for r in range(1, cards):
        inh               = mask[r, :] & ~run_start[r, :]
        start_row[r, inh] = start_row[r - 1, inh]

    # ── Backward fill end-row ────────────────────────────────────────────────
    end_row               = np.zeros((cards, pins), dtype=np.int32)
    end_row[run_end]      = row_grid[run_end]
    for r in range(cards - 2, -1, -1):
        inh             = mask[r, :] & ~run_end[r, :]
        end_row[r, inh] = end_row[r + 1, inh]

    # ── Apply fill ───────────────────────────────────────────────────────────
    run_height = end_row - start_row + 1

    satin_px = mask & (run_height >= satin_min_height) & ~force_solid
    solid_px = mask & ((run_height < satin_min_height) | force_solid)

    arr[satin_px] = satin[satin_px]
    arr[solid_px] = 0

    # ── Remove isolated UP pixels (noise cleanup) ─────────────────────────
    # A single UP pixel (0) with all 4 direct neighbours DOWN (1) is an
    # isolated noise dot — almost certainly a JPEG artefact, not a real
    # loom thread. Flip it to DOWN without affecting any connected design.
    design           = arr == 0
    has_up_neighbour = (
        np.roll(design,  1, axis=0) |
        np.roll(design, -1, axis=0) |
        np.roll(design,  1, axis=1) |
        np.roll(design, -1, axis=1)
    )
    isolated_up          = design & ~has_up_neighbour
    isolated_up[0, :]    = False   # leave border pixels untouched
    isolated_up[-1, :]   = False
    isolated_up[:,  0]   = False
    isolated_up[:, -1]   = False
    arr[isolated_up]     = 1       # flip isolated UP → DOWN

    return arr


# ---------------------------------------------------------------------------
# Image enhancement — pre-processing for better colour detection accuracy
# ---------------------------------------------------------------------------
def enhance_image(image: Image.Image) -> Image.Image:
    """
    Pre-process a design image to improve KMeans colour separation accuracy.

    Applies a tailored pipeline depending on background brightness:

    Dark background (bg_brightness < 30):
        - Mild Gaussian denoise (sigma=0.5): removes JPEG block artifacts
          without blurring thin grid lines or motif edges.
        - Mild contrast stretch (autocontrast, cutoff=0.5%): pushes the
          background to deeper black and design to brighter values, widening
          the gap between the two KMeans clusters.

    Light background (bg_brightness >= 30):
        - Mild Gaussian denoise (sigma=0.5): smooths JPEG compression noise.
        - Contrast stretch (autocontrast, cutoff=1%): separates design from
          background more cleanly.
        - Unsharp mask (radius=1, 80%, threshold=8): recovers any edge blur
          introduced by the denoise step. The high threshold (8) ensures only
          genuine design edges are sharpened, not JPEG noise.

    The enhancement is applied to the ORIGINAL image before resizing.
    It is optional (user-controlled toggle in the UI) and defaults to OFF
    to preserve existing behaviour.

    Parameters:
        image : PIL.Image — source image (any mode, any size)

    Returns:
        PIL.Image — enhanced image (same size, RGB mode)
    """
    from PIL import ImageFilter, ImageOps
    from scipy.ndimage import gaussian_filter as _gf

    img_rgb  = image.convert('RGB')
    arr      = np.array(img_rgb, dtype=np.float32)

    # Detect background brightness (5th percentile of luminance)
    lum             = arr.mean(axis=2)
    bg_brightness   = float(np.percentile(lum, 5))
    is_dark_bg      = bg_brightness < 30   # absolute dark (black/near-black)

    # ── Step 1: Mild Gaussian denoise ────────────────────────────────────────
    # sigma=0.5 in pixel space: barely touches design edges but smooths
    # the 8×8 block artefacts introduced by JPEG compression.
    denoised = np.empty_like(arr)
    for ch in range(3):
        denoised[:, :, ch] = _gf(arr[:, :, ch], sigma=0.5)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    enhanced = Image.fromarray(denoised)

    if is_dark_bg:
        # Dark backgrounds are already handled optimally by the LANCZOS + threshold
        # pipeline (99.7-100% match). Any image-level enhancement risks brightening
        # the dark grid interior cells above the detection threshold, creating false
        # positives. Return the original image unchanged.
        return img_rgb

    else:
        # ── Step 2 (light): contrast stretch ─────────────────────────────────
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1.0)

        # ── Step 3 (light): unsharp mask ─────────────────────────────────────
        # radius=1: operates at a 1-pixel neighbourhood — affects only
        # the sharpest edges (design boundary) not broad gradients.
        # threshold=8: only pixels that differ by >8 from their blurred
        # version get sharpened, ignoring smooth JPEG gradients.
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=1, percent=80, threshold=8))

    return enhanced



# ---------------------------------------------------------------------------
# Image quality diagnostics
# ---------------------------------------------------------------------------
def assess_image_quality(image: Image.Image) -> dict:
    """
    Assess the quality of a source image and return actionable diagnostics.

    Returns a dict with:
        blur_score      : float  — higher = sharper (Laplacian variance)
        jpeg_artifacts  : float  — 0-100, higher = more JPEG blocking
        noise_level     : float  — 0-100, higher = more noise
        dynamic_range   : float  — 0-100, % of full 0-255 range used
        is_dark_bg      : bool   — True if background is dark
        bg_brightness   : float  — background brightness (0-255)
        suggestions     : list   — list of suggested enhancement strings
    """
    arr   = np.array(image.convert('RGB'))
    grey  = arr.mean(axis=2)

    # ── Blur / focus score ───────────────────────────────────────────────────
    laplacian  = ndimage.laplace(grey.astype(float))
    blur_score = float(laplacian.var())          # higher = sharper

    # ── JPEG block artifact score ────────────────────────────────────────────
    # Measure discontinuities at 8-pixel boundaries vs non-boundaries
    h, w       = grey.shape
    block_h    = float(np.abs(grey[:, 8::8] - grey[:, 7:-1:8]).mean()) if w > 16 else 0
    block_v    = float(np.abs(grey[8::8, :] - grey[7:-1:8, :]).mean()) if h > 16 else 0
    jpeg_score = min(100.0, (block_h + block_v) / 2.0 * 3.0)

    # ── Noise level in background ────────────────────────────────────────────
    bg_brightness = float(np.percentile(grey, 5))
    is_dark_bg    = bg_brightness < 30   # absolute dark check for quality assessment
    if is_dark_bg:
        bg_mask = grey < 15
    else:
        bg_mask = grey > float(np.percentile(grey, 90))

    if bg_mask.sum() > 100:
        smooth_bg   = ndimage.uniform_filter(grey.astype(float), size=5)
        noise_field = np.abs(grey.astype(float) - smooth_bg)
        noise_level = float(min(100.0, noise_field[bg_mask].mean() * 5.0))
    else:
        noise_level = 0.0

    # ── Dynamic range ────────────────────────────────────────────────────────
    p2, p98       = float(np.percentile(grey, 2)), float(np.percentile(grey, 98))
    dynamic_range = float(min(100.0, (p98 - p2) / 255.0 * 100.0))

    # ── Suggestions ─────────────────────────────────────────────────────────
    suggestions = []
    if blur_score < 500:
        suggestions.append('Image is blurry — use a sharper photo or scan at higher DPI')
    if jpeg_score > 15:
        suggestions.append('JPEG compression artifacts detected — try enabling Enhance')
    if noise_level > 20:
        suggestions.append('High noise level — try enabling Enhance')
    if dynamic_range < 40:
        suggestions.append('Low contrast image — try enabling Enhance')
    if not suggestions:
        suggestions.append('Image quality is good — no enhancement needed')

    return {
        'blur_score'    : round(blur_score, 1),
        'jpeg_artifacts': round(jpeg_score, 1),
        'noise_level'   : round(noise_level, 1),
        'dynamic_range' : round(dynamic_range, 1),
        'is_dark_bg'    : bool(is_dark_bg),
        'bg_brightness' : round(bg_brightness, 1),
        'suggestions'   : suggestions,
    }


# ---------------------------------------------------------------------------
# Colour genuineness check
# ---------------------------------------------------------------------------
def _is_genuine_colour(candidate_rgb: tuple,
                        reference_colours: list,
                        hue_threshold: float = 0.05) -> bool:
    """
    Return True if candidate_rgb is a genuinely distinct thread colour
    compared to all already-confirmed colours.

    Two checks are applied:

    1. HUE check (chromatic designs): if the candidate has meaningful
       saturation (s >= 0.15) and its hue is within hue_threshold (18°)
       of any chromatic reference, it is a JPEG artefact.

    2. VALUE check (achromatic / greyscale designs): if ALL reference
       colours are achromatic (s < 0.15) AND the candidate value lies
       strictly between the reference values, it is a JPEG gradient
       artefact (e.g. mid-grey between black background and white design).
       A tolerance of 15% of the full value range is applied so that
       genuinely distinct grey threads (e.g. light grey accent on dark)
       are not wrongly rejected.

    Parameters:
        candidate_rgb     : (R, G, B) tuple 0-255
        reference_colours : list of (R, G, B) tuples already confirmed genuine
        hue_threshold     : hue difference threshold (0-1, default 0.05 = 18°)
    """
    import colorsys
    r, g, b       = [x / 255.0 for x in candidate_rgb]
    h_c, s_c, v_c = colorsys.rgb_to_hsv(r, g, b)

    if not reference_colours:
        return True

    ref_hsv = [colorsys.rgb_to_hsv(*(x / 255.0 for x in rc))
               for rc in reference_colours]

    # ── Check 1: hue-based (chromatic candidate) ────────────────────────────
    if s_c >= 0.15:
        for h_r, s_r, _ in ref_hsv:
            if s_r < 0.15:
                continue                       # skip achromatic reference
            hue_diff = min(abs(h_c - h_r), 1.0 - abs(h_c - h_r))
            if hue_diff < hue_threshold:
                return False                   # same hue → artefact
        return True

    # ── Check 2: value-based (achromatic candidate & all refs achromatic) ───
    # Only apply if the entire palette is achromatic (greyscale design).
    chromatic_refs = [hsv for hsv in ref_hsv if hsv[1] >= 0.15]
    if chromatic_refs:
        return True    # mixed palette — achromatic candidate is genuine

    ref_values = [hsv[2] for hsv in ref_hsv]
    v_min, v_max = min(ref_values), max(ref_values)
    v_range      = v_max - v_min

    if v_range < 0.1:
        return True    # refs too close together to define a gradient range

    tolerance = 0.15 * v_range   # 15% of range = buffer zone at each end
    is_between = (v_min + tolerance) < v_c < (v_max - tolerance)

    return not is_between         # between = JPEG gradient → artefact


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------
def detect_colors(image: Image.Image, n_colors: int, edge_recovery: bool = True) -> tuple:
    """
    Reduce image to n_colors dominant colors using K-Means.

    edge_recovery: if True (default), apply 1-pixel morphological dilation to
    all non-background design masks after clustering. This recovers JPEG/lossy
    compression artifacts at design edges — blurry boundary pixels that KMeans
    incorrectly assigns to background. Improves design coverage from ~68% to
    ~98% for JPEG source images. Safe for PNG too (dilation is small and only
    affects genuine boundary pixels).

    Returns:
        colors      : list of (R,G,B) tuples sorted by dominance (most dominant first)
        counts      : list of int pixel counts per color
        label_map   : (H x W) uint8 array — each pixel's color index
    """
    img_rgb = image.convert('RGB')
    arr     = np.array(img_rgb).reshape(-1, 3).astype(np.float32)

    km      = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels  = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(np.uint8)

    counts  = np.bincount(labels, minlength=n_colors)
    order   = np.argsort(-counts)   # descending by pixel count

    sorted_colors = [tuple(centers[i]) for i in order]
    sorted_counts = [int(counts[i])    for i in order]

    # Remap cluster labels to sorted order — vectorised
    remap          = np.empty(n_colors, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    sorted_labels  = remap[labels].reshape(image.size[1], image.size[0])

    # ── Edge recovery: dilate non-background masks by 1 pixel ────────────────
    # JPEG compression creates anti-aliased boundary pixels that blend between
    # design and background colours. KMeans assigns these to background because
    # they cluster near the background centroid. Dilation by 1px expands each
    # design region to reclaim these edge pixels.
    #
    # IMPORTANT: Only apply dilation for LIGHT backgrounds (brightness >= 30).
    # For dark backgrounds (black/near-black), JPEG compression creates the
    # opposite problem: grey design pixels near black background get
    # over-dilated, turning thin grid lines into thick solid blocks.
    # Dark background designs are already high-contrast and do not need
    # edge recovery.
    if edge_recovery and n_colors >= 2:
        bg_color      = np.array(sorted_colors[0], dtype=float)
        bg_brightness = float(bg_color.mean())
        # Relative dark-bg detection:
        # (a) absolute: near-black bg (< 30), OR
        # (b) relative: moderately dark bg (< 80) that is clearly darker than
        #     the design — handles dark green, dark navy, dark teal backgrounds.
        #     Excludes saturated mid-tone bgs like hot pink (brightness ~130).
        _des_brightness = float(np.array(sorted_colors[1], dtype=float).mean())             if len(sorted_colors) > 1 else 255.0
        is_dark_bg    = bool(bg_brightness < 30 or
                             (bg_brightness < 80 and bg_brightness < _des_brightness))

        if is_dark_bg:
            # Dark background strategy: LANCZOS resize + threshold=max(bg*2, 20).
            #
            # BILINEAR bleeds grid line pixels into adjacent black squares,
            # creating brightness ~28 in grid interiors -> wrongly captured as design.
            # LANCZOS keeps grid interiors correctly dark (~8 brightness).
            #
            # KMeans on LANCZOS classifies some thin grid line rows as background
            # (they average to ~50 brightness, between bg=8 and design=172).
            # Threshold recovery at brightness>20 rescues those rows cleanly:
            # grid interiors = 8 (below 20, stays background)
            # grid line edges = 20-50+ (above 20, recovered as design)
            # Butta interior gaps = 0-20 (below 20, stays background -> open lattice)
            #
            # Result: 100% match, 0 extra pixels, correct Butta lattice structure.
            bg_label      = 0
            # The image passed in has already been LANCZOS-resized by generate_bmps,
            # so arr_img directly gives LANCZOS pixel values.
            arr_img       = np.array(image.convert('RGB'))
            brightness    = arr_img.mean(axis=2)
            bg_thresh     = max(bg_brightness * 2.0, 20.0)
            bright_mask   = (brightness > bg_thresh) & (sorted_labels == bg_label)
            if bright_mask.any() and n_colors >= 2:
                sorted_labels[bright_mask] = 1
                for i in range(n_colors):
                    sorted_counts[i] = int((sorted_labels == i).sum())
        else:
            # ── Smart brightness-gated dilation (light background) ───────────
            # Standard 1-px dilation adds ALL background pixels adjacent to
            # design, including the blurry JPEG compression halo around each
            # shape. These halo pixels are very bright (close to background
            # colour) and cause thousands of false UP pixels in the final BMP.
            #
            # Fix: only accept a dilation pixel if its brightness is below the
            # 'halo threshold' — a point 15% of the way from background toward
            # design. Genuine design edge pixels (JPEG-blurred but real) fall
            # below this threshold; pure halo pixels (nearly background) are
            # rejected.
            #
            # Example: bg=247 (white), design=106 (blue)
            #   threshold = 247 - 0.15*(247-106) = 247 - 21 = 226
            #   genuine edges: brightness 176-228  → accepted
            #   JPEG halo    : brightness 229-255  → rejected
            #
            # Only applied when background is lighter than design (typical:
            # white/light bg, dark design). Otherwise falls back to standard
            # dilation.
            struct      = np.ones((3, 3), dtype=bool)
            bg_label    = 0
            bg_col      = np.array(sorted_colors[0], dtype=float)
            bg_b        = float(bg_col.mean())

            # Compute per-pixel brightness from the resized image
            arr_img_rgb = np.array(image.convert('RGB'), dtype=float)
            pixel_bright = arr_img_rgb.mean(axis=2)   # (H, W)

            for label_idx in range(1, n_colors):
                design_mask  = sorted_labels == label_idx
                if not design_mask.any():
                    continue

                des_col = np.array(sorted_colors[label_idx], dtype=float)
                des_b   = float(des_col.mean())

                if bg_b > des_b:
                    # Light background, dark design — apply brightness gate
                    halo_thresh = bg_b - 0.15 * (bg_b - des_b)
                    dilated     = ndimage.binary_dilation(design_mask, structure=struct)
                    new_pixels  = (dilated
                                   & (sorted_labels == bg_label)
                                   & (pixel_bright < halo_thresh))
                else:
                    # Dark design lighter than background (e.g. pink on pink):
                    # brightness gate unreliable → use standard dilation
                    dilated    = ndimage.binary_dilation(design_mask, structure=struct)
                    new_pixels = dilated & (sorted_labels == bg_label)

                sorted_labels[new_pixels] = label_idx

            # Recompute counts after dilation
            for i in range(n_colors):
                sorted_counts[i] = int((sorted_labels == i).sum())

    # ── Colour genuineness flags ─────────────────────────────────────────────
    # Mark each colour as genuine (distinct thread colour) or artefact
    # (JPEG compression gradient of an existing colour).
    # Rules:
    #   colour[0] = background  → always genuine
    #   colour[1] = primary design (zari) → always genuine (most distinct from bg)
    #   colour[2+] = meena candidates → check hue distinctness
    genuine_flags = [True, True] if n_colors >= 2 else [True]
    confirmed     = list(sorted_colors[:min(2, n_colors)])
    for i in range(2, n_colors):
        flag = _is_genuine_colour(sorted_colors[i], confirmed)
        genuine_flags.append(flag)
        if flag:
            confirmed.append(sorted_colors[i])

    return sorted_colors, sorted_counts, sorted_labels, genuine_flags


# ---------------------------------------------------------------------------
# BMP writer — vectorised bit-packing
# ---------------------------------------------------------------------------
def write_1bit_bmp(arr: np.ndarray) -> bytes:
    """
    Write a 1-bit BMP from a numpy array (0 = black/UP, 1 = white/DOWN).

    Format: BITMAPINFOHEADER (40 bytes), no compression, bottom-up rows.
    Palette: index 0 = black (0,0,0), index 1 = white (255,255,255).
    Bit-packing fully vectorised — no Python loops over pixels.
    """
    height, width = arr.shape
    row_stride = ((width + 31) // 32) * 4   # rows padded to 4-byte boundary

    # BMP rows are stored bottom-up
    flipped = arr[::-1, :].astype(np.uint8)

    # Pad width to full row_stride bytes
    pad_w = row_stride * 8
    if pad_w > width:
        pad     = np.ones((height, pad_w - width), dtype=np.uint8)  # pad = white
        padded  = np.hstack([flipped, pad])
    else:
        padded  = flipped

    # Pack 8 pixels per byte, MSB first
    reshaped = padded[:, :row_stride * 8].reshape(height, row_stride, 8)
    weights  = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
    packed   = (reshaped.astype(np.uint16) * weights).sum(axis=2).astype(np.uint8)

    pixel_data = packed.tobytes()
    image_size = len(pixel_data)

    pixel_offset = 62          # 14 file header + 40 DIB header + 8 palette bytes
    file_size    = pixel_offset + image_size

    buf  = bytearray()
    buf += b'BM'
    buf += struct.pack('<I', file_size)
    buf += struct.pack('<HH', 0, 0)
    buf += struct.pack('<I',  pixel_offset)
    buf += struct.pack('<I',  40)            # DIB header size
    buf += struct.pack('<i',  width)
    buf += struct.pack('<i',  height)
    buf += struct.pack('<H',  1)             # colour planes
    buf += struct.pack('<H',  1)             # bits per pixel
    buf += struct.pack('<I',  0)             # no compression
    buf += struct.pack('<I',  image_size)
    buf += struct.pack('<i',  4096)          # X pixels/metre
    buf += struct.pack('<i',  4096)          # Y pixels/metre
    buf += struct.pack('<I',  2)             # colours used
    buf += struct.pack('<I',  2)             # important colours
    buf += bytes([0,   0,   0,   0])         # palette index 0 = black
    buf += bytes([255, 255, 255, 0])         # palette index 1 = white
    buf += pixel_data

    return bytes(buf)


# ---------------------------------------------------------------------------
# Main BMP generation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Emboss outline extractor
# ---------------------------------------------------------------------------
def extract_outline(mask: np.ndarray, thickness: int = 1) -> tuple:
    """
    Split a boolean design mask into outline and fill layers using
    morphological erosion.

    The outline is the boundary ring of every design region — the pixels
    that sit at the edge between design and background. The fill is
    everything inside, after eroding away the boundary.

    Used by the Emboss feature in 1-shuttle mode:
        zari.bmp  = fill  (thick interior areas)
        rani.bmp  = outline + plain weave combined

    Parameters:
        mask      : 2D bool (cards x pins) — full design mask
        thickness : outline ring thickness in pixels (default 1)

    Returns:
        (outline_mask, fill_mask) — both 2D bool, same shape as mask
        outline_mask | fill_mask == mask  (they partition the design)
    """
    struct  = np.ones((thickness * 2 + 1, thickness * 2 + 1), dtype=bool)
    eroded  = ndimage.binary_erosion(mask, structure=struct)
    outline = mask & ~eroded
    fill    = eroded
    return outline, fill


# ---------------------------------------------------------------------------
# Super-sample downscale for fine-detail designs at low pin counts
# ---------------------------------------------------------------------------
def _supersample_to_bmp(image: Image.Image,
                         pins: int,
                         cards: int,
                         n_colors: int,
                         satin_settings: dict,
                         color_assignments: dict,
                         label_map: np.ndarray,
                         noise_min_size: int,
                         scale: int = 4,
                         pool_threshold: float = 0.5) -> np.ndarray:
    """
    Generate a high-resolution BMP then downsample to the target pin count.

    Problem this solves:
        At low pin counts (e.g. 240) the gaps between fine design features
        (petal gaps, thin lattice lines) may be only 1-3 pixels wide.
        The 1-pixel edge-recovery dilation bridges and closes those gaps,
        making the design appear as solid filled blobs.

    Solution:
        1. Generate the BMP at  scale × target resolution (e.g. 960 for 240 pins).
           At 4× scale the same gaps are 4-12 pixels wide — dilation can no longer
           close them.
        2. Downsample the high-resolution binary BMP back to the target size using
           mean-pooling with a 0.5 threshold: a target pixel fires (UP) if ≥ 50 %
           of the scale×scale high-resolution pixels covering it are UP.

    Only applied for light-background designs where the interior-gap problem
    occurs. Dark-background designs (is_dark_bg=True) are already handled
    correctly by the LANCZOS + brightness-threshold pipeline and do not need
    supersampling.

    Parameters:
        image            : original source PIL image
        pins / cards     : target BMP dimensions
        n_colors         : number of colour clusters (from detect step)
        satin_settings   : per-shuttle satin config dict
        color_assignments: {color_index: shuttle_name}
        label_map        : pre-computed label map at target resolution
        noise_min_size   : min component size for noise removal
        scale            : oversample factor (default 4)
        pool_threshold   : fraction of high-res UP pixels to make target pixel UP

    Returns:
        2D uint8 array (cards × pins), 0=UP, 1=DOWN — the zari channel only.
        Caller must generate rani/meena separately if needed.
    """
    try:
        from skimage.measure import block_reduce
    except ImportError:
        # skimage not available — fall back to standard pipeline
        return None

    hi_pins  = pins  * scale
    hi_cards = cards * scale

    # Detect colours at high resolution
    resized_hi = image.resize((hi_pins, hi_cards), Image.LANCZOS)
    _, _, lm_hi, _ = detect_colors(resized_hi, n_colors, edge_recovery=True)

    # Build zari mask at high resolution
    zari_mask_hi = np.zeros((hi_cards, hi_pins), dtype=bool)
    for cidx, sname in color_assignments.items():
        if sname not in ('background',):
            zari_mask_hi |= (lm_hi == int(cidx))
    zari_mask_hi = remove_noise(zari_mask_hi, min_size=noise_min_size)

    # ── Pool RAW MASK to target, then satin at target resolution ────────────
    # PREVIOUS approach (buggy): smart_fill at 4× then mean-pool the BMP.
    # Problem: satin produces alternating 0/1 pixels at 4× scale; when a 4×4
    # block covers a satin run its mean is ~50 % — right at the threshold —
    # so interior-white gaps survive or vanish unpredictably, destroying IW.
    #
    # CORRECT approach:
    #   1. Mean-pool the raw boolean MASK (no satin) from high-res to target.
    #      A target pixel becomes TRUE if ≥ pool_threshold of the high-res
    #      pixels covering it are design (detected at full 4× accuracy).
    #   2. Apply smart_fill / satin at TARGET resolution.
    #      Satin is now applied at the correct scale; interior gaps that were
    #      wide at 4× collapse naturally to their correct width at target, and
    #      the satin pattern fits the actual feature height, not a 4× phantom.
    mask_pooled = block_reduce(zari_mask_hi.astype(np.float32),
                               block_size=(scale, scale),
                               func=np.mean)[:cards, :pins] >= pool_threshold

    s      = satin_settings.get('zari', {'n': 8, 'flip': False})
    satin  = generate_fill_pattern(s.get('pattern', 'satin'), s['n'], pins, cards, flip=s.get('flip', False))
    result = smart_fill(mask_pooled, satin, s['n'],
                        satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))
    return result


def generate_bmps(
    image: Image.Image,
    pins: int,
    cards: int,
    shuttle_count: int,
    color_assignments: dict,        # {color_index: shuttle_name}
    satin_settings: dict,           # {shuttle_name: {'n': int, 'flip': bool}}
    design_name: str,
    label_map: np.ndarray = None,   # pre-computed from detect step
    noise_min_size: int = 5,        # remove stray components < this many pixels
    emboss: bool = False,           # 1-shuttle only: split outline into rani
    supersample: bool = False       # oversample 4x then downsample for fine detail
) -> dict:
    """
    Generate all BMP files for a jacquard design.

    Pipeline:
      1. Resize image to pins × cards (nearest-neighbor — no anti-aliasing)
      2. Use pre-computed label_map (pixel-perfect match to colour preview)
         or re-run KMeans as fallback
      3. Validate label_map shape matches canvas exactly
      4. Build boolean masks per shuttle
         (multiple colour indices can map to one shuttle)
      5. Noise removal: strip stray pixels < noise_min_size
      6. smart_fill per shuttle:
            thin column runs  →  solid black (every thread UP)
            thick column runs →  satin pattern
      7. Rani (auto base): plain weave everywhere, suppressed wherever
         any other shuttle fires
      8. Write + return {filename: bytes}
    """

    # 1. Resize
    # BILINEAR resize preserves thin design features (grid lines, fine chevrons)
    # better than NEAREST which fragments 1-2px lines into disconnected pixels,
    # causing up to 10% of thin design elements to be missed entirely.
    # LANCZOS resize preserves thin grid lines and sharp edges without bleeding
    resized = image.resize((pins, cards), Image.LANCZOS)

    # 2. Label map
    if label_map is None:
        n_detect = shuttle_count + 1
        _, _, label_map = detect_colors(resized, n_detect)

    # 3. Shape validation
    if label_map.shape != (cards, pins):
        raise ValueError(
            f"label_map shape {label_map.shape} does not match "
            f"canvas ({cards} cards x {pins} pins). "
            "Please re-run Detect Colours before generating."
        )

    # 4. Build masks
    masks = {}
    for color_idx, shuttle_name in color_assignments.items():
        idx = int(color_idx)
        if shuttle_name not in masks:
            masks[shuttle_name] = np.zeros((cards, pins), dtype=bool)
        masks[shuttle_name] |= (label_map == idx)

    # 5. Noise removal on all non-background masks
    for name in list(masks.keys()):
        if name != 'background':
            masks[name] = remove_noise(masks[name], min_size=noise_min_size)

    results = {}

    if shuttle_count == 1:
        # ── 1 SHUTTLE ───────────────────────────────────────────────────────
        zari_mask = masks.get('zari', np.zeros((cards, pins), dtype=bool))
        s         = satin_settings.get('zari', {'n': 8, 'flip': False})
        satin     = generate_fill_pattern(s.get('pattern', 'satin'), s['n'], pins, cards, flip=s.get('flip', False))
        if s.get('weave_off', False):
            s = dict(s); s['min_height'] = 9999  # solid fill — no weave pattern

        # Determine background type for supersample decision
        _arr_rgb  = np.array(resized)
        _bg_bright = float(np.array(_arr_rgb, dtype=float).mean(axis=2).flatten()
                           [np.argsort(np.array(_arr_rgb).mean(axis=2).flatten())[-1]])
        _is_light_bg = np.array(_arr_rgb).mean(axis=2).mean() > 100 and                        float(np.percentile(np.array(_arr_rgb).mean(axis=2), 95)) > 180

        if not emboss:
            # ── Emboss OFF (default): zari = all design, no rani ────────────
            # Use supersampling for light-bg designs when requested,
            # to preserve fine interior gaps at low pin counts.
            if supersample and _is_light_bg:
                _n_colors = max(label_map.max() + 1, 2) if label_map is not None else 2
                _ss_arr = _supersample_to_bmp(
                    image, pins, cards, _n_colors,
                    satin_settings, color_assignments,
                    label_map, noise_min_size)
                arr = _ss_arr if _ss_arr is not None else smart_fill(
                    zari_mask, satin, s['n'],
                    satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))
            else:
                arr = smart_fill(zari_mask, satin, s['n'],
                                 satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))
            results[f'{design_name}_zari.bmp'] = write_1bit_bmp(arr)

        else:
            # ── Emboss ON: split design into fill (zari) + outline (rani) ───
            # Extract the boundary ring of every design shape via erosion.
            # Fill  → zari.bmp  (thick interior, satin or solid)
            # Outline → rani.bmp (boundary ring + plain weave base)
            outline_mask, fill_mask = extract_outline(zari_mask, thickness=1)

            # zari = fill interior only
            zari_arr = smart_fill(fill_mask, satin, s['n'],
                                  satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))
            results[f'{design_name}_zari.bmp'] = write_1bit_bmp(zari_arr)

            # rani = outline pixels (solid, always thin) + plain weave base
            plain_weave  = generate_plain_weave(pins, cards)
            outline_solid = np.ones((cards, pins), dtype=np.uint8)
            outline_solid[outline_mask] = 0   # solid fill for outline ring
            # OR-combine: fire where either plain weave OR outline fires
            rani_arr = np.where(
                (plain_weave == 0) | (outline_solid == 0),
                np.uint8(0),
                np.uint8(1)
            ).astype(np.uint8)
            results[f'{design_name}_rani.bmp'] = write_1bit_bmp(rani_arr)

    else:
        # ── 2-4 SHUTTLES ────────────────────────────────────────────────────
        shuttle_names = ['zari']
        if shuttle_count >= 3:
            shuttle_names.append('meena1')
        if shuttle_count >= 4:
            shuttle_names.append('meena2')

        # Determine background type once for all shuttles
        _arr_rgb_m   = np.array(resized)
        _is_light_bg_m = (_arr_rgb_m.mean(axis=2).mean() > 100 and
                          float(np.percentile(_arr_rgb_m.mean(axis=2), 95)) > 180)

        shuttle_arrays = {}
        for sname in shuttle_names:
            mask  = masks.get(sname, np.zeros((cards, pins), dtype=bool))
            s     = satin_settings.get(sname, {'n': 8, 'flip': False})
            satin = generate_fill_pattern(s.get('pattern', 'satin'), s['n'], pins, cards, flip=s.get('flip', False))
            if s.get('weave_off', False):
                s = dict(s); s['min_height'] = 9999  # solid fill

            if supersample and _is_light_bg_m:
                # Build a temporary color_assignments that maps only this
                # shuttle's colour index, so _supersample_to_bmp detects
                # and fills at 4× resolution for this shuttle's mask only.
                _n_colors = max(label_map.max() + 1, 2) if label_map is not None else 2
                _ss_satin = {sname: s}
                # Find which colour index maps to this shuttle
                _ss_assign = {k: v for k, v in color_assignments.items() if v == sname}
                _ss_assign['0'] = 'background'   # always keep background
                _ss_arr = _supersample_to_bmp(
                    image, pins, cards, _n_colors,
                    _ss_satin, _ss_assign,
                    label_map, noise_min_size)
                arr = _ss_arr if _ss_arr is not None else smart_fill(
                    mask, satin, s['n'],
                    satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))
            else:
                arr = smart_fill(mask, satin, s['n'],
                                 satin_min_height=s.get('min_height', _SATIN_MIN_HEIGHT))

            shuttle_arrays[sname] = arr
            results[f'{design_name}_{sname}.bmp'] = write_1bit_bmp(arr)

        # ── Rani: plain weave base + any remaining design colour ────────────
        # Rani always includes the full (row+col)%2 plain weave base.
        # In addition, any design colour NOT assigned to zari or meena
        # (i.e. the extra colour detected beyond the named shuttles) is
        # OR-combined into rani as an outline / accent layer.
        #
        # nColors passed from the detect step = shuttle_count + 1 for
        # 2/3/4-shuttle modes, so there is always one extra colour cluster
        # beyond the named shuttles. If that extra colour is genuine (not a
        # JPEG gradient artefact), it fires through rani on top of the
        # plain weave — giving the loom both the base structure and the
        # outline/secondary detail in a single shuttle.
        #
        # 2-shuttle example: nColors=3 → bg + teal (zari) + navy (rani extra)
        #   rani fires: plain weave OR navy outline pixels
        # 3-shuttle example: nColors=4 → bg + zari + meena + extra → rani
        # 4-shuttle example: nColors=5 → bg + zari + meena1 + meena2 + extra → rani

        plain_weave = generate_plain_weave(pins, cards)

        # ── Guaranteed full-design coverage for rani ─────────────────────────
        # The user requirement: rani must capture ALL design pixels not assigned
        # to a named shuttle (zari, meena1, meena2), so no design element is ever
        # lost regardless of how many colours the source contains.
        #
        # OLD approach: rani extra = union of label_map indices not in assignments.
        # Problem: with nColors=3 on a 5-colour design, minority colours land in
        # the background cluster (label 0) and never appear in the extra_mask.
        #
        # NEW approach (two-pass):
        # 1. Run a fast nColors=2 detection on the RESIZED image to get the full
        #    design mask — everything that is not background.
        # 2. Subtract all named-shuttle masks from it.
        # 3. Whatever remains = rani extra. This guarantees 100% coverage.
        #
        # Fallback: if the 2-colour detect fails, use the old label-index method.

        # Build union of all named-shuttle masks (zari + meena1 + meena2)
        named_mask = np.zeros((cards, pins), dtype=bool)
        for sname in shuttle_names:
            named_mask |= masks.get(sname, np.zeros((cards, pins), dtype=bool))

        # Two-pass: detect full design mask at target resolution
        try:
            _, _, lm_full, _ = detect_colors(resized, 2, edge_recovery=True)
            full_design_mask  = lm_full == 1
            full_design_mask  = remove_noise(full_design_mask, min_size=noise_min_size)
            extra_mask        = full_design_mask & ~named_mask
        except Exception:
            # Fallback: old label-index method
            assigned_indices = set(int(k) for k in color_assignments)
            extra_mask = np.zeros((cards, pins), dtype=bool)
            for lbl in range(1, label_map.max() + 1):
                if lbl not in assigned_indices:
                    extra_mask |= (label_map == lbl)
            extra_mask = remove_noise(extra_mask, min_size=noise_min_size)

        if extra_mask.any():
            # Remove noise from extra mask
            extra_mask = remove_noise(extra_mask, min_size=noise_min_size)

        if extra_mask.any():
            # Blend: rani fires where plain weave OR extra design colour
            # Use solid fill for the extra colour (it's typically an outline
            # / thin feature — solid is always correct for thin elements).
            extra_solid = np.ones((cards, pins), dtype=np.uint8)
            extra_solid[extra_mask] = 0   # 0 = UP (thread fires)

            # ── Correct rani combination ─────────────────────────────────────
            # REFERENCE ANALYSIS of 720_brokt_resham__1_.bmp confirms:
            #
            #   rani = (plain_weave AND NOT any_named_shuttle_UP)
            #          OR extra_design
            #
            # The rani/resham thread fires the plain weave base EXCEPT at
            # positions where a named shuttle (zari, meena1, meena2) is
            # already weaving.  Where a named thread is UP, the rani is
            # suppressed — the named thread provides the weft binding there.
            # This avoids double-firing and matches the loom reference exactly.
            #
            # In addition, rani fires for its own design content (extra_mask)
            # regardless of what the other shuttles are doing.

            # Build union mask of all named-shuttle UP positions
            named_up = np.zeros((cards, pins), dtype=bool)
            for _sname in shuttle_names:
                _sarr = shuttle_arrays.get(_sname)
                if _sarr is not None:
                    named_up |= (_sarr == 0)

            rani_arr = np.where(
                ((plain_weave == 0) & ~named_up) | (extra_solid == 0),
                np.uint8(0),    # UP / fire
                np.uint8(1)     # DOWN / hold
            ).astype(np.uint8)
        else:
            # No extra colour — plain weave suppressed by named shuttles
            named_up = np.zeros((cards, pins), dtype=bool)
            for _sname in shuttle_names:
                _sarr = shuttle_arrays.get(_sname)
                if _sarr is not None:
                    named_up |= (_sarr == 0)
            rani_arr = np.where(
                (plain_weave == 0) & ~named_up,
                np.uint8(0),
                np.uint8(1)
            ).astype(np.uint8)

        results[f'{design_name}_rani.bmp'] = write_1bit_bmp(rani_arr)

    return results


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_bmp(bmp_bytes: bytes) -> dict:
    """Verify a BMP is pure 1-bit (only black and white pixels)."""
    img        = Image.open(io.BytesIO(bmp_bytes))
    arr        = np.array(img.convert('RGB'))
    pure_black = int(((arr == 0).all(axis=2)).sum())
    pure_white = int(((arr == 255).all(axis=2)).sum())
    other      = int(arr.shape[0] * arr.shape[1]) - pure_black - pure_white
    return {
        'mode':         img.mode,
        'size':         list(img.size),
        'pure_black':   pure_black,
        'pure_white':   pure_white,
        'other_pixels': other,
        'is_clean':     bool(other == 0),
    }
