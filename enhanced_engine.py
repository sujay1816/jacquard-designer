"""
Enhanced Engine — fabric preprocessing + border image analysis.

Sits on top of bmp_engine.py without modifying it.

Two responsibilities:

  1. FABRIC PREPROCESSING  (for the generator page)
     Real fabric photos have two problems flat KMeans cannot handle:
       a) Lighting gradients — bright centre / dim edges from hand-held
          photography. A single gold thread appears as two different shades
          so KMeans splits it into two clusters.
       b) Weave texture — the 1–2 px periodic warp/weft pattern in the
          background cloth adds colour noise that pollutes clusters.

     preprocess_fabric_image() corrects both BEFORE detect_colors() runs,
     making KMeans work on a normalised, texture-free version of the image.

  2. BORDER IMAGE ANALYSIS  (for the border page)
     analyze_border_image() inspects an uploaded border strip and returns
     suggested slider values (pins, ink_sensitivity, detail_retention,
     noise_min_size) with a plain-English reason for each, so the user does
     not have to guess settings for an unfamiliar design.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, sobel

from bmp_engine import detect_colors


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fabric preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_fabric_image(image: Image.Image,
                             normalize_lighting: bool = True,
                             suppress_texture:   bool = True) -> Image.Image:
    """
    Normalise a real fabric photo before colour detection.

    normalize_lighting
        Estimates the lighting envelope with a large-sigma Gaussian and
        divides each pixel by it, equalising luminance across the frame.
        Fixes the "same gold thread = two different shades" problem.

    suppress_texture
        A 3×3 uniform filter removes the 1–2 px periodic weave pattern in
        the background cloth without blurring design edges (which are larger).
    """
    arr = np.asarray(image.convert('RGB'), dtype=np.float32)
    H, W = arr.shape[:2]

    if normalize_lighting:
        lum   = arr.mean(2)
        sigma = max(20.0, min(H, W) / 8.0)
        bg    = gaussian_filter(lum, sigma=sigma) + 1.0   # avoid divide-by-zero
        correction = (bg.mean() / bg)[:, :, np.newaxis]
        arr   = np.clip(arr * correction, 0.0, 255.0)

    if suppress_texture:
        for ch in range(3):
            arr[:, :, ch] = uniform_filter(arr[:, :, ch], size=3)

    return Image.fromarray(arr.astype(np.uint8))


def detect_colors_enhanced(image: Image.Image, n_colors: int,
                            edge_recovery: bool = True) -> tuple:
    """
    detect_colors with fabric preprocessing applied first.
    Returns the same (colors, counts, label_map, genuine_flags) tuple.
    """
    return detect_colors(
        preprocess_fabric_image(image),
        n_colors,
        edge_recovery=edge_recovery,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Border image analysis
# ─────────────────────────────────────────────────────────────────────────────

_STANDARD_PINS = [200, 240, 320, 400, 480, 540, 600, 720, 960]


def analyze_border_image(image: Image.Image) -> dict:
    """
    Analyze a border strip and return suggested slider values.

    Metrics:
      • Contrast ratio (p10 → p90)  →  ink_sensitivity
      • Normalised gradient density →  detail_retention + noise_min_size
      • Source dimensions           →  pins

    Returns:
        pins            : int
        ink_sensitivity : float  0.25–3.0
        detail_retention: float  0.00–0.60
        noise_min_size  : int    1–6
        reasons         : dict   one plain-English string per key
    """
    img = image.convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape[:2]
    lum = arr.mean(2)

    # ── Contrast → ink sensitivity ────────────────────────────────────────
    p10 = float(np.percentile(lum, 10))
    p90 = float(np.percentile(lum, 90))
    contrast = max((p90 - p10) / 255.0, 0.01)   # avoid zero

    if contrast < 0.15:
        ink_s, ink_why = 2.0, 'very low contrast — faint lines need maximum sensitivity'
    elif contrast < 0.30:
        ink_s, ink_why = 1.5, 'low contrast — boosted sensitivity for faint lines'
    elif contrast < 0.50:
        ink_s, ink_why = 1.2, 'moderate contrast — slight boost recommended'
    else:
        ink_s, ink_why = 1.0, 'clear contrast — standard sensitivity sufficient'

    # ── Gradient density → detail retention + noise min size ─────────────
    gx = sobel(lum, axis=1)
    gy = sobel(lum, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    # Normalise by contrast so bright designs don't score artificially high
    edge_density = float(np.mean(grad)) / (contrast * 255.0)

    if edge_density > 1.4:
        det_r, noise_n = 0.06, 1
        det_why = 'very dense / fine pattern — maximum detail preserved'
    elif edge_density > 0.9:
        det_r, noise_n = 0.10, 1
        det_why = 'fine pattern — tight threshold keeps small dots'
    elif edge_density > 0.55:
        det_r, noise_n = 0.15, 1
        det_why = 'medium detail — balanced threshold'
    elif edge_density > 0.30:
        det_r, noise_n = 0.20, 2
        det_why = 'moderately coarse design'
    else:
        det_r, noise_n = 0.28, 3
        det_why = 'coarse / thick-line design'

    # ── Pins ─────────────────────────────────────────────────────────────
    raw  = max(200, min(960, round(W * 0.65 / 20) * 20))
    pins = min(_STANDARD_PINS, key=lambda p: abs(p - raw))
    pins_why = f'source is {W}×{H} px — maintains proportional detail at loom scale'

    return {
        'pins':              pins,
        'ink_sensitivity':   round(ink_s,   2),
        'detail_retention':  round(det_r,   2),
        'noise_min_size':    noise_n,
        'reasons': {
            'pins':              pins_why,
            'ink_sensitivity':   ink_why,
            'detail_retention':  det_why,
            'noise_min_size':    (f'minimum {noise_n} px regions — '
                                  f'{"preserves finest dots" if noise_n == 1 else "removes sub-pixel noise"}'),
        },
    }
