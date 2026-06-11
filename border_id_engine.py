"""
Border Identification Engine — enhanced BMP generation for fine-detail border designs.

Extends border_engine with three targeted improvements that keep small dots,
thin scroll lines, and fine secondary motifs from disappearing:

  1. ADAPTIVE UPSCALE  — Automatically selects 4–8× upscale so the narrowest
     feature in the source image always has ≥ 6 px at high-resolution before
     any pooling step reduces it away.

  2. DUAL-THRESHOLD POOLING  — Pools the high-res ink mask twice in parallel:
     once at the standard threshold to preserve main lines, and once at a very
     low threshold (0.05) to recover single-pixel dots and hair-thin features.
     The two results are OR'd, so neither loses what the other finds.

  3. PRE-POOL CLOSING  — Applies a 1-pixel structural dilation to the high-res
     mask before block-reduce. This bridges anti-aliased gaps in thin lines
     (typically 1 dark / 1 light / 1 dark after JPEG compression) without
     merging genuinely separate design elements 3+ px apart.

All bmp_engine and border_engine functions are imported unchanged.
Output contract ({filename: bytes}) is identical to border_engine.
"""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation as _ndi_dilate

from bmp_engine import (
    remove_noise,
    smart_fill,
    generate_fill_pattern,
    generate_plain_weave,
    write_1bit_bmp,
    detect_colors,
)
from border_engine import segment_ink_first

_SOLID_MIN_HEIGHT = 9999   # keep border designs solid (no interior satin by default)
_FINE_THRESHOLD   = 0.05   # second-pass threshold: catches isolated dots
_MIN_FEATURE_PX   = 6      # minimum feature width guaranteed at high-res
_MAX_SCALE        = 8      # hard cap to avoid excessive memory on large images


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Adaptive scale selection
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_scale(src_size, pins, cards):
    """
    Choose the smallest integer upscale factor ≥ 4 that guarantees
    the narrowest significant feature in the source image has at least
    _MIN_FEATURE_PX pixels at high-resolution.

    Assumes the finest meaningful feature is ~3 px wide in the source.
    """
    src_w, src_h = src_size
    px_per_pin  = src_w / max(1, pins)
    px_per_card = src_h / max(1, cards)
    src_to_tgt  = min(px_per_pin, px_per_card)          # pixels per target pixel

    # Width of the finest feature at target resolution
    fine_at_tgt = max(1.0, 3.0 / src_to_tgt)

    # scale so: fine_at_tgt * scale >= _MIN_FEATURE_PX
    scale = math.ceil(_MIN_FEATURE_PX / fine_at_tgt)
    return max(4, min(_MAX_SCALE, scale))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dual-threshold pool with pre-closing
# ─────────────────────────────────────────────────────────────────────────────

def _pool_refined(mask_hi, scale, cards, pins, detail_retention, noise_min_size,
                  fine_thr=None):
    """
    Reduce a high-resolution boolean mask to (cards × pins) with:
      • 3×3 closing before block-reduce  (bridges anti-aliased gaps)
      • Two thresholds OR'd together      (standard + fine-detail)
    fine_thr overrides the fixed fine-dot threshold (used by auto-match so the
    fine pass can tighten on dense designs instead of flooring the ink).
    """
    try:
        from skimage.measure import block_reduce
    except ImportError:
        # Fallback: any-coverage max-pool (no skimage)
        out = np.zeros((cards, pins), dtype=bool)
        sh, sw = mask_hi.shape
        bh = max(1, sh // cards)
        bw = max(1, sw // pins)
        for r in range(cards):
            for c in range(pins):
                blk = mask_hi[r*bh:min((r+1)*bh, sh), c*bw:min((c+1)*bw, sw)]
                out[r, c] = blk.any()
        return remove_noise(out, min_size=noise_min_size)

    # Pre-pool closing: 3×3 dilation connects anti-aliased line gaps.
    # Radius = 1 px → dots ≥ 3 px apart remain separate.
    struct = np.ones((3, 3), dtype=bool)
    closed = _ndi_dilate(mask_hi, structure=struct)

    cov = block_reduce(
        closed.astype(np.float32),
        block_size=(scale, scale),
        func=np.mean,
    )[:cards, :pins]

    thr   = max(0.0, float(detail_retention))
    fthr  = _FINE_THRESHOLD if fine_thr is None else max(0.0, float(fine_thr))
    main = (cov >= thr) if thr > 0.0 else (cov > 0.0)   # standard pass
    fine = cov >= fthr                                    # catches isolated dots

    return remove_noise(main | fine, min_size=noise_min_size)


def _auto_detail_thr_refined(mask_hi, scale, cards, pins, noise_min_size):
    """
    Pick a tightness `t` so the refined-pool output ink-% matches the source
    ink-%. `t` drives BOTH thresholds (standard + fine), so dense designs can
    be brought back down instead of being floored by the fixed fine pass.
    Returns t (used as detail_retention and fine_thr together).
    """
    src = float((mask_hi > 0.5).mean())
    best, bt = None, 0.12
    for t in np.arange(0.06, 0.56, 0.04):
        ink = float(_pool_refined(
            mask_hi, scale, cards, pins, float(t), noise_min_size,
            fine_thr=float(t)).mean())
        d = abs(ink - src)
        if best is None or d < best:
            best, bt = d, float(t)
    return round(bt, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  High-res mask builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_hi_masks_inkfirst(image, pins, cards, color_assignments,
                              ink_centers, ink_sensitivity,
                              scale, detail_retention, noise_min_size,
                              auto_detail=False):
    """
    Ink-first segmentation at (pins*scale × cards*scale), pooled with
    _pool_refined. Uses border_engine.segment_ink_first for consistent
    ink/paper separation.
    Returns (shuttle_masks_dict, full_design_mask) or (None, None).
    """
    try:
        import skimage.measure  # noqa: F401
    except ImportError:
        return None, None

    hi = image.convert('RGB').resize((pins * scale, cards * scale), Image.LANCZOS)
    _, _, lab_hi = segment_ink_first(
        hi, ink_centers=ink_centers, ink_sensitivity=ink_sensitivity)

    if auto_detail:
        detail_retention = _auto_detail_thr_refined(
            lab_hi > 0, scale, cards, pins, noise_min_size)
    fine_ov = detail_retention if auto_detail else None

    shuttle_idxs = {}
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        if sname != 'background' and idx >= 1:
            shuttle_idxs.setdefault(sname, set()).add(idx)

    masks = {}
    for sname, idxs in shuttle_idxs.items():
        m_hi = np.zeros(lab_hi.shape, dtype=bool)
        for idx in idxs:
            m_hi |= (lab_hi == idx)
        masks[sname] = _pool_refined(
            m_hi, scale, cards, pins, detail_retention, noise_min_size,
            fine_thr=fine_ov)

    full_design = _pool_refined(
        lab_hi > 0, scale, cards, pins, detail_retention, noise_min_size,
        fine_thr=fine_ov)

    return masks, full_design


def _build_hi_masks_kmeans(image, pins, cards, n_colors, color_assignments,
                            scale, detail_retention, noise_min_size,
                            auto_detail=False):
    """KMeans high-res fallback when no ink_centers palette is available."""
    try:
        import skimage.measure  # noqa: F401
    except ImportError:
        return None, None

    hi = image.convert('RGB').resize((pins * scale, cards * scale), Image.LANCZOS)
    colors_hi, _, lm_hi, _ = detect_colors(hi, n_colors, edge_recovery=True)

    bg = {int(k) for k, v in color_assignments.items() if v == 'background'}
    shuttle_idxs = {}
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        if sname != 'background' and idx >= 1:
            shuttle_idxs.setdefault(sname, set()).add(idx)

    design_hi = np.zeros(lm_hi.shape, dtype=bool)
    for j in range(len(colors_hi)):
        if j not in bg:
            design_hi |= (lm_hi == j)

    if auto_detail:
        detail_retention = _auto_detail_thr_refined(
            design_hi, scale, cards, pins, noise_min_size)
    fine_ov = detail_retention if auto_detail else None

    masks = {}
    for sname, idxs in shuttle_idxs.items():
        m_hi = np.zeros(lm_hi.shape, dtype=bool)
        for idx in idxs:
            if idx < len(colors_hi):
                m_hi |= (lm_hi == idx)
        masks[sname] = _pool_refined(
            m_hi, scale, cards, pins, detail_retention, noise_min_size,
            fine_thr=fine_ov)

    full_design = _pool_refined(
        design_hi, scale, cards, pins, detail_retention, noise_min_size,
        fine_thr=fine_ov)

    return masks, full_design


def _build_lowres_masks(label_map, pins, cards, color_assignments, noise_min_size):
    """Last-resort: use the target-resolution label map directly."""
    masks, bg = {}, set()
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        if sname == 'background':
            bg.add(idx); continue
        masks.setdefault(sname, np.zeros((cards, pins), dtype=bool))
        masks[sname] |= (label_map == idx)
    for sname in masks:
        masks[sname] = remove_noise(masks[sname], min_size=noise_min_size)
    full_design = np.zeros((cards, pins), dtype=bool)
    for lbl in range(int(label_map.max()) + 1):
        if lbl not in bg:
            full_design |= (label_map == lbl)
    return masks, remove_noise(full_design, min_size=noise_min_size)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_border_id_bmps(
    image,
    pins,
    cards,
    shuttle_count,
    color_assignments,
    satin_settings,
    design_name,
    label_map=None,
    palette_rgb=None,
    detail_retention=0.12,   # tighter default than border_engine (0.18)
    ink_sensitivity=1.0,
    noise_min_size=1,         # keep finer dots by default (border_engine uses 2)
    hi_detail=True,
    auto_detail=False,
):
    """
    Generate jacquard BMPs with enhanced fine-detail preservation.

    Improvements over border_engine.generate_border_bmps:
      • Adaptive upscale  (4–8× chosen per image, not hardcoded 4×)
      • Dual-threshold pooling  (main lines + isolated dots both survive)
      • Pre-pool closing  (bridges anti-aliased gaps in thin lines)
      • Tighter defaults  (detail_retention 0.12, noise_min_size 1)

    Output: {filename: bytes} — identical contract to border_engine.
    """
    image   = image.convert('RGB')
    scale   = _adaptive_scale(image.size, pins, cards)
    resized = image.resize((pins, cards), Image.LANCZOS)

    if label_map is not None and label_map.shape != (cards, pins):
        raise ValueError(
            f'label_map shape {label_map.shape} does not match canvas '
            f'({cards} cards × {pins} pins). Re-run Detect first.')

    # ── Mask building: ink-first → KMeans hi-res → low-res fallback ──────────
    masks = full_design = None

    if hi_detail and palette_rgb and len(palette_rgb) >= 2:
        ink_centers = [list(map(float, c)) for c in palette_rgb[1:]]
        masks, full_design = _build_hi_masks_inkfirst(
            image, pins, cards, color_assignments, ink_centers,
            ink_sensitivity, scale, detail_retention, noise_min_size,
            auto_detail=auto_detail)

    if masks is None and hi_detail:
        n_colors = max(2,
            len(palette_rgb)         if palette_rgb     else
            int(label_map.max()) + 1 if label_map is not None else
            shuttle_count + 1)
        masks, full_design = _build_hi_masks_kmeans(
            image, pins, cards, n_colors, color_assignments,
            scale, detail_retention, noise_min_size, auto_detail=auto_detail)

    if masks is None:
        if label_map is None:
            _, _, label_map, _ = detect_colors(resized, max(2, shuttle_count + 1))
        masks, full_design = _build_lowres_masks(
            label_map, pins, cards, color_assignments, noise_min_size)

    # ── Shuttle setup ─────────────────────────────────────────────────────────
    shuttle_names = ['zari']
    if shuttle_count >= 3: shuttle_names.append('meena1')
    if shuttle_count >= 4: shuttle_names.append('meena2')

    def _satin(sname):
        s = dict(satin_settings.get(sname, {}))
        s.setdefault('n', 8)
        s.setdefault('flip', False)
        s.setdefault('pattern', 'satin')
        if s.get('weave_off', True) and 'min_height' not in s:
            s['min_height'] = _SOLID_MIN_HEIGHT
        s.setdefault('min_height', _SOLID_MIN_HEIGHT)
        return s

    results = {}

    # ── 1-shuttle shortcut ────────────────────────────────────────────────────
    if shuttle_count == 1:
        mask = masks.get('zari', full_design if full_design is not None
                         else np.zeros((cards, pins), dtype=bool))
        s   = _satin('zari')
        pat = generate_fill_pattern(s['pattern'], s['n'], pins, cards, flip=s['flip'])
        arr = smart_fill(mask, pat, s['n'], satin_min_height=s['min_height'])
        results[f'{design_name}_zari.bmp'] = write_1bit_bmp(arr)
        return results

    # ── 2–4 shuttles ─────────────────────────────────────────────────────────
    shuttle_arrs = {}
    for sname in shuttle_names:
        mask = masks.get(sname, np.zeros((cards, pins), dtype=bool))
        s    = _satin(sname)
        pat  = generate_fill_pattern(s['pattern'], s['n'], pins, cards, flip=s['flip'])
        shuttle_arrs[sname] = smart_fill(
            mask, pat, s['n'], satin_min_height=s['min_height'])

    # Mutual exclusion: zari > meena1 > meena2
    claimed = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        arr      = shuttle_arrs[sname]
        conflict = (arr == 0) & claimed
        if conflict.any():
            arr = arr.copy(); arr[conflict] = 1
            shuttle_arrs[sname] = arr
        claimed |= (shuttle_arrs[sname] == 0)
        results[f'{design_name}_{sname}.bmp'] = write_1bit_bmp(shuttle_arrs[sname])

    # Rani: plain-weave base, suppressed under named shuttles
    plain      = generate_plain_weave(pins, cards)
    named_mask = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        named_mask |= masks.get(sname, np.zeros((cards, pins), dtype=bool))
    named_up = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        named_up |= (shuttle_arrs[sname] == 0)

    extra = None
    if full_design is not None:
        extra = remove_noise(full_design & ~named_mask, min_size=noise_min_size)

    if extra is not None and extra.any():
        ex_solid         = np.ones((cards, pins), dtype=np.uint8)
        ex_solid[extra]  = 0
        rani = np.where(
            ((plain == 0) & ~named_up) | (ex_solid == 0),
            np.uint8(0), np.uint8(1)).astype(np.uint8)
    else:
        rani = np.where(
            (plain == 0) & ~named_up,
            np.uint8(0), np.uint8(1)).astype(np.uint8)

    results[f'{design_name}_rani.bmp'] = write_1bit_bmp(rani)
    return results
