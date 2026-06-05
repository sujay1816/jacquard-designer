"""
Border Engine — high-detail BMP generation for border / running-line designs.

Sits ON TOP of bmp_engine.py and does NOT modify it. It imports and composes
bmp_engine's public functions only:

    detect_colors, remove_noise, smart_fill, generate_fill_pattern,
    generate_plain_weave, write_1bit_bmp, verify_bmp

Why a separate engine?
----------------------
Border designs (long, dense, fine-line saree-edge panels) lose detail in the
standard generator in two ways:

  1. COLOUR DETECTION. A flat RGB KMeans is the wrong model for line-art on a
     light ground. Resizing blends every thin line into the paper, so KMeans
     spends an entire cluster on the anti-aliased "blend halo" (a muddy
     pinkish-grey). That halo steals pixels that belong to the design and
     produces washed-out, improper colours.

  2. DOWNSCALE. A 1-pixel line crossing a 4x4 block fills only ~25% of it,
     below the engine's fixed 0.5 pool threshold, so thin lines vanish.

This engine fixes both, WITHOUT touching the core engine:

  * INK-FIRST SEGMENTATION. Separate ink from paper with an adaptive local
    threshold (plus a saturation rule that catches faint coloured lines), then
    classify ONLY the ink pixels into true thread colours. Colour centres are
    learned from the strong "core" of each line, so the blend halo never
    pollutes the palette. Result: clean, correct colours and full line capture.

  * HIGH-RES + LOW-THRESHOLD POOLING. Segment at scale x target resolution,
    then pool the mask down with a tunable "detail retention" threshold
    (default 0.18; ->0 approaches any-coverage) so thin lines survive.

Satin fill, mutual exclusion, and the rani plain-weave base are delegated to
bmp_engine's own functions so behaviour stays consistent with the main app.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans

from bmp_engine import (
    detect_colors,
    remove_noise,
    smart_fill,
    generate_fill_pattern,
    generate_plain_weave,
    write_1bit_bmp,
    verify_bmp,
)

# Border designs are line-art, not large solid bodies -> default solid crisp
# fill (very high min_height keeps everything solid). UI can override per shuttle.
_BORDER_SOLID_MIN_HEIGHT = 9999
_DEFAULT_SCALE = 4


# ---------------------------------------------------------------------------
# Ink-first segmentation
# ---------------------------------------------------------------------------
def _adaptive_ink(arr, lum, sat, sensitivity):
    """
    Return (full_ink, core_ink) boolean masks.

    full_ink : every pixel that is darker than its local paper level OR clearly
               coloured — captures even faint / blended lines.
    core_ink : the strong centre of each line — used to learn true thread
               colours without halo contamination.

    sensitivity > 1 catches fainter lines; < 1 is stricter.
    """
    H, W = lum.shape
    win = max(15, (W // 30) | 1)            # odd window, scales with resolution
    loc = uniform_filter(lum, size=win)     # local paper level

    s = max(0.25, float(sensitivity))
    C = 7.0 / s
    sat_thresh = max(14.0, 38.0 / s)

    darker  = lum < (loc - C)
    colored = (sat > sat_thresh) & (lum < 250)
    full_ink = darker | colored

    core = full_ink & ((lum < (loc - 2.2 * C)) | (sat > 75))
    return full_ink, core


def segment_ink_first(image, max_ink_colors=2, ink_sensitivity=1.0,
                      ink_centers=None):
    """
    Segment a line-art image into paper + N ink colours.

    Returns
    -------
    paper_rgb   : (3,) uint8        estimated paper colour
    ink_centers : (k,3) float       thread colours (ordered by count, or as
                                     supplied if ink_centers given)
    label_map   : (H,W) int         0 = paper, 1..k = ink colour index

    If ink_centers is supplied (e.g. from the detect step), it is reused for
    pixel assignment instead of re-clustering, so a later high-resolution pass
    produces labels perfectly aligned with the palette the user assigned.
    """
    im = image.convert('RGB')
    arr = np.asarray(im, dtype=np.float32)
    H, W = arr.shape[:2]
    lum = arr.mean(2)
    sat = np.asarray(im.convert('HSV'))[:, :, 1].astype(np.float32)

    full_ink, core = _adaptive_ink(arr, lum, sat, ink_sensitivity)

    # paper colour from the non-ink region
    if (~full_ink).any():
        paper_rgb = np.median(arr[~full_ink], axis=0)
    else:
        paper_rgb = np.array([245, 242, 242], dtype=np.float32)

    if ink_centers is not None:
        centers = np.asarray(ink_centers, dtype=np.float32)
        reorder = False
    else:
        k = max(1, int(max_ink_colors))
        sample = arr[core] if core.sum() >= k else arr[full_ink]
        if sample.shape[0] < k:
            # Degenerate (almost no ink) — fabricate centres from paper
            centers = np.tile(paper_rgb, (k, 1)).astype(np.float32)
        else:
            km = KMeans(n_clusters=k, n_init=4, random_state=42,
                        max_iter=100).fit(sample)
            centers = km.cluster_centers_.astype(np.float32)
        reorder = True

    # Assign every full-ink pixel to its nearest thread colour
    label = np.zeros((H, W), dtype=np.int32)
    if full_ink.any():
        fi = arr[full_ink]
        d = ((fi[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        nn = np.argmin(d, axis=1)
        label[full_ink] = nn + 1

    if reorder and centers.shape[0] > 1:
        counts = np.array([(label == i + 1).sum() for i in range(centers.shape[0])])
        order = np.argsort(-counts)
        remap = np.zeros(centers.shape[0] + 1, dtype=np.int32)
        for new_i, old_i in enumerate(order):
            remap[old_i + 1] = new_i + 1
        label = np.where(label > 0, remap[label], 0)
        centers = centers[order]

    return paper_rgb.astype(np.uint8), centers, label


# ---------------------------------------------------------------------------
# Detection for the API (palette + preview)
# ---------------------------------------------------------------------------
def detect_border(image, pins, cards, max_ink_colors=2, ink_sensitivity=1.0):
    """
    Run ink-first detection at target resolution and return UI-ready data.

    Returns dict:
        colors      : list of {index, rgb, hex, percentage, is_genuine}
                      index 0 = paper, 1..k = ink colours
        label_map   : (cards,pins) int   0=paper, 1..k=ink
        preview     : (cards,pins,3) uint8  colour-mapped preview
        ink_centers : list[[r,g,b], ...]    carried to generate for consistency
    """
    resized = image.convert('RGB').resize((pins, cards), Image.LANCZOS)
    paper, centers, label = segment_ink_first(
        resized, max_ink_colors=max_ink_colors, ink_sensitivity=ink_sensitivity)

    total = pins * cards
    palette = [paper] + [c.astype(np.uint8) for c in centers]
    colors = []
    for i, c in enumerate(palette):
        cnt = int((label == i).sum())
        colors.append({
            'index': i,
            'rgb': [int(x) for x in c],
            'hex': '#{:02x}{:02x}{:02x}'.format(*[int(x) for x in c]),
            'percentage': round(100 * cnt / total, 1),
            'count': cnt,
            'is_genuine': True,
        })

    preview = np.zeros((cards, pins, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        preview[label == i] = c

    return {
        'colors': colors,
        'label_map': label.astype(np.uint8),
        'preview': preview,
        'ink_centers': [[int(x) for x in c] for c in centers],
    }


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------
def _pool(mask_hi, scale, cards, pins, detail_retention, noise_min_size):
    from skimage.measure import block_reduce
    mask_hi = remove_noise(mask_hi, min_size=max(2, noise_min_size))
    cov = block_reduce(mask_hi.astype(np.float32),
                       block_size=(scale, scale), func=np.mean)[:cards, :pins]
    thr = max(0.0, float(detail_retention))
    return (cov > 0.0) if thr <= 0.0 else (cov >= thr)


def _hi_masks_inkfirst(image, pins, cards, color_assignments, ink_centers,
                       ink_sensitivity, scale, detail_retention, noise_min_size):
    """
    High-resolution ink-first masks. Returns (masks_by_shuttle, full_design) or
    (None, None) if skimage is unavailable.
    """
    try:
        import skimage.measure  # noqa: F401
    except Exception:
        return None, None

    hi = image.convert('RGB').resize((pins * scale, cards * scale), Image.LANCZOS)
    _, _, lab_hi = segment_ink_first(
        hi, ink_centers=ink_centers, ink_sensitivity=ink_sensitivity)

    # colour index (1..k) -> shuttle
    shuttle_indices = {}
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        if sname != 'background' and idx >= 1:
            shuttle_indices.setdefault(sname, set()).add(idx)

    masks = {}
    for sname, idxs in shuttle_indices.items():
        m_hi = np.zeros(lab_hi.shape, dtype=bool)
        for idx in idxs:
            m_hi |= (lab_hi == idx)
        masks[sname] = _pool(m_hi, scale, cards, pins, detail_retention, noise_min_size)

    full_design = _pool(lab_hi > 0, scale, cards, pins, detail_retention, noise_min_size)
    return masks, full_design


def _hi_masks_kmeans(image, pins, cards, n_colors, color_assignments,
                     palette_rgb, scale, noise_min_size, detail_retention):
    """Fallback high-res path using bmp_engine KMeans (no palette / ink model)."""
    try:
        import skimage.measure  # noqa: F401
    except Exception:
        return None, None

    hi = image.convert('RGB').resize((pins * scale, cards * scale), Image.LANCZOS)
    colors_hi, _, lm_hi, _ = detect_colors(hi, n_colors, edge_recovery=True)

    if palette_rgb and len(palette_rgb) >= n_colors:
        pal = np.asarray(palette_rgb, dtype=np.float32)
        hi_to_t = {}
        for j, c in enumerate(colors_hi):
            d = np.sqrt(((pal - np.asarray(c, np.float32)) ** 2).sum(1))
            hi_to_t[j] = int(np.argmin(d))
    else:
        hi_to_t = {j: j for j in range(n_colors)}

    shuttle_indices, bg = {}, set()
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        (bg.add(idx) if sname == 'background'
         else shuttle_indices.setdefault(sname, set()).add(idx))

    masks = {}
    for sname, idxs in shuttle_indices.items():
        m_hi = np.zeros(lm_hi.shape, dtype=bool)
        for j, t in hi_to_t.items():
            if t in idxs:
                m_hi |= (lm_hi == j)
        masks[sname] = _pool(m_hi, scale, cards, pins, detail_retention, noise_min_size)

    design_hi = np.zeros(lm_hi.shape, dtype=bool)
    for j, t in hi_to_t.items():
        if t not in bg:
            design_hi |= (lm_hi == j)
    full_design = _pool(design_hi, scale, cards, pins, detail_retention, noise_min_size)
    return masks, full_design


def _build_lowres_masks(label_map, pins, cards, color_assignments, noise_min_size):
    """Last-resort fallback from a target-resolution label map."""
    masks, bg = {}, set()
    for cidx, sname in color_assignments.items():
        idx = int(cidx)
        if sname == 'background':
            bg.add(idx); continue
        masks.setdefault(sname, np.zeros((cards, pins), dtype=bool))
        masks[sname] |= (label_map == idx)
    for sname in masks:
        masks[sname] = remove_noise(masks[sname], min_size=noise_min_size)
    design = np.zeros((cards, pins), dtype=bool)
    for lbl in range(int(label_map.max()) + 1):
        if lbl not in bg:
            design |= (label_map == lbl)
    return masks, remove_noise(design, min_size=noise_min_size)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_border_bmps(
    image: Image.Image,
    pins: int,
    cards: int,
    shuttle_count: int,
    color_assignments: dict,
    satin_settings: dict,
    design_name: str,
    label_map: np.ndarray = None,
    palette_rgb: list = None,          # [paper_rgb, ink1, ink2, ...] from detect
    detail_retention: float = 0.18,
    ink_sensitivity: float = 1.0,
    scale: int = _DEFAULT_SCALE,
    noise_min_size: int = 2,
    hi_detail: bool = True,
) -> dict:
    """
    Generate jacquard BMP files tuned for border / running-line designs.

    Prefers the ink-first model (when a palette from detect_border is supplied),
    falling back to a KMeans high-res path and finally a low-res label map.
    Output contract matches bmp_engine.generate_bmps ({filename: bytes}).
    """
    image = image.convert('RGB')
    resized = image.resize((pins, cards), Image.LANCZOS)
    if label_map is not None and label_map.shape != (cards, pins):
        raise ValueError(
            f"label_map shape {label_map.shape} does not match canvas "
            f"({cards} cards x {pins} pins). Re-run Detect first.")

    # ── Build masks: ink-first -> kmeans hi -> lowres ────────────────────────
    masks = full_design = None
    if hi_detail and palette_rgb and len(palette_rgb) >= 2:
        ink_centers = [list(map(float, c)) for c in palette_rgb[1:]]
        masks, full_design = _hi_masks_inkfirst(
            image, pins, cards, color_assignments, ink_centers,
            ink_sensitivity, scale, detail_retention, noise_min_size)

    if masks is None:
        n_colors = (len(palette_rgb) if palette_rgb
                    else (int(label_map.max()) + 1 if label_map is not None
                          else shuttle_count + 1))
        n_colors = max(2, n_colors)
        if hi_detail:
            masks, full_design = _hi_masks_kmeans(
                image, pins, cards, n_colors, color_assignments,
                palette_rgb, scale, noise_min_size, detail_retention)

    if masks is None:
        if label_map is None:
            _, _, label_map, _ = detect_colors(resized, max(2, shuttle_count + 1))
        masks, full_design = _build_lowres_masks(
            label_map, pins, cards, color_assignments, noise_min_size)

    # ── Shuttle setup ────────────────────────────────────────────────────────
    shuttle_names = ['zari']
    if shuttle_count >= 3: shuttle_names.append('meena1')
    if shuttle_count >= 4: shuttle_names.append('meena2')

    def _satin_for(sname):
        s = dict(satin_settings.get(sname, {}))
        s.setdefault('n', 8); s.setdefault('flip', False); s.setdefault('pattern', 'satin')
        if s.get('weave_off', True) and 'min_height' not in s:
            s['min_height'] = _BORDER_SOLID_MIN_HEIGHT
        s.setdefault('min_height', _BORDER_SOLID_MIN_HEIGHT)
        return s

    results = {}

    # 1 shuttle: zari = all design, no rani (mirror main app)
    if shuttle_count == 1:
        mask = masks.get('zari', full_design if full_design is not None
                         else np.zeros((cards, pins), dtype=bool))
        s = _satin_for('zari')
        satin = generate_fill_pattern(s['pattern'], s['n'], pins, cards, flip=s['flip'])
        arr = smart_fill(mask, satin, s['n'], satin_min_height=s['min_height'])
        results[f'{design_name}_zari.bmp'] = write_1bit_bmp(arr)
        return results

    # 2-4 shuttles
    shuttle_arrays = {}
    for sname in shuttle_names:
        mask = masks.get(sname, np.zeros((cards, pins), dtype=bool))
        s = _satin_for(sname)
        satin = generate_fill_pattern(s['pattern'], s['n'], pins, cards, flip=s['flip'])
        shuttle_arrays[sname] = smart_fill(mask, satin, s['n'], satin_min_height=s['min_height'])

    # Mutual exclusion (priority: zari > meena1 > meena2)
    claimed = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        arr = shuttle_arrays[sname]
        conflict = (arr == 0) & claimed
        if conflict.any():
            arr = arr.copy(); arr[conflict] = 1; shuttle_arrays[sname] = arr
        claimed |= (shuttle_arrays[sname] == 0)
        results[f'{design_name}_{sname}.bmp'] = write_1bit_bmp(shuttle_arrays[sname])

    # Rani: plain weave base, suppressed under named shuttles, + leftover design
    plain = generate_plain_weave(pins, cards)
    named_mask = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        named_mask |= masks.get(sname, np.zeros((cards, pins), dtype=bool))
    named_up = np.zeros((cards, pins), dtype=bool)
    for sname in shuttle_names:
        named_up |= (shuttle_arrays[sname] == 0)

    extra = None
    if full_design is not None:
        extra = remove_noise(full_design & ~named_mask, min_size=noise_min_size)

    if extra is not None and extra.any():
        extra_solid = np.ones((cards, pins), dtype=np.uint8); extra_solid[extra] = 0
        rani = np.where(((plain == 0) & ~named_up) | (extra_solid == 0),
                        np.uint8(0), np.uint8(1)).astype(np.uint8)
    else:
        rani = np.where((plain == 0) & ~named_up,
                        np.uint8(0), np.uint8(1)).astype(np.uint8)

    results[f'{design_name}_rani.bmp'] = write_1bit_bmp(rani)
    return results
