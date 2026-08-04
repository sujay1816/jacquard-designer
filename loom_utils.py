"""
Loom utilities — physical-size conversion and weave-ability validation.

Pure, dependency-light helpers shared by the Butta / Border / Generator paths:
  * physical_size() — pins/cards -> real-world width/height at a given reed.
  * loom_warnings() — flag designs that exceed loom limits or contain features
    that won't weave cleanly (isolated single pixels, single-thread runs).
"""
import numpy as np

# Conservative defaults; callers can override per loom.
DEFAULT_MAX_PINS = 2640      # ends across the warp
DEFAULT_MAX_CARDS = 6000     # picks / cards


def physical_size(pins, cards, reed_epi=60.0, picks_ppi=None):
    """
    Convert a pin x card grid to a physical size.

    reed_epi   : ends (warp threads) per inch — the reed count.
    picks_ppi  : picks (weft) per inch; defaults to reed_epi (square sett).
    Returns a dict of width/height in inches and centimetres.
    """
    reed_epi = float(reed_epi) if reed_epi else 60.0
    picks_ppi = float(picks_ppi) if picks_ppi else reed_epi
    w_in = pins / reed_epi
    h_in = cards / picks_ppi
    return {
        'reed_epi': round(reed_epi, 2),
        'picks_ppi': round(picks_ppi, 2),
        'width_in': round(w_in, 2),
        'height_in': round(h_in, 2),
        'width_cm': round(w_in * 2.54, 1),
        'height_cm': round(h_in * 2.54, 1),
    }


def _count_isolated(mask, max_size=1):
    """Count connected ink components no larger than max_size pixels."""
    try:
        from scipy.ndimage import label
    except Exception:
        return 0
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    lbl, n = label(m)
    if n == 0:
        return 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # background
    return int(np.count_nonzero((sizes > 0) & (sizes <= max_size)))


def count_long_floats(mask, max_float=12):
    """
    Count runs of consecutive 'thread up' cells longer than max_float, in both
    the warp (down columns) and weft (along rows) directions. Long floats snag,
    sag, and weaken the cloth, so weavers cap them. Returns (count, longest).
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0, 0

    def _axis(a):
        cnt = lng = 0
        for row in a:
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            if starts.size:
                runs = ends - starts
                lng = max(lng, int(runs.max()))
                cnt += int((runs > max_float).sum())
        return cnt, lng

    ch, lh = _axis(m)        # weft floats (within each row)
    cv, lv = _axis(m.T)      # warp floats (within each column)
    return ch + cv, max(lh, lv)


def loom_warnings(mask, pins, cards,
                  max_pins=DEFAULT_MAX_PINS, max_cards=DEFAULT_MAX_CARDS):
    """
    Return a list of human-readable warnings for a 1-bit design mask
    (True = ink / thread up). Empty list means nothing to flag.
    """
    warnings = []
    if pins > max_pins:
        warnings.append(
            f"{pins} pins exceeds the typical loom limit of {max_pins}.")
    if cards > max_cards:
        warnings.append(
            f"{cards} cards exceeds the typical loom limit of {max_cards}.")

    if mask is not None:
        specks = _count_isolated(mask, max_size=1)
        if specks:
            warnings.append(
                f"{specks} isolated single-pixel point"
                f"{'s' if specks != 1 else ''} may not weave cleanly "
                f"(consider despeckle or a higher pin count).")
    return warnings


def source_resolution_check(image, pins, min_threads_per_stroke=2.0):
    """
    Check whether a source image carries enough detail for the requested pins.

    The binding constraint on sharpness is usually the artwork, not the
    algorithm: a stroke must land on at least min_threads_per_stroke threads
    or it rounds to a single thread, which thickens the ink and closes the
    hairline gaps between adjacent strokes.

    Measured on a 340x201 line-art butta whose finest strokes are 2px: at 200
    pins (1.7x reduction) ink grew 15% and the white background fragmented
    from 13 regions into 222. At 340 pins (1:1) ink drift was 2% and the gap
    structure survived intact at 14 regions.

    Returns a dict with the measured stroke width, the recommended minimum
    pin count, and a human-readable warning (empty when the source is fine).
    """
    import numpy as np

    try:
        from scipy import ndimage
    except Exception:
        return {'ok': True, 'warnings': []}

    g = np.asarray(image.convert('L'))
    ink = g < 128
    if not ink.any() or ink.all():
        return {'ok': True, 'warnings': []}

    # Median stroke width via the distance transform: 2x the distance from an
    # ink pixel to the nearest non-ink pixel is that stroke's local width.
    dt = ndimage.distance_transform_edt(ink)
    stroke_px = float(2 * np.median(dt[ink]))
    if stroke_px <= 0:
        return {'ok': True, 'warnings': []}

    src_w = image.size[0]
    reduction = src_w / max(pins, 1)
    threads_per_stroke = stroke_px / max(reduction, 1e-6)
    min_pins = int(np.ceil(pins * min_threads_per_stroke / max(threads_per_stroke, 1e-6)))

    warnings = []
    if threads_per_stroke < min_threads_per_stroke:
        warnings.append(
            f"Source is {src_w}px wide with {stroke_px:.1f}px strokes, so at "
            f"{pins} pins each stroke lands on only {threads_per_stroke:.1f} "
            f"threads. Detail will thicken and fine gaps will close. Use at "
            f"least {min_pins} pins, or supply artwork at least "
            f"{int(np.ceil(src_w * min_pins / max(pins, 1)))}px wide.")

    return {
        'ok': not warnings,
        'stroke_px': round(stroke_px, 1),
        'source_width': src_w,
        'reduction': round(reduction, 2),
        'threads_per_stroke': round(threads_per_stroke, 2),
        'recommended_min_pins': min_pins,
        'warnings': warnings,
    }
