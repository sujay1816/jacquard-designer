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
