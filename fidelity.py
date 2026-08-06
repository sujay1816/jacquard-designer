"""
Conversion fidelity — does the output still look like the design that went in?

The app already verifies that output is LOOM-SAFE: bmp_engine.verify_bmp checks
the file is pure 1-bit, and app._design_warnings flags long floats and isolated
pixels. Nothing checked whether the output is FAITHFUL to the uploaded design.

That gap is not theoretical. A butta reduced to 240 pins produced a perfectly
clean BMP with no warnings, while its white ground had fragmented from 13
regions into 147 — every hairline gap closed, the motif turned to mush. The
file was valid. The design was ruined. Only a comparison against the source
catches that.

The measures here are deliberately scale-invariant, because source and output
live at different resolutions and cannot be compared pixel to pixel:

  ink fraction   — proportion of the canvas carrying thread. Should barely
                   move. Growth means strokes thickened; loss means detail
                   dropped out.
  white regions  — count of separately-enclosed background areas. This is the
                   topology of the design: every hole in a motif, every gap
                   between strokes. It is the single most sensitive indicator
                   of a bad reduction.
  ink components — count of separate ink pieces. A large rise means the design
                   shattered; a large fall means pieces merged together.
  isolated cells — single lifted threads with no neighbour. These cannot weave.

None of this needs the source and output to be aligned or the same size, so it
works for any pipeline: generator, border, or butta.
"""
import numpy as np

try:
    from scipy import ndimage
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False

# Thresholds chosen from measured behaviour on real designs. A reduction that
# closed a butta's gaps showed +13% ink and an 11x rise in white regions; a
# good reduction of the same art showed +2% and 14 vs 13.
INK_DRIFT_WARN = 0.10          # 10% change in thread coverage
INK_DRIFT_FAIL = 0.25
TOPOLOGY_WARN = 3.0            # white regions multiplied by this much
TOPOLOGY_FAIL = 8.0
# Digital topology requires OPPOSITE connectivity for foreground and
# background, or the two disagree about whether diagonal touching counts and
# the counts become meaningless. Ink is 8-connected (a diagonal neighbour is
# the same stroke); background is 4-connected. Using 8 for both was measured
# to collapse 147 distinct background pockets into 13, hiding exactly the
# failure this module exists to detect.
_INK_CONN = np.ones((3, 3), dtype=bool)
_BG_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _binary(image, threshold=None):
    """
    Ink mask from a PIL image: True where thread is up.

    The threshold is chosen from the image's own histogram (Otsu) rather than
    fixed at mid-grey. Faint artwork breaks a fixed threshold completely: a
    pencil-drawn saree layout scanned on white paper has its linework around
    190-230 grey, so at 128 only 0.6% of the page registered as ink and the
    report claimed +3759% coverage drift on a conversion that was correct.

    Otsu splits paper from graphite wherever the ink actually sits, so the
    comparison holds for faint sketches, dense colour artwork, and clean line
    art alike. An explicit threshold can still be passed if needed.
    """
    grey = np.asarray(image.convert('L'))
    if threshold is None:
        try:
            from skimage.filters import threshold_otsu
            vals = grey[np.isfinite(grey)]
            threshold = float(threshold_otsu(vals)) if vals.size else 128
        except Exception:
            threshold = 128
    return grey < threshold


def _topology(mask):
    """(ink_components, white_regions, isolated_ink_cells)."""
    if not _HAVE_SCIPY:
        return None, None, None
    lbl, ink_n = ndimage.label(mask, structure=_INK_CONN)
    _, white_n = ndimage.label(~mask, structure=_BG_CONN)
    sizes = np.bincount(lbl.ravel())
    if sizes.size:
        sizes[0] = 0
    isolated = int(((sizes > 0) & (sizes <= 1)).sum())
    return int(ink_n), int(white_n), isolated


def fidelity_report(source_image, output_mask, threshold=None):
    """
    Compare a generated design mask against the uploaded source.

    source_image : PIL image as uploaded (any resolution)
    output_mask  : 2D bool array at loom resolution, True where thread is up
    threshold    : ink cutoff for the source; None picks it per image (Otsu)

    Returns a dict of metrics plus 'verdict' ('ok' | 'warn' | 'fail') and
    'messages', written for a weaver rather than a developer.
    """
    src = _binary(source_image, threshold)
    out = np.asarray(output_mask, dtype=bool)

    src_ink, out_ink = float(src.mean()), float(out.mean())
    drift = (out_ink / src_ink - 1.0) if src_ink > 0 else 0.0

    s_comp, s_white, _ = _topology(src)
    o_comp, o_white, o_iso = _topology(out)

    report = {
        'source_ink_pct': round(100 * src_ink, 1),
        'output_ink_pct': round(100 * out_ink, 1),
        'ink_drift_pct': round(100 * drift, 1),
        'source_white_regions': s_white,
        'output_white_regions': o_white,
        'source_ink_components': s_comp,
        'output_ink_components': o_comp,
        'isolated_cells': o_iso,
        'messages': [],
        'verdict': 'ok',
    }

    def flag(level, msg):
        report['messages'].append(msg)
        if level == 'fail' or report['verdict'] == 'fail':
            report['verdict'] = 'fail'
        elif level == 'warn' and report['verdict'] == 'ok':
            report['verdict'] = 'warn'

    # ── Topology health gates how harshly coverage drift is judged ─────────
    # A deliberate trade exists: thinning strokes to keep gaps open lowers
    # coverage while IMPROVING the design. Condemning that would tell the
    # weaver the better output is the worse one, so drift is only a failure
    # when the structure is also wrong.
    topology_ok = not (s_white and o_white) or (
        1.0 / TOPOLOGY_WARN <= o_white / max(s_white, 1) <= TOPOLOGY_WARN)

    # ── Thread coverage ────────────────────────────────────────────────────
    if abs(drift) >= INK_DRIFT_FAIL and not topology_ok:
        flag('fail', (
            f"Thread coverage changed {100*drift:+.0f}% against the source "
            f"({report['source_ink_pct']}% to {report['output_ink_pct']}%). "
            f"{'Strokes have thickened and fine gaps are closing.' if drift > 0 else 'Detail is dropping out.'}"))
    elif abs(drift) >= INK_DRIFT_WARN:
        note = (" Structure is intact, so this is lighter linework rather than "
                "lost detail." if topology_ok and drift < 0 else
                " Check fine detail in the preview.")
        flag('warn', f"Thread coverage changed {100*drift:+.0f}% against the source.{note}")

    # ── Topology: the sensitive one ────────────────────────────────────────
    if s_white and o_white:
        ratio = o_white / max(s_white, 1)
        if ratio >= TOPOLOGY_FAIL:
            flag('fail', (
                f"The background broke into {o_white} separate areas but the "
                f"source has only {s_white}. Thread has bridged across the gaps "
                f"between strokes, so the motif will read as a solid mass. Raise "
                f"the pin count or supply higher-resolution artwork."))
        elif ratio >= TOPOLOGY_WARN:
            flag('warn', (
                f"The background broke into {o_white} areas against the source's "
                f"{s_white}. Some fine gaps are closing."))
        elif ratio <= 1.0 / TOPOLOGY_FAIL:
            flag('warn', (
                f"Only {o_white} background areas survived against the source's "
                f"{s_white}. Enclosed detail inside motifs may have been lost."))

    # ── Fragmentation ──────────────────────────────────────────────────────
    if s_comp and o_comp and o_comp >= max(6 * s_comp, s_comp + 40):
        flag('warn', (
            f"The design split into {o_comp} separate pieces against the "
            f"source's {s_comp}. Thin strokes may be breaking up."))

    if o_iso:
        flag('warn' if o_iso < 50 else 'fail', (
            f"{o_iso} single lifted thread{'s' if o_iso != 1 else ''} with no "
            f"neighbour. These cannot weave and will show as faults."))

    if not report['messages']:
        report['messages'].append(
            f"Coverage and structure match the source "
            f"({100*drift:+.0f}% thread, {o_white} background areas vs {s_white}).")

    return report
