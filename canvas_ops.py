"""
Canvas and region operations.

The agent could reshape linework — thicken, thin, close gaps — but it could not
change the cloth itself. It could not widen a panel for a bigger loom, trim
blank selvedge, shift a motif off a fold, clear a region and put something else
there, or repeat a butta across a field. Those are the operations a designer
spends most of their time on, and their absence is why the assistant felt like
a converter with a chat box rather than a design tool.

Everything here works on the LABEL MAP — the array of per-thread class indices
that the rest of the pipeline already uses — not on RGB pixels. That matters:

  * Class indices survive every operation, so a three-colour design keeps its
    shuttle separation through a move, a crop or a paste. Working in RGB and
    re-clustering afterwards would let boundaries drift on every edit.
  * Ground stays ground. Every operation that creates new canvas fills it with
    class 0, which is thread down, which is bare cloth. New canvas is never
    ambiguous.
  * There is no resampling unless it is asked for. Moving and cropping are
    index operations; nothing is interpolated, so nothing softens.

Coordinates are (x, y) in threads and cards from the top-left, matching how the
BMP editor already talks about position.
"""
import numpy as np

MAX_PINS, MAX_CARDS = 2640, 6000
GROUND = 0

# Named regions, so the model can say "the left border" instead of computing a
# box. Each returns (x0, y0, x1, y1) as fractions of the canvas.
NAMED_REGIONS = {
    'all':          (0.00, 0.00, 1.00, 1.00),
    'top':          (0.00, 0.00, 1.00, 0.25),
    'bottom':       (0.00, 0.75, 1.00, 1.00),
    'left':         (0.00, 0.00, 0.25, 1.00),
    'right':        (0.75, 0.00, 1.00, 1.00),
    'centre':       (0.25, 0.25, 0.75, 0.75),
    'center':       (0.25, 0.25, 0.75, 0.75),
    'left_border':  (0.00, 0.00, 0.10, 1.00),
    'right_border': (0.90, 0.00, 1.00, 1.00),
    'pallu':        (0.00, 0.85, 1.00, 1.00),
    'body':         (0.10, 0.00, 0.90, 0.85),
    'top_half':     (0.00, 0.00, 1.00, 0.50),
    'bottom_half':  (0.00, 0.50, 1.00, 1.00),
}


def resolve_region(lm, region=None, box=None):
    """
    Turn a named region or an explicit box into pixel bounds.

    Returns (x0, y0, x1, y1) clipped to the canvas, or raises ValueError. The
    bounds are half-open, so x1 and y1 are exclusive and an empty region is
    detectable rather than silently one thread wide.
    """
    h, w = lm.shape
    if box:
        try:
            x0, y0, x1, y1 = (int(v) for v in box)
        except (TypeError, ValueError):
            raise ValueError('A box must be four numbers: x0, y0, x1, y1.')
    elif region:
        key = str(region).strip().lower().replace(' ', '_')
        if key not in NAMED_REGIONS:
            raise ValueError(f"Unknown region '{region}'. Try: "
                             f"{', '.join(sorted(NAMED_REGIONS))}, or give a box.")
        fx0, fy0, fx1, fy1 = NAMED_REGIONS[key]
        x0, y0, x1, y1 = (int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h))
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('That region has no area on this canvas.')
    return x0, y0, x1, y1


def _check_size(w, h):
    if not (10 <= w <= MAX_PINS):
        raise ValueError(f'{w} threads is outside the loom range 10-{MAX_PINS}.')
    if not (10 <= h <= MAX_CARDS):
        raise ValueError(f'{h} cards is outside the range 10-{MAX_CARDS}.')


# ── Canvas size ─────────────────────────────────────────────────────────────

def extend(lm, left=0, right=0, top=0, bottom=0):
    """
    Grow the canvas, filling the new area with ground.

    New cloth is bare cloth. Filling with anything else — mirroring the edge,
    repeating the last column — would put thread lifts on the loom that nobody
    asked for, and they are hard to spot in a design that already has a repeat.
    """
    h, w = lm.shape
    left, right = max(0, int(left)), max(0, int(right))
    top, bottom = max(0, int(top)), max(0, int(bottom))
    if not (left or right or top or bottom):
        raise ValueError('Nothing to extend by.')
    _check_size(w + left + right, h + top + bottom)
    return np.pad(lm, ((top, bottom), (left, right)),
                  mode='constant', constant_values=GROUND).astype(np.uint8)


def crop(lm, region=None, box=None):
    """Keep only the given region, discarding the rest."""
    x0, y0, x1, y1 = resolve_region(lm, region, box)
    _check_size(x1 - x0, y1 - y0)
    return lm[y0:y1, x0:x1].copy()


def trim(lm, margin=0):
    """
    Crop away blank cloth around the design.

    Useful after a move or a crop leaves a band of ground down one side. Raises
    rather than returning an empty array if the canvas has no design on it —
    silently returning a 0x0 array would fail much later and much less clearly.
    """
    rows = np.any(lm != GROUND, axis=1)
    cols = np.any(lm != GROUND, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError('The canvas is empty — there is nothing to trim to.')
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    m = max(0, int(margin))
    h, w = lm.shape
    y0, y1 = max(0, y0 - m), min(h - 1, y1 + m)
    x0, x1 = max(0, x0 - m), min(w - 1, x1 + m)
    _check_size(x1 - x0 + 1, y1 - y0 + 1)
    return lm[y0:y1 + 1, x0:x1 + 1].copy()


def resize_canvas(lm, pins=None, cards=None, anchor='center'):
    """
    Set the canvas to an exact size, extending or cropping as needed.

    The design is NOT scaled — it keeps its thread count and is placed within
    the new canvas. Scaling a design to fit a different loom is a different
    operation with different consequences (see `scale`), and conflating them
    would silently resample a design when the weaver asked to re-mount it.
    """
    h, w = lm.shape
    tw = int(pins) if pins else w
    th = int(cards) if cards else h
    _check_size(tw, th)

    ax = {'left': 0.0, 'center': 0.5, 'centre': 0.5, 'right': 1.0}
    ay = {'top': 0.0, 'center': 0.5, 'centre': 0.5, 'bottom': 1.0}
    a = str(anchor).lower()
    fx = next((v for k, v in ax.items() if k in a), 0.5)
    fy = next((v for k, v in ay.items() if k in a), 0.5)

    out = np.full((th, tw), GROUND, dtype=np.uint8)
    # Overlap of source and destination, positioned by the anchor.
    dx = int((tw - w) * fx)
    dy = int((th - h) * fy)
    sx0, sy0 = max(0, -dx), max(0, -dy)
    dx0, dy0 = max(0, dx), max(0, dy)
    cw = min(w - sx0, tw - dx0)
    ch = min(h - sy0, th - dy0)
    if cw > 0 and ch > 0:
        out[dy0:dy0 + ch, dx0:dx0 + cw] = lm[sy0:sy0 + ch, sx0:sx0 + cw]
    return out


def scale(lm, pins=None, cards=None):
    """
    Resample the design to a new thread count.

    Nearest-neighbour, deliberately. Interpolating a label map would invent
    class indices that lie between two shuttles and belong to neither. This
    does lose detail going down and blocks it going up — say so rather than
    presenting it as a free operation.
    """
    h, w = lm.shape
    tw = int(pins) if pins else w
    th = int(cards) if cards else h
    _check_size(tw, th)
    ys = np.clip((np.arange(th) * h / th).astype(int), 0, h - 1)
    xs = np.clip((np.arange(tw) * w / tw).astype(int), 0, w - 1)
    return lm[ys][:, xs].astype(np.uint8)


# ── Moving ──────────────────────────────────────────────────────────────────

def move(lm, dx=0, dy=0, wrap=False):
    """
    Shift the design across the canvas.

    wrap=True rolls it round, which is what an all-over repeat wants — moving a
    seam off the fold without losing anything. wrap=False pushes design off the
    edge and it is gone, which is what a single motif usually wants.
    """
    dx, dy = int(dx), int(dy)
    if wrap:
        return np.roll(np.roll(lm, dy, axis=0), dx, axis=1).astype(np.uint8)
    out = np.full_like(lm, GROUND)
    h, w = lm.shape
    sx0, dx0 = (0, dx) if dx >= 0 else (-dx, 0)
    sy0, dy0 = (0, dy) if dy >= 0 else (-dy, 0)
    cw, ch = w - abs(dx), h - abs(dy)
    if cw > 0 and ch > 0:
        out[dy0:dy0 + ch, dx0:dx0 + cw] = lm[sy0:sy0 + ch, sx0:sx0 + cw]
    return out


def centre(lm):
    """Move the design so its bounding box sits centred on the canvas."""
    rows = np.any(lm != GROUND, axis=1)
    cols = np.any(lm != GROUND, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError('The canvas is empty — there is nothing to centre.')
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    h, w = lm.shape
    return move(lm, dx=int((w - (x1 + x0 + 1)) / 2), dy=int((h - (y1 + y0 + 1)) / 2))


# ── Regions ─────────────────────────────────────────────────────────────────

def clear(lm, region=None, box=None):
    """Erase a region back to bare cloth."""
    x0, y0, x1, y1 = resolve_region(lm, region, box)
    out = lm.copy()
    out[y0:y1, x0:x1] = GROUND
    return out


def copy_region(lm, region=None, box=None):
    """Lift a region out as its own array, for pasting elsewhere."""
    x0, y0, x1, y1 = resolve_region(lm, region, box)
    return lm[y0:y1, x0:x1].copy()


def paste(lm, patch, x=0, y=0, mode='over'):
    """
    Place a patch onto the canvas.

    mode='over' writes everything including the patch's ground, which is what
    replacing a region means. mode='blend' writes only the patch's design
    cells, leaving whatever is underneath showing through the gaps — which is
    what layering a motif onto a ground means. Getting these confused is how a
    stamped butta ends up punching a white rectangle through a lattice.
    """
    out = lm.copy()
    h, w = out.shape
    ph, pw = patch.shape
    x, y = int(x), int(y)
    sx0, dx0 = max(0, -x), max(0, x)
    sy0, dy0 = max(0, -y), max(0, y)
    cw, ch = min(pw - sx0, w - dx0), min(ph - sy0, h - dy0)
    if cw <= 0 or ch <= 0:
        raise ValueError('That would land the patch entirely off the canvas.')
    sub = patch[sy0:sy0 + ch, sx0:sx0 + cw]
    if mode == 'blend':
        target = out[dy0:dy0 + ch, dx0:dx0 + cw]
        out[dy0:dy0 + ch, dx0:dx0 + cw] = np.where(sub != GROUND, sub, target)
    else:
        out[dy0:dy0 + ch, dx0:dx0 + cw] = sub
    return out.astype(np.uint8)


def transform_region(lm, op, region=None, box=None):
    """Mirror or rotate just one region, leaving the rest of the cloth alone."""
    x0, y0, x1, y1 = resolve_region(lm, region, box)
    sub = lm[y0:y1, x0:x1]
    if op in ('flip_horizontal', 'mirror'):
        new = np.fliplr(sub)
    elif op == 'flip_vertical':
        new = np.flipud(sub)
    elif op == 'rotate_180':
        new = np.rot90(sub, 2)
    elif op == 'invert':
        new = np.where(sub == GROUND, 1, GROUND)
    else:
        raise ValueError(f"Cannot do '{op}' to a region. Use mirror, "
                         f"flip_vertical, rotate_180 or invert.")
    out = lm.copy()
    out[y0:y1, x0:x1] = new
    return out


def mirror_across(lm, axis='vertical'):
    """
    Mirror one half onto the other, making the panel symmetric.

    This is how a saree border pair is usually built — draw one selvedge, then
    mirror it — and doing it by hand means getting the fold exactly on the
    centre thread, which is easy to be one off on.
    """
    out = lm.copy()
    h, w = out.shape
    if axis in ('vertical', 'v', 'left_to_right'):
        half = w // 2
        out[:, w - half:] = np.fliplr(out[:, :half])
    elif axis in ('horizontal', 'h', 'top_to_bottom'):
        half = h // 2
        out[h - half:, :] = np.flipud(out[:half, :])
    else:
        raise ValueError("axis must be 'vertical' or 'horizontal'.")
    return out


def tile_region(lm, cols=1, rows=1, region=None, box=None):
    """
    Repeat a region across the whole canvas.

    The way a single drawn butta becomes a field. Repeat counts are capped
    because a thousand-fold tile is always a typo, never a design.
    """
    patch = copy_region(lm, region, box)
    cols, rows = max(1, min(40, int(cols))), max(1, min(60, int(rows)))
    h, w = lm.shape
    tile = np.tile(patch, (rows, cols))
    out = np.full_like(lm, GROUND)
    ch, cw = min(h, tile.shape[0]), min(w, tile.shape[1])
    out[:ch, :cw] = tile[:ch, :cw]
    return out


# The weaves bmp_engine can actually build. This list is checked BEFORE
# calling it, because generate_fill_pattern falls back to satin for any name it
# does not recognise rather than raising — so asking for a twill silently
# returned a satin, and asking for a tartan returned a satin too. A weaver who
# asked for twill and got satin has no way to tell until the cloth is off the
# loom.
WEAVES = ('satin', 'satin_inv', 'plain_weave', 'twill22', 'twill31', 'dots',
          'diagonal', 'crosshatch', 'honeycomb', 'diamond', 'herringbone',
          'basket', 'crepe', 'rib')

# What weavers say, mapped to what the engine calls it.
WEAVE_ALIASES = {'twill': 'twill22', 'twill_2_2': 'twill22',
                 'twill_3_1': 'twill31', 'plain': 'plain_weave',
                 'matt': 'basket', 'hopsack': 'basket'}


def fill_region_weave(lm, pattern='satin', n=8, region=None, box=None,
                      design_only=True):
    """
    Texture a region with a weave.

    design_only=True fills only the cells that already carry thread, which is
    what texturing a motif means. False fills the whole rectangle, which is
    what laying a patterned ground means. The default is the safe one: filling
    a whole rectangle over a motif buries it.
    """
    from bmp_engine import generate_fill_pattern

    key = str(pattern).lower().strip().replace(' ', '_')
    key = WEAVE_ALIASES.get(key, key)
    if key not in WEAVES:
        raise ValueError(f"No weave called '{pattern}'. Available: "
                         f"{', '.join(WEAVES)}.")

    x0, y0, x1, y1 = resolve_region(lm, region, box)
    rw, rh = x1 - x0, y1 - y0
    try:
        pat = np.asarray(generate_fill_pattern(key, int(n), rw, rh))
    except Exception as e:
        raise ValueError(f"Cannot make a '{pattern}' weave: {e}")
    if pat.shape != (rh, rw):
        pat = np.resize(pat, (rh, rw))
    # bmp_engine's convention is 0 = thread UP, 1 = thread DOWN. Reading 1 as
    # "lift" inverts the weave — the float lands exactly where the binding
    # should be, and the texture comes out as its own negative.
    lift = pat == 0

    out = lm.copy()
    sub = out[y0:y1, x0:x1]
    if design_only:
        cls = sub[sub != GROUND]
        keep = int(np.bincount(cls).argmax()) if cls.size else 1
        out[y0:y1, x0:x1] = np.where((sub != GROUND) & lift, keep, GROUND)
    else:
        out[y0:y1, x0:x1] = np.where(lift, 1, GROUND)
    return out.astype(np.uint8)


def stamp_motif(lm, motif, width_threads, x=0, y=0, mode='blend', threads=2):
    """
    Draw a motif from the library straight onto the canvas.

    Built at width_threads, so its strokes are sized for the space it will
    occupy rather than inherited from the full cloth width — the same rule the
    composition layer applies to borders and tiles.
    """
    import motif_library as ml

    if motif not in ml.MOTIFS:
        raise ValueError(f"Unknown motif '{motif}'. Available: "
                         f"{', '.join(sorted(ml.MOTIFS))}")
    wt = max(16, min(int(width_threads), lm.shape[1]))
    svg = ml.build_svg(motif, wt, colours=threads + 1)
    img = ml.render(svg, wt).convert('L')
    patch = (np.asarray(img) < 128).astype(np.uint8)   # ink -> class 1
    return paste(lm, patch, x=x, y=y, mode=mode)


def stats(lm):
    """Describe the canvas: size, coverage, where the design actually sits."""
    h, w = lm.shape
    design = lm != GROUND
    out = {'pins': int(w), 'cards': int(h),
           'thread_coverage_pct': round(float(design.mean()) * 100, 1),
           'classes': sorted(int(c) for c in np.unique(lm))}
    if design.any():
        rows, cols = np.any(design, axis=1), np.any(design, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        out['design_box'] = [int(x0), int(y0), int(x1 + 1), int(y1 + 1)]
        out['blank_margins'] = {'left': int(x0), 'right': int(w - x1 - 1),
                                'top': int(y0), 'bottom': int(h - y1 - 1)}
    else:
        out['design_box'] = None
        out['note'] = 'The canvas is empty.'
    return out
