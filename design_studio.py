"""
Design studio — intent in, finished layout out.

The problem this solves is that the agent's design tools were parameter tools.
A weaver had to arrive already knowing they wanted `layout="jaal"`,
`motif="lotus"`, `cols=5`, `rows=4`, `spacing=0.25`. That is not a
conversation, it is a form with a chat box in front of it, and it puts the
design decisions on the person least equipped to make them at that moment.

So this module moves the decisions into code that can actually check them:

  * plan()     — from a pin count, a reed and a description of the cloth,
                 work out what can be designed at that width and rank the
                 options by how well they fit.
  * compose()  — build a WHOLE PANEL, not a bare field. Real saree work is a
                 body between two side borders, usually with a cross border.
                 A rectangle of repeating paisleys is not a saree, and handing
                 one over as if it were is the single biggest gap between what
                 the generator produced and what a mill can use.
  * explore()  — render several candidates, convert and SCORE each one against
                 the loom, and return them ranked. The search is deterministic;
                 the model reads the scores and picks.
  * refine()   — weaver-language adjustment ("bolder", "more open", "finer")
                 applied to the SPEC and re-rendered from vector, not smeared
                 across pixels.

Everything is built from the engines already in the product — motif_library
for geometry, auto_convert for conversion, fidelity for scoring, loom_utils
for physical size. Nothing here touches a pixel directly.

Why re-render from the spec rather than edit the raster: a spec is a handful of
numbers, so any change can be re-rendered at the exact pin count with stroke
weights recomputed for it. Editing the raster instead means every adjustment
degrades what came before, and ten small tweaks leave a design that no single
step broke and none of them can undo.
"""
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import motif_library as ml
import loom_utils

# Motifs that read as a running band rather than a standalone butta. Used for
# borders and for the rules between motif rows.
BAND_MOTIFS = ('vine_border', 'chevron_border')
BUTTA_MOTIFS = ('paisley', 'lotus')
GROUND_MOTIFS = ('diamond_jaal', 'check_ground', 'dotted_field')

# A side border narrower than this cannot carry a running motif — it becomes a
# stripe. Below it, plan() drops the border rather than shipping a band whose
# detail has already closed up.
MIN_BORDER_THREADS = 28

# Fraction of cloth width one side border takes, before clamping.
DEFAULT_BORDER_FRAC = 0.09


# ── The spec ────────────────────────────────────────────────────────────────

@dataclass
class LayoutSpec:
    """
    A complete panel, described in numbers small enough to re-render at will.

    This is the unit the assistant passes around: proposing, refining and
    regenerating all operate on a spec, and the raster is derived from it. A
    design is therefore never in a state that cannot be explained or reversed.
    """
    pins: int = 480
    cards: Optional[int] = None
    threads: int = 2                    # ink threads, excluding the ground

    body_motif: str = 'paisley'
    body_layout: str = 'half_drop'
    cols: int = 6
    rows: int = 8
    spacing: float = 0.25
    mirror: bool = False

    border: bool = True
    border_motif: str = 'vine_border'
    border_frac: float = DEFAULT_BORDER_FRAC

    cross_border: bool = False          # pallu band across the width
    cross_motif: str = 'chevron_border'
    cross_frac: float = 0.12

    reed: Optional[float] = None
    picks: Optional[float] = None

    def dict(self):
        return asdict(self)

    def border_pins(self):
        return int(self.pins * self.border_frac) if self.border else 0

    def body_pins(self):
        return max(24, self.pins - 2 * self.border_pins())


# ── Planning ────────────────────────────────────────────────────────────────

# Words a weaver actually uses, mapped onto the geometry that produces them.
# The point is that "rich" and "open" are real design vocabulary with real
# consequences for thread count, and the model should not have to invent the
# translation each time.
FEEL = {
    'rich':       {'spacing': 0.12, 'density': 1.30, 'layout': 'jaal'},
    'dense':      {'spacing': 0.10, 'density': 1.40, 'layout': 'half_drop'},
    'traditional':{'spacing': 0.22, 'density': 1.00, 'layout': 'half_drop'},
    'classic':    {'spacing': 0.22, 'density': 1.00, 'layout': 'half_drop'},
    'open':       {'spacing': 0.45, 'density': 0.70, 'layout': 'straight'},
    'light':      {'spacing': 0.55, 'density': 0.60, 'layout': 'straight'},
    'minimal':    {'spacing': 0.70, 'density': 0.45, 'layout': 'straight'},
    'geometric':  {'spacing': 0.20, 'density': 1.10, 'layout': 'straight'},
    'formal':     {'spacing': 0.20, 'density': 1.00, 'layout': 'banded'},
}


def plan(pins=None, reed=None, picks=None, cards=None, feel=None, threads=2,
         motif=None, borders=True, pallu=False, length_in=None, width_in=None):
    """
    Work out what can be designed at this width, and rank the options.

    Returns the geometry AND the physical size, because those are the same
    decision. A weaver asking for 480 pins is really asking for a certain
    number of inches of cloth, and at reed 60 versus reed 100 that is 8 inches
    of difference — which changes how many motifs should go across before
    anyone picks one.
    """
    # A brief may arrive as inches instead of threads. Resolving it here means
    # every caller — plan, auto_design, the agent tool — accepts either.
    if not pins and width_in:
        pins = int(round(float(width_in) * float(reed or 60.0)))
    pins = _clamp_int(pins, 10, 2640, 480)
    feel_key = _match_feel(feel)
    prefs = FEEL[feel_key]

    geom = geometry(pins, cards=cards, reed=reed, picks=picks, length_in=length_in)
    cards = geom['cards']

    border_pins = int(pins * DEFAULT_BORDER_FRAC) if borders else 0
    if borders and border_pins < MIN_BORDER_THREADS:
        # Widen to the minimum if the cloth can spare it; otherwise drop the
        # border rather than ship a band whose motif has already closed up.
        if pins * 0.30 >= 2 * MIN_BORDER_THREADS:
            border_pins = MIN_BORDER_THREADS
        else:
            borders, border_pins = False, 0
    body_pins = max(24, pins - 2 * border_pins)

    options = []
    for name in (BUTTA_MOTIFS + GROUND_MOTIFS):
        if motif and name != motif:
            continue
        need = ml.MIN_THREADS_PER_MOTIF.get(name, 32)
        max_across = max(1, body_pins // need)
        if max_across < 1:
            continue
        cols = max(1, int(round(max_across * 0.62 * prefs['density'])))
        cols = max(1, min(cols, max_across))
        threads_each = body_pins // cols
        options.append({
            'motif': name,
            'kind': 'butta' if name in BUTTA_MOTIFS else 'ground',
            'cols': cols,
            'threads_per_motif': threads_each,
            'max_across': max_across,
            'headroom': round(threads_each / need, 2),
            'reads_well': threads_each >= need,
        })

    # Rank by comfort: enough threads to hold detail, but not so few motifs
    # that the cloth looks empty. Headroom near 1.4 is the sweet spot found by
    # MIN_THREADS_PER_MOTIF — at 1.0 the detail only just survives.
    #
    # Buttas outrank grounds when both read well, because someone asking for a
    # saree body means a motif. Ranking on headroom alone always wins with
    # dotted_field: it needs 12 threads against paisley's 48, so it scores best
    # at every width and the weaver is handed scattered dots for a request that
    # meant paisley. Grounds are the fallback for cloth too narrow to hold a
    # butta, not the default answer.
    def rank(o):
        return (not o['reads_well'],
                0 if o['kind'] == 'butta' else 1,
                abs(o['headroom'] - 1.4))

    options.sort(key=rank)

    best = options[0] if options else None
    cols = best['cols'] if best else 6
    body_cards = int(cards * (1 - (0.12 if pallu else 0.0)))
    body_layout = prefs['layout'] if not best or best['kind'] == 'butta' else 'straight'
    rows = _rows_to_fill(best['motif'] if best else 'dotted_field', cols,
                         prefs['spacing'], body_pins, body_cards, threads,
                         layout=body_layout)

    spec = LayoutSpec(
        pins=pins, cards=cards, threads=_clamp_int(threads, 1, 3, 2),
        body_motif=best['motif'] if best else 'dotted_field',
        body_layout=body_layout,
        cols=best['cols'] if best else 6, rows=rows,
        spacing=prefs['spacing'],
        border=bool(borders), border_frac=(border_pins / pins) if border_pins else 0.0,
        cross_border=bool(pallu),
        reed=reed, picks=picks)

    return {
        'spec': spec.dict(),
        'feel': feel_key,
        'geometry': geom,
        'body_pins': body_pins,
        'border_pins': border_pins,
        'options': options[:6],
        'chosen': best,
        'why': _explain(best, body_pins, border_pins, geom, feel_key, borders),
    }


def _explain(best, body_pins, border_pins, geom, feel_key, borders):
    """One sentence a weaver can overrule, not a parameter dump."""
    if not best:
        return 'This width is too narrow to carry a motif; a plain ground is the honest answer.'
    bits = [f"{geom['pins']} pins is {geom['width_in']}in at reed {geom['reed_epi']}"]
    if borders and border_pins:
        bits.append(f"two {border_pins}-thread borders leave {body_pins} for the body")
    bits.append(f"{best['cols']} {best['motif']} across gives each one "
                f"{best['threads_per_motif']} threads")
    if best['headroom'] < 1.15:
        bits.append('which is tight — the interior detail will only just read')
    elif best['headroom'] > 2.2:
        bits.append('with room to spare, so more across would still work')
    return '; '.join(bits) + f" ({feel_key} feel)"


def _match_feel(feel):
    if not feel:
        return 'traditional'
    text = str(feel).lower()
    for key in FEEL:
        if key in text:
            return key
    return 'traditional'


def _rows_to_fill(motif, cols, spacing, body_pins, body_cards, threads=2,
                  layout='half_drop'):
    """
    Rows needed to fill the body, measured from the motif's real aspect ratio.

    Two things have to be right here, and both are easy to get wrong.

    First, the aspect: guessing from a square tile leaves bare cloth above the
    pallu, because a paisley tile is appreciably taller than it is wide and a
    lotus is not. The motif is built once at tile size and measured, which
    costs nothing next to rendering and is right for every motif rather than
    for whichever one an estimate was tuned on.

    Second, the pitch. Spacing does NOT add row height in allover(): the gap
    divides into the tile scale, so sh = mh * tile_w / (mw * (1+gap)) and
    row_h = sh * (1+gap) = tile_w * aspect, with the gap cancelling exactly.
    Treating spacing as extra vertical pitch under-counts rows and leaves the
    foot of the panel empty. The jaal layout is the exception — it steps by the
    lattice cell, not the tile.
    """
    cols = max(1, int(cols))
    tile_pins = body_pins / cols
    if layout == 'jaal':
        row_pins = tile_pins                 # lattice cell, not tile height
    else:
        try:
            _, mw, mh = ml._tile(motif, max(24, tile_pins), colours=threads + 1)
            aspect = (mh / mw) if mw else 1.0
        except Exception:
            aspect = 1.0
        row_pins = tile_pins * aspect
    return max(1, min(40, int(math.ceil(body_cards / max(row_pins, 1)))))


def _clamp_int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


# ── Physical geometry ───────────────────────────────────────────────────────

def geometry(pins=None, cards=None, reed=None, picks=None,
             width_in=None, length_in=None):
    """
    Convert freely between threads and inches, in whichever direction is known.

    The reed is what makes a pin count mean anything physical, and until now
    nothing exposed it to the assistant — so pin counts were being chosen with
    no idea of the finished cloth width. Give inches and get threads, or give
    threads and get inches.
    """
    reed = float(reed) if reed else 60.0
    picks = float(picks) if picks else reed

    if pins is None and width_in:
        pins = int(round(float(width_in) * reed))
    pins = _clamp_int(pins, 10, 2640, 480)

    if cards is None and length_in:
        cards = int(round(float(length_in) * picks))
    if cards is None:
        cards = int(round(pins * 1.4))          # a workable default panel
    cards = _clamp_int(cards, 10, 6000, 672)

    size = loom_utils.physical_size(pins, cards, reed_epi=reed, picks_ppi=picks)
    return {
        'pins': pins, 'cards': cards,
        'reed_epi': reed, 'picks_ppi': picks,
        'width_in': round(pins / reed, 2),
        'length_in': round(cards / picks, 2),
        'width_cm': round(pins / reed * 2.54, 1),
        'length_cm': round(cards / picks * 2.54, 1),
        'raw': size,
    }


# ── Composition ─────────────────────────────────────────────────────────────

def compose(spec: LayoutSpec) -> str:
    """
    Build the whole panel as one SVG: side borders, body field, cross border.

    Each region is built as if the loom were only as wide as that region, which
    is the rule motif_library established for tiles and which matters even more
    here — a border occupying 9% of the cloth would otherwise inherit strokes
    scaled to the full width and land far under the weavable threshold.
    """
    view = ml._VIEW
    border_pins = spec.border_pins()
    body_pins = spec.body_pins()

    body_view = view * body_pins / spec.pins
    border_view = view * border_pins / spec.pins

    # Body height in view units, derived from the requested card count so the
    # panel comes out the shape that was asked for.
    total_h = int(round(view * (spec.cards or int(spec.pins * 1.4)) / spec.pins))
    cross_h = int(total_h * spec.cross_frac) if spec.cross_border else 0
    body_h = max(40, total_h - cross_h)

    parts = []

    # Rows are recomputed here, not taken from the spec. A refinement that
    # changes cols or spacing changes the tile height too, and carrying the old
    # row count forward would leave bare cloth above the pallu after an
    # adjustment that had nothing to do with height.
    body_cards = int(round((spec.cards or int(spec.pins * 1.4)) * body_h / max(total_h, 1)))
    rows = _rows_to_fill(spec.body_motif, spec.cols, spec.spacing,
                         body_pins, body_cards, spec.threads,
                         layout=spec.body_layout)

    # Body — built at body_pins, so its motifs are sized for the space they get.
    body_svg = ml.allover(
        max(24, body_pins), layout=spec.body_layout, motif=spec.body_motif,
        cols=spec.cols, rows=rows, spacing=spec.spacing,
        mirror=spec.mirror, colours=spec.threads + 1,
        cards=int(round(body_pins * body_h / max(body_view, 1))))
    inner, bw, bh = ml._inner(body_svg)
    s = body_view / bw
    parts.append(f'<g transform="translate({border_view:.1f},0) scale({s:.5f})">'
                 f'{inner}</g>')

    # Side borders — the same band rotated up each selvedge.
    if border_pins >= MIN_BORDER_THREADS:
        band = _vertical_band(spec.border_motif, border_pins, border_view,
                              body_h, spec.threads + 1)
        parts.append(f'<g transform="translate(0,0)">{band}</g>')
        parts.append(f'<g transform="translate({view - border_view:.1f},0)'
                     f' scale(-1,1)"><g transform="translate({-border_view:.1f},0)">'
                     f'{band}</g></g>')

    # Cross border (pallu) across the full width at the foot.
    if cross_h:
        cb = ml.build_svg(spec.cross_motif, spec.pins,
                          repeats=max(4, spec.cols * 2))
        ci, cw, ch = ml._inner(cb)
        cs = view / cw
        reps = max(1, int(math.ceil(cross_h / max(ch * cs, 1))))
        for i in range(reps):
            parts.append(f'<g transform="translate(0,{body_h + i*ch*cs:.1f}) '
                         f'scale({cs:.5f})">{ci}</g>')

    return ml._svg(''.join(parts), view, total_h)


def _vertical_band(motif, band_pins, band_view, height_view, colours):
    """
    A running border rotated to run up the selvedge, tiled to full height.

    Built at band_pins so the motif is scaled for the border's own width. SVG
    rotate(90) maps (x, y) to (-y, x), so the band lands in negative x and is
    translated back by its own height.
    """
    svg = ml.build_svg(motif, max(24, band_pins), repeats=3, colours=colours)
    inner, w, h = ml._inner(svg)
    s = band_view / h                     # the band's HEIGHT becomes its width
    run = w * s                           # how far one copy reaches vertically
    reps = max(1, int(math.ceil(height_view / max(run, 1))))
    out = []
    for i in range(reps):
        out.append(f'<g transform="translate({band_view:.1f},{i*run:.1f}) '
                   f'rotate(90) scale({s:.5f})">{inner}</g>')
    return ''.join(out)


def render(spec: LayoutSpec):
    """Rasterise a spec at its own pin count."""
    svg = compose(spec)
    return ml.render(svg, spec.pins, cards=spec.cards)


# ── Exploration ─────────────────────────────────────────────────────────────

def variants(base: LayoutSpec, n=3):
    """
    Alternative specs worth showing beside the base.

    Deliberately varied along axes a weaver can SEE — density, layout,
    borders — rather than along parameters that produce near-identical cloth.
    Showing three specs that differ only in spacing wastes the weaver's time.
    """
    out = [base]
    d = base.dict()

    lighter = LayoutSpec(**{**d, 'cols': max(1, int(base.cols * 0.7)),
                            'spacing': min(1.0, base.spacing + 0.25)})
    denser = LayoutSpec(**{**d, 'cols': base.cols + 2,
                           'spacing': max(0.05, base.spacing - 0.10),
                           'body_layout': 'jaal' if base.body_layout != 'jaal'
                                          else 'half_drop'})
    plain_ground = LayoutSpec(**{**d, 'body_motif': 'diamond_jaal',
                                 'body_layout': 'straight',
                                 'cols': max(2, base.cols)})
    for v in (denser, lighter, plain_ground):
        if len(out) < max(1, n):
            out.append(v)
    return out[:max(1, n)]


def explore(base: LayoutSpec, n=3, n_colors=None):
    """
    Render, convert and score each variant; return them ranked.

    This is the part worth doing in code rather than in the model. Whether a
    field will weave is a measurement, and measuring three candidates costs
    seconds; asking a model to predict it costs accuracy. The model's job is to
    read the scores and say which one suits the cloth — a judgement the numbers
    do not contain.
    """
    from auto_convert import auto_convert

    results = []
    for spec in variants(base, n):
        try:
            img = render(spec)
        except Exception as e:
            results.append({'spec': spec.dict(), 'error': str(e)})
            continue
        colours = n_colors or (spec.threads + 1)
        conv = auto_convert(img, pins=spec.pins, n_colors=max(2, min(4, colours)))
        best = conv.get('best')
        if not best:
            results.append({'spec': spec.dict(), 'error': conv.get('summary', 'would not convert')})
            continue
        rep = best['report']
        results.append({
            'spec': spec.dict(),
            'summary': describe(spec),
            'verdict': conv['verdict'],
            'score': best.get('score'),
            'thread_drift_pct': rep['ink_drift_pct'],
            'design_gaps': rep['output_white_regions'],
            'pins': best['pins'], 'cards': best['cards'],
            '_image': img, '_conversion': conv,
        })

    ok = [r for r in results if 'error' not in r]
    # auto_convert's `score` is a sort-key TUPLE where lower is better, not a
    # quality number — negating it raises. Rank on the reported metrics
    # instead: verdict first, then how far thread coverage drifted.
    rank = {'ok': 0, 'warn': 1, 'fail': 2}
    ok.sort(key=lambda r: (rank.get(str(r.get('verdict', 'fail')).lower(), 2),
                           r.get('thread_drift_pct') or 0))
    return ok + [r for r in results if 'error' in r]


# ── Refinement ──────────────────────────────────────────────────────────────

# Weavers do not say "increase spacing to 0.4". They say the cloth is busy.
# Each entry adjusts the SPEC, so the change is re-rendered from vector at full
# quality rather than applied to pixels that have already been reduced.
REFINEMENTS = {
    'more_open':   ('spacing', +0.15, 'motifs spaced further apart'),
    'denser':      ('spacing', -0.10, 'motifs packed closer'),
    'fewer_motifs':('cols', -1, 'fewer motifs across, each one larger'),
    'more_motifs': ('cols', +1, 'more motifs across, each one smaller'),
    'taller':      ('rows', +1, 'another row of motifs'),
    'shorter':     ('rows', -1, 'one row fewer'),
    'wider_border':('border_frac', +0.02, 'wider side borders'),
    'narrower_border': ('border_frac', -0.02, 'narrower side borders'),
}

ALIASES = {
    'busy': 'more_open', 'crowded': 'more_open', 'too tight': 'more_open',
    'empty': 'denser', 'sparse': 'denser', 'bare': 'denser',
    'bigger': 'fewer_motifs', 'larger': 'fewer_motifs',
    'smaller': 'more_motifs', 'finer': 'more_motifs',
}


def refine(spec: LayoutSpec, instruction: str):
    """
    Apply a named refinement, or a phrase that maps to one.

    Returns (new_spec, description, error). Refusing an unknown instruction is
    deliberate: silently doing nothing looks identical to doing the wrong
    thing, and the weaver would only find out at the loom.
    """
    key = str(instruction or '').strip().lower().replace(' ', '_')
    if key not in REFINEMENTS:
        plain = key.replace('_', ' ')
        key = next((v for k, v in ALIASES.items() if k in plain), None)
    if key not in REFINEMENTS:
        return None, None, (f"I do not have a refinement for that. Available: "
                            f"{', '.join(sorted(REFINEMENTS))}")

    field_name, delta, desc = REFINEMENTS[key]
    d = spec.dict()
    value = d.get(field_name, 0)
    new = value + delta

    limits = {'spacing': (0.0, 1.2), 'cols': (1, 24), 'rows': (1, 40),
              'border_frac': (0.0, 0.25)}
    lo, hi = limits.get(field_name, (0, 10 ** 6))
    clamped = max(lo, min(hi, new))
    if clamped == value:
        return None, None, (f"Already at the limit for that — {field_name} is "
                            f"{value} and cannot go further.")
    d[field_name] = clamped
    return LayoutSpec(**d), desc, None


def describe(spec: LayoutSpec) -> str:
    """The design in a sentence, for the assistant to read back."""
    bits = [f'{spec.cols} {spec.body_motif} across in a {spec.body_layout} repeat']
    if spec.border and spec.border_pins() >= MIN_BORDER_THREADS:
        bits.append(f'{spec.border_motif} borders {spec.border_pins()} threads wide')
    if spec.cross_border:
        bits.append(f'a {spec.cross_motif} pallu band')
    bits.append(f'{spec.pins} x {spec.cards}')
    if spec.reed:
        g = geometry(spec.pins, spec.cards, spec.reed, spec.picks)
        bits.append(f"{g['width_in']}in wide at reed {int(spec.reed)}")
    return ', '.join(bits)


# ── Autonomous search ───────────────────────────────────────────────────────
#
# The agentic part. Given a goal, this tries designs, measures each against the
# loom, keeps what improved and reports the trail. It is a hill climb, not a
# model guessing: every step is scored by fidelity, so the loop cannot talk
# itself into a design that will not weave.
#
# Why a loop rather than one shot: the first plan is an estimate. Whether a
# field actually converts depends on how the motif's strokes land on the
# specific thread grid, which is only knowable by rendering it. One render
# costs under a second, so trying six and keeping the best is cheap — and it
# is exactly the work a person should not have to do by hand.

# Directions the climb can move. Each is a spec edit whose effect on
# convertibility is monotonic enough to hill-climb on.
MOVES = (
    ('more_open',    'spacing',     +0.12),
    ('denser',       'spacing',     -0.08),
    ('fewer_motifs', 'cols',        -1),
    ('more_motifs',  'cols',        +1),
    ('wider_border', 'border_frac', +0.02),
)

_LIMITS = {'spacing': (0.0, 1.2), 'cols': (1, 24), 'rows': (1, 40),
           'border_frac': (0.0, 0.25)}

_RANK = {'ok': 0, 'warn': 1, 'fail': 2}


def score_spec(spec: LayoutSpec, n_colors=None, settings=None, full=False):
    """
    Render, convert and measure one spec. Returns a record or None.

    `full=False` does ONE conversion at fixed settings rather than calling
    auto_convert, which runs its own sixteen-candidate search internally. That
    search is right when converting an unknown photograph — it does not know
    which settings suit the image. It is wrong inside a hill climb: the search
    was already done once for the starting spec, the settings do not change as
    spacing and column count move, and paying for it at every step took a step
    from under a second to nine.

    The winner is re-scored with full=True at the end, so what is handed over
    is a proper auto_convert record and nothing downstream can tell the
    difference.
    """
    import numpy as np
    from fidelity import fidelity_report
    from vision_engine import detect_colors_smart

    try:
        img = render(spec)
    except Exception:
        return None
    colours = max(2, min(4, n_colors or spec.threads + 1))

    if full or settings is None:
        from auto_convert import auto_convert
        conv = auto_convert(img, pins=spec.pins, n_colors=colours)
        best = conv.get('best')
        if not best:
            return None
        rep = best['report']
        return {'spec': spec, 'image': img, 'conversion': conv,
                'settings': best.get('settings'),
                'verdict': str(conv['verdict']).lower(),
                'drift': rep['ink_drift_pct'],
                'gaps': rep['output_white_regions']}

    try:
        _, _, lm, _ = detect_colors_smart(img, colours, spec.pins,
                                          spec.cards, **settings)
        rep = fidelity_report(img, np.asarray(lm) > 0)
    except Exception:
        return None
    return {'spec': spec, 'image': img, 'conversion': None, 'settings': settings,
            'verdict': str(rep.get('verdict', 'fail')).lower(),
            'drift': rep['ink_drift_pct'],
            'gaps': rep['output_white_regions']}


def _quality(rec):
    """
    Lower is better. Verdict dominates, then how far thread coverage drifted.

    Drift is the honest headline metric: it is how much of the design's ink
    survived the reduction to threads, so a design that keeps its coverage kept
    its motif. Gap count is deliberately NOT in the sort — a design can gain
    gaps by breaking up, which is bad, or by opening out, which is good, and the
    number alone cannot tell those apart.
    """
    return (_RANK.get(rec['verdict'], 2), rec['drift'])


def auto_design(pins=None, reed=None, feel=None, threads=2, motif=None,
                borders=True, pallu=False, cards=None, width_in=None,
                length_in=None, rounds=4, on_step=None):
    """
    Work toward the best weavable design for a brief, and show the working.

    Returns the winning record plus a trail of what was tried and why it was
    kept or dropped. The trail matters as much as the result: a weaver who is
    told "six paisleys across, drift 11%" has to trust it, while one who is
    shown that eight across drifted 27% and five drifted 9% can see the shape
    of the trade and argue with it.
    """
    p = plan(pins=pins, reed=reed, cards=cards, feel=feel, threads=threads,
             motif=motif, borders=borders, pallu=pallu,
             width_in=width_in, length_in=length_in)
    current = score_spec(LayoutSpec(**p['spec']), full=True)
    if current is None:
        return {'error': 'That brief does not produce a design that converts.',
                'plan': p}

    trail = [{'step': 'start', 'design': describe(current['spec']),
              'verdict': current['verdict'], 'drift': current['drift'],
              'kept': True}]
    if on_step:
        on_step(trail[-1])

    tried = set()
    for _ in range(max(1, min(8, int(rounds)))):
        best_move, best_rec = None, None
        for name, field_name, delta in MOVES:
            d = current['spec'].dict()
            lo, hi = _LIMITS.get(field_name, (0, 10 ** 6))
            new_val = max(lo, min(hi, d.get(field_name, 0) + delta))
            if new_val == d.get(field_name):
                continue
            d[field_name] = new_val
            key = (d['cols'], round(d['spacing'], 3), round(d['border_frac'], 3))
            if key in tried:
                continue
            tried.add(key)
            rec = score_spec(LayoutSpec(**d), settings=current.get('settings'))
            if rec and (best_rec is None or _quality(rec) < _quality(best_rec)):
                best_move, best_rec = name, rec

        if best_rec is None or _quality(best_rec) >= _quality(current):
            trail.append({'step': 'stop',
                          'note': 'No further change improved it.', 'kept': False})
            if on_step:
                on_step(trail[-1])
            break

        improvement = current['drift'] - best_rec['drift']
        current = best_rec
        trail.append({'step': best_move, 'design': describe(current['spec']),
                      'verdict': current['verdict'], 'drift': current['drift'],
                      'improved_drift_by': round(improvement, 1), 'kept': True})
        if on_step:
            on_step(trail[-1])

    # Re-score the winner properly. The climb used one fixed setting for
    # speed; what gets handed over must be a full auto_convert record, or the
    # shuttle assignment and file generation downstream have nothing to work
    # from.
    if current.get('conversion') is None:
        final = score_spec(current['spec'], full=True)
        if final is not None:
            current = final

    return {'best': current, 'trail': trail, 'plan': p,
            'rounds_used': len([t for t in trail if t.get('kept')]) - 1}


def thumbnail(img, max_px=640, fmt='PNG'):
    """
    Base64 image for a vision model, downscaled and greyscale.

    Downscaled because the model is being asked about composition — whether the
    borders balance, whether the field reads as cloth — and that survives at
    640px while the token cost does not. Greyscale because the design is
    already one bit per thread; colour would be three channels of the same
    information.
    """
    import base64
    import io as _io
    im = img.convert('L')
    if max(im.size) > max_px:
        scale = max_px / max(im.size)
        im = im.resize((max(1, int(im.size[0] * scale)),
                        max(1, int(im.size[1] * scale))))
    buf = _io.BytesIO()
    im.save(buf, format=fmt)
    return {'media_type': f'image/{fmt.lower()}',
            'data': base64.b64encode(buf.getvalue()).decode()}
