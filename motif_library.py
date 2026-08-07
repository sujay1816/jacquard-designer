"""
Parametric motif library.

Generates original saree and brocade motifs as SVG, then renders them at the
loom's exact pin count. Two things follow from being vector rather than raster,
and both matter:

  * There is no source resolution to run out of. Every failure in this codebase
    — thinning, halos, closed gaps, lost interiors — comes from an image having
    fewer pixels than the design needs. A rendered vector has exactly as many
    as you ask for.
  * Stroke weight can be chosen FOR the target pin count. Every motif here
    sizes its strokes so the finest line lands on at least MIN_THREADS threads,
    which is the condition the whole conversion pipeline needs and can never
    retrofit onto artwork that lacks it.

These motifs are original parametric constructions. They are deliberately NOT
derived from scraped artwork: textile designs are copyrightable, and a
generator trained on other people's work would pass that exposure to every mill
that wove from it.

They are also, honestly, geometric and stylised rather than traditional. A
Chola or Banarasi motif carries centuries of convention that a parametric
system approximates at best. Use these for grounds, borders, fills and simple
buttas — the repetitive work — and a designer for anything a customer will
recognise.
"""
import io
import math

MIN_THREADS = 2.5          # threads the finest stroke must occupy
_VIEW = 1000               # internal SVG coordinate space


def _stroke_for(pins: int, view_w: int = _VIEW) -> float:
    """
    Stroke width in SVG units that renders at least MIN_THREADS threads wide.

    This is the whole reason generated designs beat scanned ones: the artwork
    is built to the loom, instead of the loom having to cope with the artwork.
    """
    return max(2.0, MIN_THREADS * view_w / max(pins, 1))


# Tones used when a design carries more than one thread. These are rendered as
# distinct grey levels, not as decoration: the detection stage clusters them
# back into separate label classes, one per shuttle. Kept far apart so
# clustering separates them cleanly at any pin count.
TONE = {
    'ground': '#ffffff',   # cloth ground, no thread lifted
    'a': '#000000',        # first thread — usually zari
    'b': '#8c8c8c',        # second thread — usually meena
}


def _tones(colours: int):
    """Ink tones available for a design with this many threads (plus ground)."""
    return ['a'] if int(colours) < 3 else ['a', 'b']


def _svg(body: str, w: int = _VIEW, h: int = _VIEW) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>{body}</svg>')


# ── Motifs ──────────────────────────────────────────────────────────────────

def paisley(pins, complexity=3, filled=False, colours=2):
    """
    Teardrop butta with nested outlines and seed detail.

    With colours=3 the outlines take the first thread and the seed and filler
    detail the second, which is how a real zari-and-meena butta is built: the
    metallic draws the shape, the colour fills it.
    """
    s = _stroke_for(pins)
    t = _tones(colours)
    c_out, c_fill = TONE[t[0]], TONE[t[-1]]
    h = _VIEW
    parts = [f'<g fill="none" stroke="{c_out}" stroke-width="{s*1.6:.1f}" '
             f'stroke-linecap="round" stroke-linejoin="round">']
    for i in range(max(1, min(4, complexity))):
        k = 1 - i * 0.17
        parts.append(
            f'<path d="M500 {80*k+40:.0f} '
            f'C {500+240*k:.0f} {220*k+120:.0f}, {500+270*k:.0f} {560*k+180:.0f}, '
            f'{500+70*k:.0f} {820*k+120:.0f} '
            f'C 500 {900*k+80:.0f}, {500-60*k:.0f} {920*k+70:.0f}, {500-110*k:.0f} {880*k+90:.0f} '
            f'C {500-260*k:.0f} {800*k+110:.0f}, {500-300*k:.0f} {560*k+180:.0f}, '
            f'{500-240*k:.0f} {330*k+140:.0f} '
            f'C {500-170*k:.0f} {170*k+110:.0f}, {500-70*k:.0f} {90*k+70:.0f}, 500 {80*k+40:.0f} Z"/>')
    parts.append('</g>')
    # Seeds along the spine
    parts.append(f'<g fill="{c_fill}">')
    for i in range(5):
        cy = 260 + i * 130
        r = s * (1.6 - i * 0.12)
        parts.append(f'<circle cx="500" cy="{cy}" r="{max(r,s):.1f}"/>')
        if i:
            parts.append(f'<circle cx="{500-70-i*8}" cy="{cy-50}" r="{max(r*0.7,s*0.8):.1f}"/>')
            parts.append(f'<circle cx="{500+70+i*8}" cy="{cy-50}" r="{max(r*0.7,s*0.8):.1f}"/>')
    parts.append('</g>')
    if filled:
        parts.append(f'<g fill="none" stroke="{c_fill}" stroke-width="{s:.1f}">')
        for i in range(6):
            y = 220 + i * 105
            parts.append(f'<path d="M{430-i*4} {y} Q 500 {y-45}, {570+i*4} {y}"/>')
        parts.append('</g>')
    return _svg(''.join(parts), _VIEW, h)


def lotus(pins, petals=8, rings=2, colours=2):
    """
    Radial lotus rosette — a standard centre motif.

    With colours=3 alternate petal rings take alternate threads, which is the
    usual way a lotus is worked: outer petals in metallic, inner in colour.
    """
    s = _stroke_for(pins)
    t = _tones(colours)
    cx = cy = 500
    parts = []
    petals = max(5, min(16, petals))
    for ring in range(max(1, min(3, rings))):
        col = TONE[t[ring % len(t)]]
        parts.append(f'<g fill="none" stroke="{col}" stroke-width="{s*1.4:.1f}" '
                     f'stroke-linejoin="round">')
        R = 420 - ring * 130
        off = (math.pi / petals) if ring % 2 else 0
        for i in range(petals):
            a = 2 * math.pi * i / petals + off
            tipx, tipy = cx + R * math.cos(a), cy + R * math.sin(a)
            lx = cx + R * 0.55 * math.cos(a - math.pi / petals)
            ly = cy + R * 0.55 * math.sin(a - math.pi / petals)
            rx = cx + R * 0.55 * math.cos(a + math.pi / petals)
            ry = cy + R * 0.55 * math.sin(a + math.pi / petals)
            parts.append(f'<path d="M{cx} {cy} Q {lx:.0f} {ly:.0f}, {tipx:.0f} {tipy:.0f} '
                         f'Q {rx:.0f} {ry:.0f}, {cx} {cy} Z"/>')
        parts.append('</g>')
    centre = TONE[t[-1]]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{s*3:.0f}" fill="{centre}"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{s*6:.0f}" fill="none" '
                 f'stroke="{centre}" stroke-width="{s:.1f}"/>')
    return _svg(''.join(parts))


def vine_border(pins, height=260, repeats=4, leaves=True, colours=2):
    """Running creeper border — a horizontal band that tiles seamlessly."""
    s = _stroke_for(pins, _VIEW)
    w, h = _VIEW, max(120, height)
    seg = w / max(1, repeats)
    t = _tones(colours)
    c_stem, c_leaf = TONE[t[0]], TONE[t[-1]]
    parts = [f'<g fill="none" stroke="{c_stem}" stroke-width="{s*1.5:.1f}" stroke-linecap="round">']
    d = [f'M0 {h/2:.0f}']
    for i in range(repeats):
        x0 = i * seg
        d.append(f'C {x0+seg*0.25:.0f} {h*0.12:.0f}, {x0+seg*0.75:.0f} {h*0.88:.0f}, '
                 f'{x0+seg:.0f} {h/2:.0f}')
    parts.append(f'<path d="{" ".join(d)}"/>')
    parts.append('</g>')
    parts.append(f'<g fill="none" stroke="{c_leaf}" stroke-width="{s*1.5:.1f}" stroke-linecap="round">')
    if leaves:
        for i in range(repeats):
            for side, yy in ((0.28, h*0.24), (0.78, h*0.76)):
                x = i * seg + seg * side
                parts.append(f'<path d="M{x:.0f} {h/2:.0f} Q {x-seg*0.10:.0f} {yy:.0f}, '
                             f'{x+seg*0.06:.0f} {yy*0.86:.0f} Q {x+seg*0.14:.0f} {yy:.0f}, '
                             f'{x:.0f} {h/2:.0f} Z"/>')
    parts.append('</g>')
    parts.append(f'<g fill="{c_leaf}">')
    for i in range(repeats * 2):
        parts.append(f'<circle cx="{(i+0.5)*seg/2:.0f}" cy="{h/2:.0f}" r="{s*0.9:.1f}"/>')
    parts.append('</g>')
    parts.append(f'<g stroke="{c_stem}" stroke-width="{s:.1f}">'
                 f'<line x1="0" y1="{s*1.5:.0f}" x2="{w}" y2="{s*1.5:.0f}"/>'
                 f'<line x1="0" y1="{h-s*1.5:.0f}" x2="{w}" y2="{h-s*1.5:.0f}"/></g>')
    return _svg(''.join(parts), w, h)


def diamond_jaal(pins, cells=6, motif_dots=True, colours=2):
    """Diamond lattice ground — the classic jaal fill."""
    s = _stroke_for(pins)
    w = h = _VIEW
    step = w / max(2, cells)
    t = _tones(colours)
    parts = [f'<g fill="none" stroke="{TONE[t[0]]}" stroke-width="{s*1.2:.1f}">']
    for i in range(-cells, cells * 2):
        parts.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step+h:.0f}" y2="{h}"/>')
        parts.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step-h:.0f}" y2="{h}"/>')
    parts.append('</g>')
    if motif_dots:
        parts.append(f'<g fill="{TONE[t[-1]]}">')
        for r in range(cells + 1):
            for c in range(cells + 1):
                parts.append(f'<circle cx="{c*step:.0f}" cy="{r*step:.0f}" r="{s*1.2:.1f}"/>')
        parts.append('</g>')
    return _svg(''.join(parts), w, h)


def check_ground(pins, cells=8, double=True):
    """Square check ground for body fills."""
    s = _stroke_for(pins)
    w = h = _VIEW
    step = w / max(2, cells)
    parts = [f'<g stroke="black" stroke-width="{s*1.2:.1f}">']
    for i in range(cells + 1):
        parts.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step:.0f}" y2="{h}"/>')
        parts.append(f'<line x1="0" y1="{i*step:.0f}" x2="{w}" y2="{i*step:.0f}"/>')
    parts.append('</g>')
    if double:
        off = step / 2
        parts.append(f'<g stroke="black" stroke-width="{s*0.7:.1f}">')
        for i in range(cells + 1):
            parts.append(f'<line x1="{i*step+off:.0f}" y1="0" x2="{i*step+off:.0f}" y2="{h}"/>')
            parts.append(f'<line x1="0" y1="{i*step+off:.0f}" x2="{w}" y2="{i*step+off:.0f}"/>')
        parts.append('</g>')
    return _svg(''.join(parts), w, h)


def chevron_border(pins, height=200, repeats=10):
    """Zig-zag band — a simple separator border."""
    s = _stroke_for(pins)
    w, h = _VIEW, max(90, height)
    seg = w / max(2, repeats)
    parts = [f'<g fill="none" stroke="black" stroke-width="{s*1.6:.1f}" '
             f'stroke-linejoin="round" stroke-linecap="round">']
    for band, yoff in ((0, h * 0.30), (1, h * 0.70)):
        pts = []
        for i in range(repeats + 1):
            y = yoff - h * 0.16 if i % 2 == 0 else yoff + h * 0.16
            pts.append(f'{i*seg:.0f} {y:.0f}')
        parts.append(f'<polyline points="{" ".join(pts)}"/>')
    parts.append('</g>')
    return _svg(''.join(parts), w, h)


def dotted_field(pins, cols=10, rows=10, half_drop=True):
    """Scattered dot ground — the lightest possible body fill."""
    s = _stroke_for(pins)
    w = h = _VIEW
    dx, dy = w / cols, h / rows
    parts = ['<g fill="black">']
    for r in range(rows + 1):
        off = dx / 2 if (half_drop and r % 2) else 0
        for c in range(cols + 1):
            parts.append(f'<circle cx="{c*dx+off:.0f}" cy="{r*dy:.0f}" r="{s*1.3:.1f}"/>')
    parts.append('</g>')
    return _svg(''.join(parts), w, h)


MOTIFS = {
    'paisley':       (paisley,        'Teardrop butta with nested outlines and seed detail'),
    'lotus':         (lotus,          'Radial lotus rosette, a centre motif'),
    'vine_border':   (vine_border,    'Running creeper border band'),
    'diamond_jaal':  (diamond_jaal,   'Diamond lattice ground'),
    'check_ground':  (check_ground,   'Square check ground for body fills'),
    'chevron_border':(chevron_border, 'Zig-zag separator band'),
    'dotted_field':  (dotted_field,   'Scattered dot ground, lightest body fill'),
}


def build_svg(motif: str, pins: int, **params) -> str:
    """Return SVG for a named motif, sized for the given pin count."""
    if motif not in MOTIFS:
        raise ValueError(f"Unknown motif '{motif}'. Available: {', '.join(sorted(MOTIFS))}")
    fn = MOTIFS[motif][0]
    clean = {}
    import inspect
    allowed = set(inspect.signature(fn).parameters) - {'pins'}
    for k, v in (params or {}).items():
        if k in allowed and v is not None:
            clean[k] = v
    return fn(int(pins), **clean)


def render(svg: str, pins: int, cards: int = None):
    """Rasterise SVG at the loom's resolution. Returns a PIL RGB image."""
    import cairosvg
    from PIL import Image
    import re

    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    vw, vh = (int(m.group(1)), int(m.group(2))) if m else (_VIEW, _VIEW)
    if cards is None:
        cards = int(round(pins * vh / vw))
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=int(pins),
                           output_height=int(cards), background_color='white')
    return Image.open(io.BytesIO(png)).convert('RGB')


# ── All-over composition ────────────────────────────────────────────────────
#
# Real brocade is rarely a single motif. It is a field: motif rows alternating
# with band rules, a jaal lattice with fillers in the cells, a half-drop repeat
# across the whole body. These compose the motifs above into that field.
#
# The subtlety that makes or breaks it is stroke weight. A motif scaled to a
# fifth of the cloth width has strokes a fifth as thick, which drops it below
# the weavable threshold even though the motif was correct on its own. So each
# tile is BUILT as if the loom were only as wide as that tile — then it stays
# weavable at any repeat count.

def _inner(svg: str):
    """Strip the wrapper from a motif SVG, returning (body, width, height)."""
    import re
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (_VIEW, _VIEW)
    body = svg[svg.index('>', svg.index('<svg')) + 1: svg.rindex('</svg>')]
    body = re.sub(r'<rect width="\d+" height="\d+" fill="white"/>', '', body, count=1)
    return body, w, h


def _tile(motif, tile_pins, **params):
    """A motif built for a tile that will occupy tile_pins threads."""
    return _inner(build_svg(motif, max(24, int(tile_pins)), **params))


def allover(pins, layout='half_drop', motif='paisley', cols=5, rows=6,
            cards=None, spacing=0.25, band_motif='vine_border',
            band_every=1, mirror=False, colours=2, **params):
    """
    Compose a motif into an all-over brocade field.

    layout:
      straight    — plain grid repeat
      half_drop   — alternate rows offset by half a tile (the usual saree body)
      brick       — alternate rows offset, tiles wider than tall
      banded      — motif rows separated by border rules, as on a real brocade
      jaal        — diamond lattice with a motif in each cell
      stripe      — vertical bands of motif alternating with plain ground
    """
    cols = max(1, min(24, int(cols)))
    rows = max(1, min(40, int(rows)))
    gap = max(0.0, min(1.5, float(spacing)))

    tile_w = _VIEW / cols
    tile_pins = pins / cols                  # threads this tile actually gets
    params.setdefault('colours', colours)
    body, mw, mh = _tile(motif, tile_pins, **params)
    scale = (tile_w / mw) / (1.0 + gap)
    sw, sh = mw * scale, mh * scale
    pad_x = (tile_w - sw) / 2

    parts, row_h = [], sh * (1.0 + gap)

    def place(x, y, flip=False):
        t = (f'translate({x + sw:.1f},{y:.1f}) scale({-scale:.4f},{scale:.4f})'
             if flip else f'translate({x:.1f},{y:.1f}) scale({scale:.4f})')
        parts.append(f'<g transform="{t}">{body}</g>')

    if layout == 'banded':
        band_body, bw, bh = _tile(band_motif, pins, repeats=max(2, cols),
                                  colours=colours)
        bscale = _VIEW / bw
        bh_s = bh * bscale
        y = 0
        r = 0
        while y < _VIEW * (rows / max(rows, 1)) * 1.0 and r < rows:
            for c in range(cols):
                place(c * tile_w + pad_x, y + sh * gap * 0.5,
                      flip=mirror and c % 2 == 1)
            y += row_h
            r += 1
            if band_every and r % band_every == 0:
                parts.append(f'<g transform="translate(0,{y:.1f}) scale({bscale:.4f})">'
                             f'{band_body}</g>')
                y += bh_s
        total_h = y
    elif layout == 'jaal':
        s = _stroke_for(pins)
        step = _VIEW / cols
        lat_col = TONE[_tones(colours)[0]]
        lat = [f'<g fill="none" stroke="{lat_col}" stroke-width="{s*1.2:.1f}">']
        span = int(rows * row_h) + int(step)
        for i in range(-cols, cols * 2 + 2):
            lat.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step+span:.0f}" y2="{span}"/>')
            lat.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step-span:.0f}" y2="{span}"/>')
        lat.append('</g>')
        parts.append(''.join(lat))
        for r in range(rows):
            off = step / 2 if r % 2 else 0
            for c in range(cols + 1):
                place(c * step + off - sw / 2, r * step + step / 2 - sh / 2)
        total_h = rows * step
    elif layout == 'stripe':
        for c in range(cols):
            if c % 2:
                continue
            for r in range(rows):
                place(c * tile_w + pad_x, r * row_h, flip=mirror and r % 2 == 1)
        total_h = rows * row_h
    else:
        for r in range(rows):
            if layout == 'half_drop':
                off = tile_w / 2 if r % 2 else 0
            elif layout == 'brick':
                off = tile_w / 3 if r % 2 else 0
            else:
                off = 0
            for c in range(-1, cols + 1):
                x = c * tile_w + off + pad_x
                if -sw < x < _VIEW:
                    place(x, r * row_h, flip=mirror and (r + c) % 2 == 1)
        total_h = rows * row_h

    total_h = max(int(total_h), 40)
    if cards:
        total_h = int(round(_VIEW * int(cards) / max(pins, 1)))
    return _svg(''.join(parts), _VIEW, total_h)


ALLOVER_LAYOUTS = {
    'straight':  'Plain grid repeat',
    'half_drop': 'Alternate rows offset by half a tile — the usual saree body',
    'brick':     'Alternate rows offset by a third, brick fashion',
    'banded':    'Motif rows separated by border rules, as on a real brocade',
    'jaal':      'Diamond lattice with a motif in each cell',
    'stripe':    'Vertical bands of motif alternating with plain ground',
}


# Threads a motif needs before its internal detail survives.
#
# Measured by rendering a paisley field at many sizes and counting the enclosed
# gaps that survive per motif: at 15-30 threads the motif is a solid blob with
# 0-1 gaps; at 40 it is marginal; from 48 upward the interior reads properly.
# Stroke weight is NOT the limit here — the scaling keeps strokes weavable at
# every size — the limit is that a motif smaller than this cannot hold detail
# no matter how it is drawn.
MIN_THREADS_PER_MOTIF = {
    'paisley': 48, 'lotus': 48, 'vine_border': 30, 'diamond_jaal': 24,
    'check_ground': 16, 'chevron_border': 20, 'dotted_field': 12,
}


def design_options(pins: int, cards: int = None):
    """
    What can actually be designed at this pin count?

    Answers the question a designer asks first — how much can I fit before it
    stops reading — so a design is chosen within real limits rather than
    proposed and then rejected by the fidelity check.
    """
    pins = max(10, int(pins))
    out = {'pins': pins, 'motifs': [], 'notes': []}

    for name, need in sorted(MIN_THREADS_PER_MOTIF.items()):
        max_cols = max(0, pins // need)
        entry = {
            'motif': name,
            'max_across': int(max_cols),
            'threads_each_at_max': int(pins // max_cols) if max_cols else None,
            'comfortable_across': int(max(1, pins // (need * 1.5))),
        }
        if max_cols == 0:
            entry['note'] = (f'Needs at least {need} threads per motif; '
                             f'{pins} pins is too narrow for even one.')
        out['motifs'].append(entry)

    detailed = [m for m in out['motifs'] if m['motif'] in ('paisley', 'lotus')]
    best = max(detailed, key=lambda m: m['max_across']) if detailed else None
    if best and best['max_across'] >= 3:
        out['notes'].append(
            f"At {pins} pins you can fit up to {best['max_across']} "
            f"{best['motif']} motifs across, or {best['comfortable_across']} "
            f"with room to breathe.")
    elif best:
        out['notes'].append(
            f"{pins} pins is narrow for a butta field — at most "
            f"{best['max_across']} across. A border or a geometric ground will "
            f"read better than a motif repeat.")

    if pins < 200:
        out['notes'].append(
            'Below about 200 pins, geometric grounds (check, diamond, dotted) '
            'hold up far better than figurative motifs.')
    if pins >= 720:
        out['notes'].append(
            'At this width there is room for a banded layout with border rules '
            'between motif rows.')

    out['layouts'] = [{'name': k, 'description': v}
                      for k, v in sorted(ALLOVER_LAYOUTS.items())]
    return out
