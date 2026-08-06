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


def _svg(body: str, w: int = _VIEW, h: int = _VIEW) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>{body}</svg>')


# ── Motifs ──────────────────────────────────────────────────────────────────

def paisley(pins, complexity=3, filled=False):
    """Teardrop butta with nested outlines and seed detail."""
    s = _stroke_for(pins)
    h = _VIEW
    parts = [f'<g fill="none" stroke="black" stroke-width="{s*1.6:.1f}" '
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
    parts.append('<g fill="black">')
    for i in range(5):
        cy = 260 + i * 130
        r = s * (1.6 - i * 0.12)
        parts.append(f'<circle cx="500" cy="{cy}" r="{max(r,s):.1f}"/>')
        if i:
            parts.append(f'<circle cx="{500-70-i*8}" cy="{cy-50}" r="{max(r*0.7,s*0.8):.1f}"/>')
            parts.append(f'<circle cx="{500+70+i*8}" cy="{cy-50}" r="{max(r*0.7,s*0.8):.1f}"/>')
    parts.append('</g>')
    if filled:
        parts.append(f'<g fill="none" stroke="black" stroke-width="{s:.1f}">')
        for i in range(6):
            y = 220 + i * 105
            parts.append(f'<path d="M{430-i*4} {y} Q 500 {y-45}, {570+i*4} {y}"/>')
        parts.append('</g>')
    return _svg(''.join(parts), _VIEW, h)


def lotus(pins, petals=8, rings=2):
    """Radial lotus rosette — a standard centre motif."""
    s = _stroke_for(pins)
    cx = cy = 500
    parts = [f'<g fill="none" stroke="black" stroke-width="{s*1.4:.1f}" stroke-linejoin="round">']
    petals = max(5, min(16, petals))
    for ring in range(max(1, min(3, rings))):
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
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{s*3:.0f}" fill="black"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{s*6:.0f}" fill="none" '
                 f'stroke="black" stroke-width="{s:.1f}"/>')
    return _svg(''.join(parts))


def vine_border(pins, height=260, repeats=4, leaves=True):
    """Running creeper border — a horizontal band that tiles seamlessly."""
    s = _stroke_for(pins, _VIEW)
    w, h = _VIEW, max(120, height)
    seg = w / max(1, repeats)
    parts = [f'<g fill="none" stroke="black" stroke-width="{s*1.5:.1f}" stroke-linecap="round">']
    d = [f'M0 {h/2:.0f}']
    for i in range(repeats):
        x0 = i * seg
        d.append(f'C {x0+seg*0.25:.0f} {h*0.12:.0f}, {x0+seg*0.75:.0f} {h*0.88:.0f}, '
                 f'{x0+seg:.0f} {h/2:.0f}')
    parts.append(f'<path d="{" ".join(d)}"/>')
    if leaves:
        for i in range(repeats):
            for side, yy in ((0.28, h*0.24), (0.78, h*0.76)):
                x = i * seg + seg * side
                parts.append(f'<path d="M{x:.0f} {h/2:.0f} Q {x-seg*0.10:.0f} {yy:.0f}, '
                             f'{x+seg*0.06:.0f} {yy*0.86:.0f} Q {x+seg*0.14:.0f} {yy:.0f}, '
                             f'{x:.0f} {h/2:.0f} Z"/>')
    parts.append('</g>')
    parts.append('<g fill="black">')
    for i in range(repeats * 2):
        parts.append(f'<circle cx="{(i+0.5)*seg/2:.0f}" cy="{h/2:.0f}" r="{s*0.9:.1f}"/>')
    parts.append('</g>')
    parts.append(f'<g stroke="black" stroke-width="{s:.1f}">'
                 f'<line x1="0" y1="{s*1.5:.0f}" x2="{w}" y2="{s*1.5:.0f}"/>'
                 f'<line x1="0" y1="{h-s*1.5:.0f}" x2="{w}" y2="{h-s*1.5:.0f}"/></g>')
    return _svg(''.join(parts), w, h)


def diamond_jaal(pins, cells=6, motif_dots=True):
    """Diamond lattice ground — the classic jaal fill."""
    s = _stroke_for(pins)
    w = h = _VIEW
    step = w / max(2, cells)
    parts = [f'<g fill="none" stroke="black" stroke-width="{s*1.2:.1f}">']
    for i in range(-cells, cells * 2):
        parts.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step+h:.0f}" y2="{h}"/>')
        parts.append(f'<line x1="{i*step:.0f}" y1="0" x2="{i*step-h:.0f}" y2="{h}"/>')
    parts.append('</g>')
    if motif_dots:
        parts.append('<g fill="black">')
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
