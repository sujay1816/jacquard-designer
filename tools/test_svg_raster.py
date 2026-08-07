"""
Tests for the built-in SVG rasteriser.

This exists so the product does not need Cairo. cairosvg is a general SVG
renderer bound to a C library pip cannot install — a GTK runtime on Windows,
brew plus a DYLD path variable on Apple Silicon — and this project does not have
general SVG. It has SVG that motif_library wrote milliseconds earlier, out of
six element types and four path commands.

What is checked: that the built-in renderer agrees with cairosvg closely enough
that a weaver could not tell which drew their design, and that it still draws
correctly when cairosvg is absent entirely.

Run:  python tools/test_svg_raster.py
"""
import io
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import motif_library as ml                                  # noqa: E402
import svg_raster as sr                                     # noqa: E402

PASS = FAIL = 0

try:
    import cairosvg
    HAVE_CAIRO = True
except Exception:
    HAVE_CAIRO = False


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def natural(svg, pins):
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    vw, vh = (int(m.group(1)), int(m.group(2))) if m else (1000, 1000)
    return pins, int(round(pins * vh / vw))


def ink(img):
    return (np.asarray(img.convert('L')) < 128)


def cairo_render(svg, w, h):
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=w,
                           output_height=h, background_color='white')
    return Image.open(io.BytesIO(png)).convert('RGB')


def main():
    print('\nEvery motif matches cairosvg')
    if not HAVE_CAIRO:
        print('  (cairosvg not installed here — comparison skipped, which is')
        print('   itself the point: everything below still works without it)')
    for name in ml.MOTIFS:
        svg = ml.build_svg(name, 400)
        w, h = natural(svg, 400)
        ours = ink(sr.render(svg, w, h))
        if HAVE_CAIRO:
            theirs = ink(cairo_render(svg, w, h))
            d = abs(ours.mean() - theirs.mean()) * 100
            agree = (ours == theirs).mean() * 100
            # Thresholds sized to what a weaver could notice: a few percent of
            # ink is well inside the variation between two conversion settings.
            check(f'{name}: ink within 5% of cairo ({d:.1f}%)', d < 5.0, d)
            check(f'{name}: pixels agree above 95% ({agree:.1f}%)', agree > 95.0, agree)
        else:
            check(f'{name}: renders with ink', 0.001 < ours.mean() < 0.9, ours.mean())

    print('\nEvery layout matches')
    for layout in ('jaal', 'half_drop', 'straight', 'banded'):
        svg = ml.allover(480, layout=layout, motif='paisley', cols=5, rows=6)
        w, h = natural(svg, 480)
        ours = ink(sr.render(svg, w, h))
        if HAVE_CAIRO:
            theirs = ink(cairo_render(svg, w, h))
            check(f'{layout}: agrees above 95%',
                  (ours == theirs).mean() > 0.95, (ours == theirs).mean())
        else:
            check(f'{layout}: renders with ink', ours.mean() > 0.001)

    print('\nThe SVG features this project actually emits')
    cases = {
        'rect fill': '<svg viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="black"/></svg>',
        'circle fill': '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="black"/></svg>',
        'line stroke': '<svg viewBox="0 0 100 100"><line x1="0" y1="50" x2="100" y2="50" stroke="black" stroke-width="10"/></svg>',
        'polyline': '<svg viewBox="0 0 100 100"><polyline points="0,80 50,20 100,80" fill="none" stroke="black" stroke-width="8"/></svg>',
        'cubic path': '<svg viewBox="0 0 100 100"><path d="M10 50 C 30 10, 70 90, 90 50" fill="none" stroke="black" stroke-width="8"/></svg>',
        'quadratic path': '<svg viewBox="0 0 100 100"><path d="M10 50 Q 50 10, 90 50" fill="none" stroke="black" stroke-width="8"/></svg>',
        'closed filled path': '<svg viewBox="0 0 100 100"><path d="M20 20 L80 20 L50 80 Z" fill="black"/></svg>',
        'translate': '<svg viewBox="0 0 100 100"><g transform="translate(50,0)"><rect width="50" height="100" fill="black"/></g></svg>',
        'scale': '<svg viewBox="0 0 100 100"><g transform="scale(0.5)"><rect width="100" height="100" fill="black"/></g></svg>',
        'nested groups': '<svg viewBox="0 0 100 100"><g transform="translate(25,25)"><g transform="scale(0.5)"><rect width="100" height="100" fill="black"/></g></g></svg>',
    }
    for label, svg in cases.items():
        img = sr.render(svg, 200, 200)
        coverage = ink(img).mean()
        check(f'{label} draws something', 0.005 < coverage < 0.95, coverage)

    print('\nInheritance and paint rules')
    # fill="none" on a group must reach the paths inside it. Getting this wrong
    # fills every outlined vine solid, which is what a bad renderer does to a
    # border and why the ink count triples.
    none_fill = sr.render(
        '<svg viewBox="0 0 100 100"><g fill="none" stroke="black" stroke-width="4">'
        '<path d="M10 10 L90 10 L90 90 L10 90 Z"/></g></svg>', 200, 200)
    solid = sr.render(
        '<svg viewBox="0 0 100 100"><path d="M10 10 L90 10 L90 90 L10 90 Z" '
        'fill="black"/></svg>', 200, 200)
    check('fill="none" is inherited by children',
          ink(none_fill).mean() < ink(solid).mean() / 3,
          (ink(none_fill).mean(), ink(solid).mean()))
    check('an outlined square is mostly empty', ink(none_fill).mean() < 0.25)

    print('\nGeometry')
    img = sr.render('<svg viewBox="0 0 1000 260"><rect width="1000" height="260" '
                    'fill="black"/></svg>', 500, 130)
    check('output is exactly the size asked for', img.size == (500, 130), img.size)
    check('a full-bleed rect covers everything', ink(img).mean() > 0.99)
    check('a viewBox offset is honoured',
          ink(sr.render('<svg viewBox="50 50 50 50"><rect x="50" y="50" '
                        'width="50" height="50" fill="black"/></svg>', 100, 100)
              ).mean() > 0.99)

    print('\nStroke width scales with the drawing')
    thin = ink(sr.render('<svg viewBox="0 0 100 100"><line x1="0" y1="50" x2="100" '
                         'y2="50" stroke="black" stroke-width="2"/></svg>', 200, 200)).mean()
    thick = ink(sr.render('<svg viewBox="0 0 100 100"><line x1="0" y1="50" x2="100" '
                          'y2="50" stroke="black" stroke-width="20"/></svg>', 200, 200)).mean()
    check('a 10x wider stroke lays down much more ink', thick > thin * 4, (thin, thick))

    print('\nIt is deterministic')
    a = sr.render(ml.build_svg('paisley', 300), 300, 300)
    b = sr.render(ml.build_svg('paisley', 300), 300, 300)
    check('the same svg renders identically', a.tobytes() == b.tobytes())

    print('\nmotif_library works with cairosvg unavailable')
    saved = ml._RENDERER
    try:
        ml._RENDERER = 'builtin'
        img = ml.render(ml.build_svg('lotus', 320), 320)
        check('a motif still renders', img.size == (320, 320), img.size)
        check('and carries ink', 0.01 < ink(img).mean() < 0.9, ink(img).mean())

        import design_studio as ds
        spec = ds.LayoutSpec(**ds.plan(pins=400, reed=80, pallu=True)['spec'])
        panel = ds.render(spec)
        check('a whole panel still composes', panel.size[0] == 400, panel.size)
        check('with borders and body', 0.01 < ink(panel).mean() < 0.9, ink(panel).mean())
        check('the renderer reports itself honestly', ml.renderer_name() == 'builtin')
    finally:
        ml._RENDERER = saved

    print('\nMalformed input does not crash the renderer')
    for label, svg in (('empty svg', '<svg viewBox="0 0 10 10"></svg>'),
                       ('no viewBox', '<svg width="50" height="50"><rect width="50" '
                                      'height="50" fill="black"/></svg>'),
                       ('unknown element', '<svg viewBox="0 0 10 10"><foo bar="1"/></svg>'),
                       ('truncated path', '<svg viewBox="0 0 10 10"><path d="M1 1 C 2"/></svg>'),
                       ('empty path', '<svg viewBox="0 0 10 10"><path d=""/></svg>'),
                       ('bad colour', '<svg viewBox="0 0 10 10"><rect width="10" '
                                      'height="10" fill="notacolour"/></svg>')):
        try:
            sr.render(svg, 50, 50)
            check(f'{label} is survived', True)
        except Exception as e:
            check(f'{label} is survived', False, f'{type(e).__name__}: {e}')

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
