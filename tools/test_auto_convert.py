"""
Tests for auto-convert — the self-checking conversion loop.

Guards the behaviour that makes it worth having: it must never return a worse
result than the plain single-shot path, it must respect a fixed pin count, and
when it cannot succeed it must say so rather than hand over a plausible file.

Run:  python tools/test_auto_convert.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_convert import auto_convert, _score          # noqa: E402
from fidelity import fidelity_report                   # noqa: E402
from vision_engine import detect_colors_smart          # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def synth(w=600, h=400):
    """Line art with thin strokes and enclosed interior gaps."""
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.ellipse((60 + i * 85, 90, 130 + i * 85, 300), outline='black', width=3)
        d.ellipse((80 + i * 85, 140, 110 + i * 85, 250), outline='black', width=2)
    for y in (40, 350):
        d.line((0, y, w, y), fill='black', width=3)
    return img


def main():
    img = synth()

    print('\nRespects a fixed pin count')
    r = auto_convert(img, pins=300, n_colors=2)
    check('returns the requested pins', r['best']['pins'] == 300, r['best']['pins'])
    check('does not offer other pin counts when fixed',
          all(a['pins'] == 300 for a in r['alternatives']), r['alternatives'])

    print('\nNever worse than a single shot')
    for pins in (300, 480):
        cards = int(round(pins * img.size[1] / img.size[0]))
        _, _, lm, _ = detect_colors_smart(img, 2, pins, cards)
        base = _score(fidelity_report(img, np.asarray(lm) > 0))
        got = auto_convert(img, pins=pins, n_colors=2)['best']['score']
        check(f'@{pins} pins at least matches the default', got <= base,
              f'auto {got} vs default {base}')

    print('\nSearches when no pin count is given')
    r = auto_convert(img, n_colors=2)
    check('picks a pin count', isinstance(r['best']['pins'], int))
    check('offers alternatives', len(r['alternatives']) >= 1)
    check('alternatives are distinct pin counts',
          len({a['pins'] for a in r['alternatives']}) == len(r['alternatives']))
    check('best scores at least as well as every alternative',
          all(r['best']['score'] <= a['score'] for a in r['alternatives']))

    print('\nAlways explains itself')
    check('has a summary', bool(r['summary']))
    check('has advice', bool(r['advice']))
    check('reports how many settings were tried', r['attempts'] >= 1)

    print('\nDeterministic')
    a = auto_convert(img, pins=360, n_colors=2)
    b = auto_convert(img, pins=360, n_colors=2)
    check('same input gives the same settings',
          a['best']['settings'] == b['best']['settings'])
    check('same input gives the same label map',
          np.array_equal(np.asarray(a['best']['label_map']),
                         np.asarray(b['best']['label_map'])))

    print('\nHonest when it cannot succeed')
    tiny = img.resize((90, 60), Image.LANCZOS)
    r = auto_convert(tiny, pins=600, n_colors=2)
    check('does not claim OK on an impossible job',
          r['verdict'] in ('warn', 'fail'), r['verdict'])
    check('says what would actually help',
          any('source' in m.lower() or 'pins' in m.lower() for m in r['advice']),
          r['advice'])

    print('\nDegenerate input does not crash')
    blank = Image.new('RGB', (200, 200), 'white')
    r = auto_convert(blank, pins=100, n_colors=2)
    check('blank image returns a record', isinstance(r, dict) and 'verdict' in r)

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
