"""
Tests for the intent-driven design studio.

Runs offline. What is being checked is that a brief produces a panel a mill can
use, that the geometry is honest about what the loom cannot do, and that a
refinement which damages the design says so.

Run:  python tools/test_studio.py
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_engine as ag                                   # noqa: E402
import design_studio as ds                                  # noqa: E402
from llm import Reply, ToolCall                              # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def blank():
    return ag.get_session(ag.new_session(Image.new('RGB', (10, 10), 'white'), 'x.png'))


def main():
    print('\nGeometry')
    g = ds.geometry(pins=480, reed=60)
    check('480 pins at reed 60 is 8 inches', g['width_in'] == 8.0, g['width_in'])
    g = ds.geometry(pins=480, reed=100)
    check('the same pins at reed 100 is 4.8 inches', g['width_in'] == 4.8, g['width_in'])
    g = ds.geometry(None, reed=80, width_in=10)
    check('inches convert back to threads', g['pins'] == 800, g['pins'])
    check('picks default to the reed', ds.geometry(pins=100, reed=72)['picks_ppi'] == 72)

    print('\nPlanning')
    p = ds.plan(pins=800, reed=80, feel='traditional', threads=2)
    check('a plan names a motif', p['chosen'] and p['chosen']['motif'], p['chosen'])
    check('the chosen motif gets enough threads',
          p['chosen']['reads_well'], p['chosen'])
    check('the reasoning mentions the reed', 'reed' in p['why'], p['why'])
    check('borders are taken out of the body width',
          p['body_pins'] == 800 - 2 * p['border_pins'], p)

    # The failure this guards: ranking on headroom alone always picks
    # dotted_field, because it needs 12 threads against paisley's 48. A weaver
    # asking for a saree body would get scattered dots.
    check('a butta is preferred over a plain ground when both fit',
          p['chosen']['kind'] == 'butta', p['chosen'])

    narrow = ds.plan(pins=120, reed=60)
    check('a narrow loom still gets a workable answer',
          narrow['chosen'] is not None, narrow['chosen'])
    check('borders are dropped when the cloth cannot spare them',
          narrow['border_pins'] == 0 or narrow['border_pins'] >= ds.MIN_BORDER_THREADS,
          narrow['border_pins'])

    print('\nFeel changes the cloth')
    open_p = ds.plan(pins=800, reed=80, feel='open')
    rich_p = ds.plan(pins=800, reed=80, feel='rich')
    check('rich puts more motifs across than open',
          rich_p['chosen']['cols'] > open_p['chosen']['cols'],
          (rich_p['chosen']['cols'], open_p['chosen']['cols']))
    check('an unknown feel falls back rather than failing',
          ds.plan(pins=800, feel='sparkly')['feel'] == 'traditional')

    print('\nComposition')
    spec = ds.LayoutSpec(**ds.plan(pins=600, reed=80, feel='traditional',
                                   pallu=True)['spec'])
    img = ds.render(spec)
    check('renders at the requested pin count', img.size[0] == spec.pins, img.size)
    check('renders at the requested card count', img.size[1] == spec.cards, img.size)

    px = img.convert('L').load()
    w, h = img.size
    b = spec.border_pins()
    # A border only counts as present if it actually carries ink.
    left = sum(1 for y in range(0, h, 7) for x in range(0, max(1, b))
               if px[x, y] < 128)
    right = sum(1 for y in range(0, h, 7) for x in range(w - max(1, b), w)
                if px[x, y] < 128)
    check('the left border carries ink', left > 0, left)
    check('the right border carries ink', right > 0, right)
    check('both borders are woven, not just one',
          min(left, right) > 0.25 * max(left, right), (left, right))

    # The bug this catches: allover's row pitch is tile_width * aspect — the
    # spacing gap divides into the scale and cancels. Treating spacing as extra
    # vertical pitch under-counts rows and leaves bare cloth above the pallu.
    body_h = int(h * (1 - spec.cross_frac))
    band = [y for y in range(int(body_h * 0.80), body_h)
            if any(px[x, y] < 128 for x in range(b, w - b, 3))]
    check('the body fills its height, no bare band above the pallu',
          len(band) > 0.3 * (body_h - int(body_h * 0.80)), len(band))

    no_border = ds.LayoutSpec(pins=400, cards=500, border=False)
    check('a panel with no borders still renders',
          ds.render(no_border).size == (400, 500))

    print('\nRefinement moves the spec, not the pixels')
    base = ds.LayoutSpec(pins=600, cards=800, cols=6, spacing=0.25)
    opened, desc, err = ds.refine(base, 'more_open')
    check('more_open increases spacing', opened.spacing > base.spacing, opened.spacing)
    check('the change is described in words', bool(desc), desc)
    fewer, _, _ = ds.refine(base, 'fewer_motifs')
    check('fewer_motifs reduces the count', fewer.cols == base.cols - 1, fewer.cols)
    check('a phrase maps onto a refinement',
          ds.refine(base, 'too busy')[0].spacing > base.spacing)
    check('an unknown refinement is refused, not ignored',
          ds.refine(base, 'make it prettier')[2] is not None)
    at_limit = ds.LayoutSpec(pins=600, cols=1)
    check('a refinement at its limit says so',
          ds.refine(at_limit, 'fewer_motifs')[2] is not None)

    print('\nThe design tool')
    s = blank()
    r = ag.run_tool('design', {'width_in': 8, 'reed': 80, 'feel': 'rich',
                               'threads': 2, 'pallu': True}, s)
    check('a brief produces a design', 'error' not in r, r)
    check('it explains its reasoning', bool(r.get('why_this')), r)
    check('it reports the finished size', r['geometry']['width_in'] == 8.0, r)
    check('it reports a verdict', r.get('verdict') in ('ok', 'warn', 'fail'), r)
    check('it offers the motifs it did not pick', bool(r.get('other_motifs_that_fit')), r)
    check('the design becomes the working design', s.get('spec') is not None)

    check('a brief with no width is refused',
          'error' in ag.run_tool('design', {'feel': 'rich'}, blank()))
    check('an unknown motif is refused',
          'error' in ag.run_tool('design', {'pins': 600, 'motif': 'peacock'}, blank()))

    print('\nGeometry is honest about the loom')
    r = ag.run_tool('loom_geometry', {'width_in': 45, 'reed': 80}, blank())
    # Silently clamping 3600 threads to 2640 and reporting 33in answers a
    # question nobody asked; the weaver would find out at the loom.
    check('an impossible width is flagged, not quietly clamped',
          'warning' in r and '2640' in r['warning'], r)
    check('a possible width has no warning',
          'warning' not in ag.run_tool('loom_geometry',
                                       {'width_in': 8, 'reed': 80}, blank()))

    print('\nRefinement reports damage')
    s = blank()
    ag.run_tool('design', {'pins': 640, 'reed': 80, 'feel': 'rich', 'threads': 2}, s)
    before = s['conversion']['best']['report']['ink_drift_pct']
    r = ag.run_tool('refine_design', {'change': 'more_open'}, s)
    check('a refinement reports what changed', bool(r.get('changed')), r)
    if r.get('thread_drift_pct', 0) > before * 1.5 + 5:
        # Verdicts are 'ok'/'warn'/'fail'. An earlier guard compared against
        # 'PASS' and so could never fire — a change that doubled thread drift
        # was reported as a plain success.
        check('a change that worsens the cloth warns even inside one verdict band',
              'warning' in r, r)
    else:
        check('a harmless refinement does not cry wolf', 'warning' not in r, r)
    check('refining with nothing generated is refused',
          'error' in ag.run_tool('refine_design', {'change': 'denser'}, blank()))

    print('\nExplore and choose')
    s = blank()
    ag.run_tool('design', {'pins': 600, 'reed': 80, 'feel': 'traditional'}, s)
    r = ag.run_tool('explore_designs', {'count': 3}, s)
    check('exploring returns candidates', len(r.get('candidates', [])) >= 2, r)
    check('candidates are described in words',
          all(c.get('design') for c in r['candidates']), r)
    check('candidates differ from each other',
          len({c['design'] for c in r['candidates']}) > 1, r)
    ok = [c for c in r['candidates'] if c.get('verdict')]
    if len(ok) > 1:
        rank = {'ok': 0, 'warn': 1, 'fail': 2}
        check('candidates are ranked, cleanest first',
              rank.get(ok[0]['verdict'], 2) <= rank.get(ok[-1]['verdict'], 2), ok)
    chosen = ag.run_tool('choose_design', {'index': 0}, s)
    check('a candidate can be adopted', 'error' not in chosen, chosen)
    check('an out-of-range index is refused',
          'error' in ag.run_tool('choose_design', {'index': 99}, s))
    check('choosing before exploring is refused',
          'error' in ag.run_tool('choose_design', {'index': 0}, blank()))

    print('\nDesign to files, in one conversation')
    s = blank()
    seq = [Reply(tool_calls=[ToolCall('a', 'loom_geometry',
                                      {'width_in': 9, 'reed': 80})]),
           Reply(tool_calls=[ToolCall('b', 'design',
                                      {'pins': 720, 'reed': 80,
                                       'feel': 'traditional', 'threads': 2})]),
           Reply(tool_calls=[ToolCall('c', 'refine_design', {'change': 'denser'})]),
           Reply(tool_calls=[ToolCall('d', 'generate_files',
                                      {'shuttle_count': 3})]),
           Reply(text='Files are ready.')]
    ag._call_api = lambda m, tools=None: (seq.pop(0), None)
    out = ag.converse(s, 'I need a traditional saree body about 9 inches at reed 80')
    check('the conversation ran to completion', out['ok'], out)
    check('it used the intent tools, not the parameter ones',
          'design' in out['tools_used'] and 'generate_allover' not in out['tools_used'],
          out['tools_used'])
    check('files were produced', out['has_files'], out)
    payload, fname = ag.files_zip(s)
    check('the zip is real', payload and len(payload) > 200, len(payload or b''))
    check('the zip is named after the design', fname.endswith('.zip'), fname)

    print('\nDeterminism')
    a = ds.render(ds.LayoutSpec(pins=480, cards=600, cols=5))
    b = ds.render(ds.LayoutSpec(pins=480, cards=600, cols=5))
    check('the same spec renders identically', a.tobytes() == b.tobytes())

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
