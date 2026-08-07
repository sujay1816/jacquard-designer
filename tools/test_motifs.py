"""
Tests for the motif library and the interlock layout.

Added when a weaver asked whether the assistant could produce work as complex as
a traditional block-print reference — layered medallions, scattered flowers, and
sprigs flowing between them. It could not: seven outline motifs on a plain
lattice. These cover what was built to close that gap.

Run:  python tools/test_motifs.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import design_studio as ds                                  # noqa: E402
import motif_library as ml                                  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def ink(img):
    return (np.asarray(img.convert('L')) < 128)


NEW = ('diamond_medallion', 'daisy', 'leaf_sprig')


def main():
    print('\nThe new motifs exist and draw')
    for name in NEW:
        check(f'{name} is registered', name in ml.MOTIFS)
        check(f'{name} declares a thread requirement',
              ml.MIN_THREADS_PER_MOTIF.get(name, 0) > 0)
        img = ml.render(ml.build_svg(name, 300, colours=3), 300)
        cov = ink(img).mean()
        # Too little and it is invisible on cloth; too much and it is a blob.
        check(f'{name} carries a sensible amount of ink ({cov*100:.0f}%)',
              0.02 < cov < 0.55, cov)

    print('\nThey survive being drawn small')
    for name in NEW:
        need = ml.MIN_THREADS_PER_MOTIF[name]
        img = ml.render(ml.build_svg(name, need, colours=3), need)
        cov = ink(img).mean()
        # At its own stated minimum a motif must still read — that number is a
        # promise the layout engine relies on when choosing columns.
        check(f'{name} still reads at {need} threads ({cov*100:.0f}%)',
              0.02 < cov < 0.75, cov)

    print('\nThe medallion centre is not lost')
    # The innermost band once inherited ink, so the rosette was drawn dark on
    # dark and vanished completely.
    img = ml.render(ml.build_svg('diamond_medallion', 400, bands=3, colours=3), 400)
    a = np.asarray(img.convert('L'))
    h, w = a.shape
    centre = a[int(h*0.42):int(h*0.58), int(w*0.42):int(w*0.58)]
    # In a three-thread design the ornament is drawn in the SECOND thread —
    # mid-grey, not black — because that is how a zari-and-meena butta is
    # built: the metallic draws the shape, the colour fills it. What matters is
    # that it contrasts with the cloth behind it, not that it is dark.
    check('the ornament contrasts with the ground behind it',
          centre.min() < 200 and centre.max() > 220, (centre.min(), centre.max()))
    two = ml.render(ml.build_svg('diamond_medallion', 400, bands=3, colours=2), 400)
    m2 = np.asarray(two.convert('L'))[int(h*0.42):int(h*0.58), int(w*0.42):int(w*0.58)]
    check('with one ink thread the ornament is that thread', m2.min() < 100, m2.min())
    for bands in (1, 2, 3, 4):
        c = ml.render(ml.build_svg('diamond_medallion', 400, bands=bands, colours=3), 400)
        m = np.asarray(c.convert('L'))[int(h*0.45):int(h*0.55), int(w*0.45):int(w*0.55)]
        # The innermost band was once inheriting ink, so the ornament was drawn
        # dark on dark and disappeared entirely at even band counts.
        check(f'{bands} bands still shows an ornament on cloth',
              m.min() < 200 and m.max() > 220, (m.min(), m.max()))

    print('\nThe sprig spreads along its stem')
    img = ml.render(ml.build_svg('leaf_sprig', 500, leaves=7), 500)
    m = ink(img)
    # Leaves once bunched into the first third because the stem curve was
    # evaluated with the branches the wrong way round.
    thirds = [m[:, :167].sum(), m[:, 167:334].sum(), m[:, 334:].sum()]
    check(f'no third of the width holds most of the leaves {thirds}',
          max(thirds) < sum(thirds) * 0.6, thirds)
    check('leaf count changes the drawing',
          ink(ml.render(ml.build_svg('leaf_sprig', 500, leaves=3), 500)).sum()
          < ink(ml.render(ml.build_svg('leaf_sprig', 500, leaves=10), 500)).sum())

    print('\nThe daisy is a flower, not a disc')
    img = ml.render(ml.build_svg('daisy', 400, petals=8), 400)
    m = ink(img)
    check('it leaves ground between the petals', 0.05 < m.mean() < 0.5, m.mean())
    check('petal count changes the drawing',
          ink(ml.render(ml.build_svg('daisy', 400, petals=5), 400)).sum()
          != ink(ml.render(ml.build_svg('daisy', 400, petals=12), 400)).sum())

    print('\nThe interlock layout')
    svg = ml.allover(900, layout='interlock', motif='diamond_medallion',
                     filler='leaf_sprig', cols=5, rows=5, colours=3, spacing=0.10)
    both = ink(ml.render(svg, 900))
    alone = ink(ml.render(ml.allover(900, layout='interlock',
                                     motif='diamond_medallion', filler=None,
                                     cols=5, rows=5, colours=3, spacing=0.10), 900))
    check('the filler adds ink between the buttas',
          both.sum() > alone.sum() * 1.15, (alone.sum(), both.sum()))
    check('but does not swamp them', both.mean() < 0.6, both.mean())
    check('an unknown filler is ignored rather than fatal',
          ink(ml.render(ml.allover(600, layout='interlock', motif='daisy',
                                   filler='not_a_motif', cols=4, rows=4), 600)).sum() > 0)

    print('\nNothing hangs past the bottom edge')
    # The field is composed INTO a panel with a pallu beneath it. An overrun
    # does not crop, it overlaps — medallions printed straight through the
    # cross border.
    spec = ds.LayoutSpec(**ds.plan(pins=600, reed=80, feel='ornate',
                                   threads=3, pallu=True)['spec'])
    img = ds.render(spec)
    m = ink(img)
    h = m.shape[0]
    body_h = int(h * (1 - spec.cross_frac))
    band = m[body_h + 4:, :]
    # The pallu band is a chevron rule: sparse. If body motifs overran into it
    # the coverage there would be close to the body's.
    body_cov = m[:body_h, :].mean()
    check(f'the pallu band stays sparse ({band.mean()*100:.0f}% vs body '
          f'{body_cov*100:.0f}%)', band.mean() < body_cov * 0.75,
          (band.mean(), body_cov))

    print('\nA busy reference maps onto a feel')
    for feel in ('flowing', 'ornate', 'floral', 'brocade'):
        p = ds.plan(pins=900, reed=80, feel=feel, threads=3)
        check(f"'{feel}' chooses the interlock layout",
              p['spec']['body_layout'] == 'interlock', p['spec']['body_layout'])
        check(f"'{feel}' picks a butta, not a ground",
              p['chosen']['kind'] == 'butta', p['chosen'])

    # Interlock carries a second motif in every gap, so the same column count
    # reads twice as busy. Traditional cloth of this kind has four or five
    # medallions across a body, not ten.
    ornate = ds.plan(pins=900, reed=80, feel='ornate', threads=3)
    rich = ds.plan(pins=900, reed=80, feel='rich', threads=3)
    check(f"an interlock field is not packed as tight as a plain one "
          f"({ornate['chosen']['cols']} vs {rich['chosen']['cols']})",
          ornate['chosen']['cols'] < rich['chosen']['cols'],
          (ornate['chosen']['cols'], rich['chosen']['cols']))

    print('\nIt still converts to a weavable design')
    from auto_convert import auto_convert
    spec = ds.LayoutSpec(**ds.plan(pins=800, reed=80, feel='ornate',
                                   threads=2, pallu=True)['spec'])
    img = ds.render(spec)
    conv = auto_convert(img, pins=spec.pins, n_colors=3)
    check('a medallion field converts', conv.get('best') is not None, conv.get('summary'))
    if conv.get('best'):
        check('with a usable verdict',
              str(conv['verdict']).lower() in ('ok', 'warn'), conv['verdict'])

    print('\nDescriptions name what was actually built')
    spec = ds.LayoutSpec(**ds.plan(pins=900, reed=80, feel='ornate', threads=3)['spec'])
    text = ds.describe(spec)
    check('the layout is named', 'interlock' in text, text)
    check('and the filler between the buttas', 'leaf_sprig' in text, text)

    print('\nDeterminism')
    a = ml.render(ml.build_svg('diamond_medallion', 300, colours=3), 300)
    b = ml.render(ml.build_svg('diamond_medallion', 300, colours=3), 300)
    check('the same motif draws identically', a.tobytes() == b.tobytes())

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
