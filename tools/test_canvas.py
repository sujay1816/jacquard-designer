"""
Tests for canvas and region work.

Runs offline. What is checked is that the cloth can be reshaped without losing
shuttle separation, that operations which look similar and are not stay
distinct, and that every canvas change is undoable and re-measured.

Run:  python tools/test_canvas.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_engine as ag                                   # noqa: E402
import canvas_ops as co                                     # noqa: E402
import llm                                                  # noqa: E402
from llm import Reply, ToolCall                             # noqa: E402
from PIL import Image                                       # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def canvas(w=80, h=100):
    """A canvas with two thread classes, so shuttle separation is testable."""
    lm = np.zeros((h, w), np.uint8)
    lm[20:60, 15:50] = 1
    lm[25:35, 20:30] = 2
    return lm


class Fake(llm.LLMProvider):
    name = 'fake'
    supports_vision = True

    def complete(self, *a, **k):
        return Reply(text='ok')

    def is_available(self):
        return True


def session_with_design():
    llm.set_provider(Fake())
    s = ag.get_session(ag.new_session(None, 't'))
    ag.run_tool('auto_design', {'pins': 320, 'reed': 80, 'effort': 1}, s)
    ag.run_tool('convert', {'pins': 320}, s)
    return s


def main():
    lm = canvas()

    print('\nReading the canvas')
    st = co.stats(lm)
    check('size is reported in threads and cards',
          st['pins'] == 80 and st['cards'] == 100, st)
    check('the design box is found', st['design_box'] == [15, 20, 50, 60], st['design_box'])
    check('blank margins are measured',
          st['blank_margins']['left'] == 15 and st['blank_margins']['bottom'] == 40, st)
    check('every thread class is listed', st['classes'] == [0, 1, 2], st['classes'])
    check('an empty canvas says so', co.stats(np.zeros((10, 10), np.uint8))['design_box'] is None)

    print('\nCanvas size')
    e = co.extend(lm, left=20, right=10, top=5, bottom=5)
    check('extending grows the canvas', e.shape == (110, 110), e.shape)
    # New cloth is bare cloth. Mirroring the edge or repeating the last column
    # would put thread lifts on the loom that nobody asked for.
    check('new cloth is bare', (e[:, :20] == 0).all() and (e[:5, :] == 0).all())
    check('the design is untouched by extending',
          (e[5:105, 20:100] == lm).all())
    check('extending by nothing is refused',
          _raises(lambda: co.extend(lm)))
    check('extending past the loom is refused',
          _raises(lambda: co.extend(lm, left=3000)))

    check('cropping keeps only the region', co.crop(lm, 'centre').shape == (50, 40))
    t = co.trim(lm)
    check('trimming cuts blank cloth away', t.shape == (40, 35), t.shape)
    check('trimming keeps a margin when asked', co.trim(lm, margin=5).shape == (50, 45))
    check('trimming an empty canvas is refused',
          _raises(lambda: co.trim(np.zeros((20, 20), np.uint8))))

    print('\nResize and scale are not the same thing')
    r = co.resize_canvas(lm, pins=120, cards=140)
    check('resize changes the canvas', r.shape == (140, 120), r.shape)
    # The design must keep its thread count — re-mounting is not resampling.
    check('resize does NOT resample the design',
          int((r == 1).sum()) + int((r == 2).sum())
          == int((lm == 1).sum()) + int((lm == 2).sum()))
    check('resize honours the anchor',
          co.resize_canvas(lm, pins=120, anchor='left')[:, :80].sum() == lm.sum())

    sc = co.scale(lm, pins=160, cards=200)
    check('scale changes the thread count', sc.shape == (200, 160), sc.shape)
    check('scale really resamples', (sc != 0).sum() > (lm != 0).sum())
    # Interpolating a label map would invent classes lying between two
    # shuttles and belonging to neither.
    check('scale invents no thread classes',
          set(np.unique(sc)) <= set(np.unique(lm)), np.unique(sc))

    print('\nMoving')
    m = co.move(lm, dx=10, dy=-5)
    check('moving shifts the design', co.stats(m)['design_box'][0] == 25, co.stats(m))
    check('moving without wrap pushes design off the edge',
          co.move(lm, dx=70)[:, :].sum() < lm.sum())
    check('wrapping keeps everything', co.move(lm, dx=70, wrap=True).sum() == lm.sum())
    c = co.centre(lm)
    b = co.stats(c)['design_box']
    check('centring balances the margins',
          abs(b[0] - (80 - b[2])) <= 1 and abs(b[1] - (100 - b[3])) <= 1, b)
    check('centring an empty canvas is refused',
          _raises(lambda: co.centre(np.zeros((20, 20), np.uint8))))

    print('\nShuttle separation survives everything')
    for name, out in [('extend', co.extend(lm, left=10)),
                      ('move', co.move(lm, dx=5, dy=5)),
                      ('mirror', co.mirror_across(lm)),
                      ('resize', co.resize_canvas(lm, pins=120)),
                      ('paste', co.paste(lm, co.copy_region(lm, 'centre'), 5, 5))]:
        check(f'{name} keeps both thread classes',
              2 in np.unique(out) and 1 in np.unique(out), np.unique(out))

    print('\nRegions')
    cl = co.clear(lm, 'top')
    check('clearing erases to bare cloth', (cl[:25, :] == 0).all())
    check('and leaves the rest alone', (cl[25:, :] == lm[25:, :]).all())
    check('an unknown region name is refused',
          _raises(lambda: co.clear(lm, 'the fiddly bit')))
    check('a zero-area box is refused',
          _raises(lambda: co.clear(lm, box=[10, 10, 10, 20])))
    check('a box off the canvas is clipped, not crashed',
          co.clear(lm, box=[-50, -50, 40, 40]).shape == lm.shape)

    patch = co.copy_region(lm, 'centre')
    check('copy lifts a region out', patch.shape == (50, 40), patch.shape)

    print('\nblend and over are not the same thing')
    ground = np.ones((60, 60), np.uint8)
    stamp = np.zeros((20, 20), np.uint8)
    stamp[5:15, 5:15] = 2
    over = co.paste(ground, stamp, 10, 10, mode='over')
    blend = co.paste(ground, stamp, 10, 10, mode='blend')
    # Stamping a butta with 'over' punches a bare rectangle through a lattice.
    check('over replaces the whole rectangle', (over[10:15, 10:15] == 0).all())
    check('blend lets the ground show through the gaps',
          (blend[10:15, 10:15] == 1).all())
    check('blend still writes the design cells', (blend[15:25, 15:25] == 2).all())
    check('a patch entirely off the canvas is refused',
          _raises(lambda: co.paste(ground, stamp, 500, 500)))

    print('\nBuilding a field from one motif')
    tiled = co.tile_region(lm, cols=2, rows=2, region='centre')
    check('tiling covers more cloth than the source',
          (tiled != 0).mean() > (lm != 0).mean(), (tiled != 0).mean())
    check('a runaway repeat count is capped, not obeyed',
          co.tile_region(lm, cols=9999, rows=9999, region='centre').shape == lm.shape)

    mir = co.mirror_across(lm, 'vertical')
    check('mirroring makes the panel symmetric',
          (mir[:, :40] == np.fliplr(mir[:, 40:])).all())
    check('an unknown mirror axis is refused',
          _raises(lambda: co.mirror_across(lm, 'diagonal')))

    print('\nWeave fill')
    solid = np.ones((40, 40), np.uint8)
    textured = co.fill_region_weave(solid, 'satin', 8, region='all')
    check('a weave leaves gaps in a solid fill',
          0 < (textured != 0).mean() < 1, (textured != 0).mean())
    # design_only=False over a motif buries it, so the default is the safe one.
    whole = co.fill_region_weave(lm, 'twill', 8, region='all', design_only=False)
    check('design_only=True textures only existing thread',
          ((co.fill_region_weave(lm, 'satin', 8, region='all') != 0)
           & (lm == 0)).sum() == 0)
    check('design_only=False reaches beyond the design',
          (whole[lm == 0] != 0).any(), (whole != 0).mean())

    # generate_fill_pattern falls back to satin for any name it does not know,
    # so a weaver asking for twill silently got satin and could not tell until
    # the cloth was off the loom. Names are checked before it is called.
    check('an unknown weave is refused, not quietly turned into satin',
          _raises(lambda: co.fill_region_weave(solid, 'tartan', 8)))
    check('twill and satin really differ',
          abs((co.fill_region_weave(solid, 'twill', 8, region='all') != 0).mean()
              - (co.fill_region_weave(solid, 'satin', 8, region='all') != 0).mean()) > 0.1)
    check("a weaver's word for a weave is understood",
          (co.fill_region_weave(solid, 'twill', 8, region='all')
           == co.fill_region_weave(solid, 'twill22', 8, region='all')).all())
    # bmp_engine uses 0 = thread UP, 1 = DOWN. Reading 1 as lift inverts the
    # weave: the float lands where the binding should be.
    check('the lift convention is not inverted',
          (co.fill_region_weave(solid, 'plain', 8, region='all') != 0).mean() > 0.4)

    print('\nStamping a motif')
    blank = np.zeros((300, 300), np.uint8)
    st2 = co.stamp_motif(blank, 'paisley', 120, x=90, y=90)
    check('a motif lands on the canvas', (st2 != 0).any())
    check('it lands where it was told',
          co.stats(st2)['design_box'][0] >= 85, co.stats(st2)['design_box'])
    check('an unknown motif is refused',
          _raises(lambda: co.stamp_motif(blank, 'peacock', 120)))

    print('\nThrough the agent')
    s = session_with_design()
    info = ag.run_tool('canvas_info', {}, s)
    check('canvas_info reports the cloth', info.get('pins') == 320, info)
    check('it lists the region names it accepts',
          'pallu' in info.get('named_regions', []), info.get('named_regions'))

    before = ag.run_tool('canvas_info', {}, s)['pins']
    r = ag.run_tool('canvas', {'operation': 'extend', 'left': 40, 'right': 40}, s)
    check('extending works through the agent', r['canvas']['pins'] == before + 80, r)
    check('a size change is called out', 'note' in r, r)
    check('the change is re-measured', 'verdict' in r, r)

    # Generated files describe the old canvas the moment it changes.
    check('generated files are invalidated', s.get('files') is None)
    # A spec that no longer matches the canvas must not be silently re-rendered.
    check('a stale spec is dropped after a size change', s.get('spec') is None)

    undone = ag.run_tool('undo_edit', {}, s)
    check('canvas work is undoable', 'error' not in undone, undone)
    check('undo restores the old width',
          ag.run_tool('canvas_info', {}, s)['pins'] == before)

    print('\nThe agent refuses clearly rather than crashing')
    check('an unknown canvas operation is named',
          'No such canvas operation' in ag.run_tool('canvas', {'operation': 'squish'}, s)['error'])
    check('an unknown region operation is named',
          'No such region operation' in ag.run_tool('region', {'operation': 'smudge'}, s)['error'])
    check('pasting with an empty clipboard is refused',
          'clipboard' in ag.run_tool('region', {'operation': 'paste'}, s)['error'])
    check('canvas work with no design is refused',
          'error' in ag.run_tool('canvas', {'operation': 'trim'},
                                 ag.get_session(ag.new_session(None, 'e'))))
    check('a stamp too small to draw is refused',
          'error' in ag.run_tool('region', {'operation': 'stamp', 'motif': 'lotus',
                                            'width_threads': 4}, s))

    print('\nCopy, paste and tile through the agent')
    s = session_with_design()
    check('copy reports the size lifted',
          'threads' in ag.run_tool('region', {'operation': 'copy',
                                              'region': 'centre'}, s).get('copied', ''))
    check('paste then works',
          'error' not in ag.run_tool('region', {'operation': 'paste',
                                                'x': 10, 'y': 10}, s))
    check('weave fill works',
          'error' not in ag.run_tool('region', {'operation': 'weave_fill',
                                                'pattern': 'twill', 'n': 8,
                                                'region': 'body'}, s))
    check('stamping works',
          'error' not in ag.run_tool('region', {'operation': 'stamp',
                                                'motif': 'lotus',
                                                'width_threads': 100,
                                                'x': 50, 'y': 50}, s))

    print('\nFiles survive the whole workflow')
    s = session_with_design()
    check('nothing generated yet is said plainly',
          'No files have been generated' in ag.run_tool('files', {}, s)['reason'],
          ag.run_tool('files', {}, s))

    ag.run_tool('generate_files', {'shuttle_count': 3}, s)
    f = ag.run_tool('files', {}, s)
    check('files are reported ready', f['ready'], f)
    check('they are listed with sizes', all(x['bytes'] > 0 for x in f['files']), f)
    # A BMP with a wrong header is rejected at the loom, the most expensive
    # place to find out.
    check('they verify as real 1-bit BMPs', f['verified'] is True, f)
    check('the zip is downloadable', len(ag.files_zip(s)[0] or b'') > 200)

    ag.run_tool('checkpoint', {'action': 'save', 'name': 'plain'}, s)
    ag.run_tool('canvas', {'operation': 'extend', 'left': 60, 'right': 60}, s)
    check('an edit clears the files and says why',
          'changed since' in ag.run_tool('files', {}, s)['reason'])

    # The bug: the conversion record kept the OLD pin count, and generate_files
    # reads it. Any canvas resize left the session permanently unable to
    # produce files, advising a re-run of Detect Colours — which cannot be done
    # from a conversation.
    r = ag.run_tool('generate_files', {'shuttle_count': 3}, s)
    check('files can still be made after a canvas resize', 'error' not in r, r)
    check('and they describe the NEW width',
          r['pins'] == 320 + 120, r.get('pins'))
    check('the zip regenerates', len(ag.files_zip(s)[0] or b'') > 200)

    print('\nThe reed follows the cloth')
    # Reporting at a hardcoded reed 60 told a weaver who designed at reed 80
    # that their 4-inch panel measured 5.3 inches.
    check('physical size is reported at the real reed',
          'reed 80' in r['physical_size_in'], r['physical_size_in'])

    print('\nCheckpoints')
    lst = ag.run_tool('checkpoint', {'action': 'list'}, s)
    check('saved versions are listed', len(lst['checkpoints']) == 1, lst)
    check('they are described, not just named',
          bool(lst['checkpoints'][0]['design']), lst)

    back = ag.run_tool('checkpoint', {'action': 'restore', 'name': 'plain'}, s)
    check('restoring goes back to the saved canvas',
          back['canvas']['pins'] == 320, back)
    # The checkpoint held a reference to the same conversion dict, so a later
    # canvas edit mutated its saved pin count underneath it — restoring a good
    # version produced a record describing a canvas that no longer existed.
    r2 = ag.run_tool('generate_files', {'shuttle_count': 3}, s)
    check('files can be made from a restored checkpoint', 'error' not in r2, r2)
    check('and they describe the restored width', r2['pins'] == 320, r2.get('pins'))

    check('restoring an unknown name is refused',
          'error' in ag.run_tool('checkpoint', {'action': 'restore', 'name': 'nope'}, s))
    check('a checkpoint needs a name',
          'error' in ag.run_tool('checkpoint', {'action': 'save'}, s))
    check('an unknown action is refused',
          'error' in ag.run_tool('checkpoint', {'action': 'yeet', 'name': 'x'}, s))
    check('deleting works',
          'deleted' in ag.run_tool('checkpoint', {'action': 'delete', 'name': 'plain'}, s))

    # Checkpoints hold full label maps; an agent saving on every step would
    # grow the session without bound.
    for i in range(12):
        ag.run_tool('checkpoint', {'action': 'save', 'name': f'v{i}'}, s)
    check('the number kept is capped',
          len(ag.run_tool('checkpoint', {'action': 'list'}, s)['checkpoints']) <= 8)

    print('\nFidelity refuses to compare unlike things')
    s = session_with_design()
    ag.run_tool('generate_files', {'shuttle_count': 3}, s)
    r = ag.run_tool('canvas', {'operation': 'extend', 'left': 200, 'right': 200}, s)
    # Extending with BLANK cloth scored -54.8% drift and a warn, because a
    # 720-thread map was being compared against a 320-thread source. Every
    # canvas resize told the weaver their design had been damaged.
    check('extending blank cloth is not reported as damage',
          'warning' not in r, r.get('warning'))
    check('and the change is still measured', r.get('verdict') in ('ok', 'warn', 'fail'), r)
    check('the weaver is told the comparison has moved',
          'against the design as it stands' in (r.get('note') or ''), r.get('note'))
    later = ag.run_tool('edit_design', {'operation': 'thicken', 'amount': 1}, s)
    check('later edits are scored against the new reference',
          later.get('thread_drift_pct') is not None, later)

    print('\nWeave names mean the same thing in every tool')
    s = session_with_design()
    # 'twill' worked in region/weave_fill and was refused by set_weave, for the
    # same word in the same conversation. The engine key is 'twill22'.
    check('set_weave accepts a weaving word',
          ag.run_tool('set_weave', {'shuttle': 'zari', 'pattern': 'twill'},
                      s).get('pattern') == 'twill22')
    check('weave_fill accepts it too',
          'error' not in ag.run_tool('region', {'operation': 'weave_fill',
                                                'pattern': 'twill',
                                                'region': 'all'}, s))
    check("'plain' maps to plain_weave",
          ag.run_tool('set_weave', {'shuttle': 'zari', 'pattern': 'plain'},
                      s).get('pattern') == 'plain_weave')
    check('a made-up weave is still refused',
          'error' in ag.run_tool('set_weave', {'shuttle': 'zari',
                                               'pattern': 'tartan'}, s))

    print('\nRefusals name what went wrong')
    s = session_with_design()
    # Both colours exist and both are assigned, but neither is the ground —
    # so this reaches the background rule rather than the earlier check that a
    # colour was detected at all.
    err = ag.run_tool('set_shuttles',
                      {'assignments': {'0': 'meena1', '1': 'zari'}}, s).get('error', '')
    # "Exactly one colour must be the background" did not say which colours
    # existed or which were left out, so the model had to guess the correction.
    check('a shuttle error lists the colours in the design',
          'Colours in this design' in err, err)
    check('and says what was actually set', 'you set 0' in err, err)
    check('and points at the fix', 'largest area' in err, err)
    # The omissions line only appears when something WAS omitted — a message
    # that always lists them would be noise on the common case.
    partial = ag.run_tool('set_shuttles', {'assignments': {'1': 'zari'}},
                          session_with_design()).get('error', '')
    check('an omitted colour is named when there is one',
          'Nothing was said about' in partial or 'background' in partial, partial)

    print('\nA long session does not grow without bound')
    import pickle
    s = session_with_design()
    ag.run_tool('explore_designs', {'count': 4}, s)
    for _ in range(10):
        ag.run_tool('edit_design', {'operation': 'thicken', 'amount': 1}, s)
    for i in range(8):
        ag.run_tool('checkpoint', {'action': 'save', 'name': f'v{i}'}, s)
    ag.run_tool('generate_files', {'shuttle_count': 3}, s)
    size_mb = len(pickle.dumps({k: v for k, v in s.items() if k != 'history'})) / 1e6
    # Holding a full image and conversion per explored variant, plus raw label
    # maps in undo and every checkpoint, reached 35 MB for one session — 1.4 GB
    # across the 40 a server keeps.
    check(f'a heavily worked session stays small ({size_mb:.1f} MB)',
          size_mb < 12, size_mb)

    print('\nAnd everything still works after all that')
    check('undo works', 'error' not in ag.run_tool('undo_edit', {}, s))
    r = ag.run_tool('checkpoint', {'action': 'restore', 'name': 'v0'}, s)
    check('a checkpoint restores', r.get('restored') == 'v0', r)
    check('files generate from it',
          'error' not in ag.run_tool('generate_files', {'shuttle_count': 3}, s))
    check('and download', len(ag.files_zip(s)[0] or b'') > 200)
    # Variants no longer keep their rendered image; the winner is rebuilt from
    # its spec, which is safe only because rendering is deterministic.
    chosen = ag.run_tool('choose_design', {'index': 0}, s)
    check('a variant is rebuilt exactly when chosen', 'error' not in chosen, chosen)

    print('\nThe model is only offered tools that can do something')
    blank_s = ag.get_session(ag.new_session(None, 'scratch'))
    offered = [t['name'] for t in ag.tools_for(blank_s)]
    # 24 schemas cost 4,433 tokens on EVERY round — 73% of the fixed prefix —
    # and most were inapplicable, which is also what degrades tool selection.
    check('a fresh design session gets a small set', len(offered) <= 10, offered)
    check('conversion tools are hidden with nothing uploaded',
          'convert' not in offered and 'inspect_design' not in offered, offered)
    check('canvas tools are hidden with nothing designed',
          'canvas' not in offered and 'region' not in offered, offered)
    check('but designing is offered', 'auto_design' in offered, offered)

    ag.run_tool('auto_design', {'pins': 320, 'reed': 80, 'effort': 1}, blank_s)
    after = [t['name'] for t in ag.tools_for(blank_s)]
    check('canvas work appears once there is a design', 'canvas' in after, after)
    # A generated design IS an image, so convert and inspect legitimately apply
    # to it — re-converting at a different pin count is a real request. The
    # gating therefore saves most in the opening rounds, before anything
    # exists, and less once a design is on the table. That is the honest
    # shape of the win: 7 tools instead of 22 at the start, not throughout.
    check('the full set returns once there is something to work on',
          len(after) >= 20, len(after))

    up = ag.get_session(ag.new_session(Image.new('RGB', (600, 400), 'white'), 'x.png'))
    check('an upload gets the conversion tools',
          'inspect_design' in [t['name'] for t in ag.tools_for(up)])

    # Gating must fall open: a tool added later without touching the lists
    # should still reach the model.
    check('an ungated tool is always offered',
          'plan_work' in offered and 'loom_geometry' in offered, offered)
    check('every offered tool can actually be dispatched',
          all(t['name'] in ag._DISPATCH for t in ag.TOOLS),
          [t['name'] for t in ag.TOOLS if t['name'] not in ag._DISPATCH])

    print('\nStale views are not paid for twice')
    llm.set_provider(Fake())
    s = session_with_design()
    seq = [Reply(tool_calls=[ToolCall('a', 'look_at_design', {})]),
           Reply(tool_calls=[ToolCall('b', 'look_at_design', {})]),
           Reply(text='Looks right.')]
    ag._call_api = lambda m, tools=None: (seq.pop(0), None)
    ag.converse(s, 'how does it look?')
    with_images = [r for m in s['history'] if m.get('role') == 'tool_results'
                   for r in m['results'] if r.get('images')]
    # A thumbnail is ~390 tokens and stays in the transcript for every later
    # round. Only the newest one describes the current cloth; older ones show
    # cloth that has since changed, so they are misleading as well as costly.
    check('only the most recent view is kept', len(with_images) == 1, len(with_images))
    dropped = [r for m in s['history'] if m.get('role') == 'tool_results'
               for r in m['results'] if 'earlier view dropped' in r.get('content', '')]
    check('and the model is told one was dropped', len(dropped) >= 1, len(dropped))

    llm.reset()
    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == '__main__':
    sys.exit(main())
