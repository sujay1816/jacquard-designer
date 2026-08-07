"""
Tests for the conversational agent.

Runs entirely offline. The API call is replaced with a scripted responder, so
the tool loop, the validation, and the session handling are all exercised
without a key — which is where every safety property lives.

Run:  python tools/test_agent.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_engine as ag                                   # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def _working_of(session):
    return session.get('working')


def design(w=600, h=400):
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)
    for i in range(5):
        d.ellipse((50 + i * 105, 80, 140 + i * 105, 300), outline='black', width=4)
        d.ellipse((75 + i * 105, 130, 115 + i * 105, 250), outline='black', width=3)
    d.line((0, 30, w, 30), fill='black', width=4)
    return img


def script(*turns):
    """Replace the API with a fixed sequence of responses."""
    seq = list(turns)

    def fake(messages):
        return (seq.pop(0) if seq else {'content': [{'type': 'text', 'text': 'done'}]}), None
    ag._call_api = fake


def tool_call(name, args, cid='t1'):
    return {'content': [{'type': 'tool_use', 'id': cid, 'name': name, 'input': args}]}


def say(text):
    return {'content': [{'type': 'text', 'text': text}]}


def main():
    img = design()

    print('\nSessions')
    t = ag.new_session(img, 'my_design.jpg')
    s = ag.get_session(t)
    check('session created', s is not None)
    check('filename kept without extension', s['filename'] == 'my_design', s['filename'])
    check('unknown token returns nothing', ag.get_session('nope') is None)

    print('\nTools run and validate')
    r = ag.run_tool('inspect_design', {}, s)
    check('inspect reports dimensions', r.get('width') == 600, r)
    check('inspect identifies line art', 'line art' in r.get('type', ''), r)
    check('inspect suggests a pin count', 'supported_pin_count' in r)

    check('unknown tool is refused',
          'error' in ag.run_tool('no_such_tool', {}, s))
    check('bad pins refused',
          'error' in ag.run_tool('convert', {'pins': 'many'}, s))
    check('out-of-range pins refused',
          'error' in ag.run_tool('convert', {'pins': 99999}, s))
    check('generate before convert refused',
          'error' in ag.run_tool('generate_files', {}, s))

    print('\nConversion and file generation')
    r = ag.run_tool('convert', {'pins': 300}, s)
    check('convert returns a verdict', r.get('verdict') in ('ok', 'warn', 'fail'), r)
    check('convert reports gap counts',
          'design_gaps_source' in r and 'design_gaps_result' in r)
    check('convert produces no files', s.get('files') is None)

    r = ag.run_tool('generate_files', {'shuttle_count': 2}, s)
    check('files generated', r.get('ready') is True, r)
    check('every file is clean 1-bit', all(f['clean_1bit'] for f in r['files']), r)
    check('shuttle budget respected', len(r['files']) <= 3, r['files'])

    r = ag.run_tool('generate_files', {'shuttle_count': 99}, s)
    check('excess shuttles clamped, not honoured', len(r['files']) <= 4, r['files'])

    payload, name = ag.files_zip(s)
    check('zip produced', payload and name.endswith('.zip'), name)

    print('\nConversation loop')
    t2 = ag.new_session(design(), 'x.png')
    s2 = ag.get_session(t2)
    script(tool_call('inspect_design', {}),
           say('Line art, 600px wide. How many pins does the job need?'))
    out = ag.converse(s2, 'here is my design')
    check('tool is executed then answered', out['ok'] and 'pins' in out['reply'], out)
    check('tool use is reported', out['tools_used'] == ['inspect_design'], out['tools_used'])
    check('no files yet', out['has_files'] is False)

    script(tool_call('convert', {'pins': 300}, 't2'),
           say('300 pins works but loses some interior detail. Shall I generate?'))
    out = ag.converse(s2, '300 pins')
    check('second turn runs convert', out['tools_used'] == ['convert'], out['tools_used'])

    script(tool_call('generate_files', {'shuttle_count': 2}, 't3'),
           say('Done — two files ready.'))
    out = ag.converse(s2, 'yes please')
    check('files flagged after generation', out['has_files'] is True, out)

    print('\nEditing')
    te = ag.new_session(design(), 'e.png')
    se = ag.get_session(te)
    check('edit before convert refused',
          'error' in ag.run_tool('edit_design', {'operation': 'thicken'}, se))
    ag.run_tool('convert', {'pins': 300}, se)

    r = ag.run_tool('edit_design', {'operation': 'thicken', 'amount': 2}, se)
    check('thicken applies', r.get('applied') == 'thicken', r)
    check('fidelity is rescored after an edit', 'thread_drift_pct' in r, r)
    check('a damaging edit is flagged, not hidden', 'warning' in r, r)

    check('unknown operation refused',
          'error' in ag.run_tool('edit_design', {'operation': 'sparkle'}, se))

    before = ag.run_tool('describe_result', {}, se)['thread_drift_pct']
    ag.run_tool('edit_design', {'operation': 'thin', 'amount': 1}, se)
    check('undo restores the previous state',
          ag.run_tool('undo_edit', {}, se)['thread_drift_pct'] == before)
    while ag.get_session(te)['undo']:
        ag.run_tool('undo_edit', {}, se)
    check('undo past the start is refused', 'error' in ag.run_tool('undo_edit', {}, se))

    shape = np.asarray(_working_of(se)).shape
    ag.run_tool('edit_design', {'operation': 'rotate_90'}, se)
    check('rotate transposes the design',
          np.asarray(_working_of(se)).shape == (shape[1], shape[0]))
    ag.run_tool('undo_edit', {}, se)

    print('\nEdits reach the generated files')
    ag.run_tool('generate_files', {'shuttle_count': 2}, se)
    plain = dict(ag.get_session(te)['files'])
    ag.run_tool('edit_design', {'operation': 'thicken', 'amount': 3}, se)
    check('editing invalidates stale files', ag.get_session(te)['files'] is None)
    ag.run_tool('generate_files', {'shuttle_count': 2}, se)
    edited = ag.get_session(te)['files']
    check('regenerated files differ after an edit',
          any(plain[k] != edited[k] for k in plain if k in edited), 'files identical')

    print('\nShuttles and weave')
    r = ag.run_tool('set_shuttles', {'shuttle_count': 1,
                                     'assignments': {'0': 'background', '1': 'zari', '2': 'meena1'}}, se)
    check('over-budget shuttle assignment refused', 'error' in r, r)
    r = ag.run_tool('set_shuttles', {'shuttle_count': 2,
                                     'assignments': {'0': 'background', '1': 'zari'}}, se)
    check('valid assignment accepted', r.get('assignments'), r)
    r = ag.run_tool('set_shuttles', {'shuttle_count': 2,
                                     'assignments': {'0': 'zari', '1': 'meena1'}}, se)
    check('assignment with no background refused', 'error' in r, r)

    check('unknown shuttle refused',
          'error' in ag.run_tool('set_weave', {'shuttle': 'silk'}, se))
    check('unknown weave refused',
          'error' in ag.run_tool('set_weave', {'shuttle': 'zari', 'pattern': 'velvet'}, se))
    r = ag.run_tool('set_weave', {'shuttle': 'zari', 'pattern': 'twill22', 'n': 6}, se)
    check('valid weave accepted', r.get('pattern') == 'twill22', r)
    r = ag.run_tool('set_weave', {'shuttle': 'zari', 'n': 99}, se)
    check('satin count clamped to 16', r.get('n') == 16, r)
    check('long float earns a warning', 'warning' in r, r)

    print('\nStatus reporting')
    r = ag.run_tool('describe_result', {}, se)
    for f in ('pins', 'verdict', 'shuttles', 'weave', 'edits_applied', 'physical_size_in'):
        check(f'describe_result reports {f}', f in r, r)

    print('\nGenerating designs')
    tg = ag.new_session(Image.new('RGB', (10, 10), 'white'), 'g.png')
    sg = ag.get_session(tg)

    r = ag.run_tool('list_motifs', {}, sg)
    check('motifs can be listed', len(r.get('motifs', [])) >= 5, r)
    check('listing is honest about tradition', 'not traditional' in r.get('note', ''), r)

    r = ag.run_tool('generate_design', {'motif': 'paisley', 'pins': 480}, sg)
    check('design is generated', r.get('created') == 'paisley', r)
    check('generated design converts cleanly', r.get('verdict') == 'ok', r)

    check('unknown motif refused',
          'error' in ag.run_tool('generate_design', {'motif': 'chola', 'pins': 480}, sg))
    check('out-of-range pins refused',
          'error' in ag.run_tool('generate_design', {'motif': 'lotus', 'pins': 99999}, sg))
    check('missing pins refused',
          'error' in ag.run_tool('generate_design', {'motif': 'lotus'}, sg))

    # Stroke weight is chosen for the loom, which is the whole point.
    import motif_library as ml
    from loom_utils import source_resolution_check
    for pins in (240, 480, 960):
        img = ml.render(ml.build_svg('lotus', pins), pins)
        tps = source_resolution_check(img, pins).get('threads_per_stroke') or 0
        check(f'lotus at {pins} pins is weavable by construction', tps >= 2.0, tps)

    r = ag.run_tool('edit_design', {'operation': 'thicken'}, sg)
    check('generated designs can then be edited', r.get('applied') == 'thicken', r)
    r = ag.run_tool('generate_files', {'shuttle_count': 2}, sg)
    check('generated designs produce clean files',
          r.get('ready') and all(f['clean_1bit'] for f in r['files']), r)

    print('\nAll-over brocade fields')
    ta = ag.new_session(Image.new('RGB', (10, 10), 'white'), 'a.png')
    sa = ag.get_session(ta)

    import motif_library as ml
    from loom_utils import source_resolution_check

    for layout in ml.ALLOVER_LAYOUTS:
        r = ag.run_tool('generate_allover',
                        {'pins': 480, 'layout': layout, 'motif': 'paisley',
                         'cols': 5, 'rows': 5}, sa)
        check(f'{layout} field builds', r.get('verdict') in ('ok', 'warn'), r)

    check('unknown layout refused',
          'error' in ag.run_tool('generate_allover', {'pins': 480, 'layout': 'spiral'}, sa))
    check('unknown motif refused',
          'error' in ag.run_tool('generate_allover',
                                 {'pins': 480, 'layout': 'jaal', 'motif': 'chola'}, sa))
    check('out-of-range pins refused',
          'error' in ag.run_tool('generate_allover', {'pins': 99999, 'layout': 'jaal'}, sa))

    # The point of rebuilding motifs at tile size: linework survives the repeat.
    for cols in (3, 6, 10):
        img = ml.render(ml.allover(480, layout='half_drop', motif='paisley',
                                   cols=cols, rows=4), 480)
        tps = source_resolution_check(img, 480).get('threads_per_stroke') or 0
        check(f'{cols} motifs across stays weavable', tps >= 2.0, tps)

    r = ag.run_tool('generate_allover',
                    {'pins': 480, 'layout': 'banded', 'motif': 'lotus',
                     'band_motif': 'chevron_border', 'cols': 4, 'rows': 4}, sa)
    check('banded field accepts a band motif', r.get('verdict') in ('ok', 'warn'), r)
    r = ag.run_tool('edit_design', {'operation': 'thicken'}, sa)
    check('all-over fields can be edited', r.get('applied') == 'thicken', r)
    r = ag.run_tool('generate_files', {'shuttle_count': 2}, sa)
    check('all-over fields produce clean files',
          r.get('ready') and all(f['clean_1bit'] for f in r['files']), r)

    print('\nDesigning to a pin count')
    td = ag.new_session(Image.new('RGB', (10, 10), 'white'), 'd2.png')
    sd = ag.get_session(td)

    r = ag.run_tool('design_options', {'pins': 480}, sd)
    check('reports what fits', r.get('motifs'), r)
    check('gives usable guidance', r.get('notes'), r)
    paisley = next(m for m in r['motifs'] if m['motif'] == 'paisley')
    check('max across is derived from the pin count',
          paisley['max_across'] == 10, paisley)
    check('comfortable count is lower than the maximum',
          paisley['comfortable_across'] < paisley['max_across'], paisley)

    wide = ag.run_tool('design_options', {'pins': 960}, sd)
    narrow = ag.run_tool('design_options', {'pins': 240}, sd)
    wp = next(m for m in wide['motifs'] if m['motif'] == 'paisley')['max_across']
    np_ = next(m for m in narrow['motifs'] if m['motif'] == 'paisley')['max_across']
    check('a wider loom fits more', wp > np_, (wp, np_))

    tiny = ag.run_tool('design_options', {'pins': 150}, sd)
    check('narrow looms are steered to geometric grounds',
          any('geometric' in n for n in tiny['notes']), tiny['notes'])

    check('out-of-range pins refused',
          'error' in ag.run_tool('design_options', {'pins': 5}, sd))

    # The recommendation must actually hold up when built.
    cols = paisley['comfortable_across']
    r = ag.run_tool('generate_allover',
                    {'pins': 480, 'layout': 'half_drop', 'motif': 'paisley',
                     'cols': cols, 'rows': 5}, sd)
    check('the recommended count converts cleanly', r.get('verdict') == 'ok', r)

    print('\nMulti-thread designs')
    tc = ag.new_session(Image.new('RGB', (10, 10), 'white'), 'c.png')
    sc = ag.get_session(tc)

    import motif_library as ml
    from vision_engine import detect_colors_smart

    for motif in ('paisley', 'lotus', 'vine_border', 'diamond_jaal'):
        img = ml.render(ml.build_svg(motif, 480, colours=3), 480)
        _, _, lm, _ = detect_colors_smart(img, 3, 480, img.size[1])
        check(f'{motif} separates into three thread classes',
              len(np.unique(np.asarray(lm))) == 3, np.unique(np.asarray(lm)))

    r = ag.run_tool('generate_allover',
                    {'pins': 480, 'layout': 'half_drop', 'motif': 'paisley',
                     'cols': 5, 'rows': 4, 'colours': 3}, sc)
    check('a two-thread field builds', r.get('threads') == 2, r)

    # shuttle_count includes the rani ground, so two design threads need 3.
    g = ag.run_tool('generate_files', {'shuttle_count': 3}, sc)
    names = {f['file'].rsplit('_', 1)[-1] for f in g['files']}
    check('a second thread produces a meena file', 'meena1.bmp' in names, names)

    g2 = ag.run_tool('generate_files', {'shuttle_count': 2}, sc)
    check('a loom with too few shuttles says so, not silently drops a thread',
          g2.get('thread_warning'), g2)
    check('every thread file is clean 1-bit',
          all(f['clean_1bit'] for f in g['files']), g['files'])

    r = ag.run_tool('generate_allover',
                    {'pins': 480, 'layout': 'jaal', 'motif': 'lotus',
                     'cols': 5, 'rows': 4, 'colours': 2}, sc)
    check('single-thread fields still work', r.get('threads') == 1, r)

    print('\nFailure handling')
    t3 = ag.new_session(design(), 'y.png')
    s3 = ag.get_session(t3)
    script(*[tool_call('inspect_design', {}, f'r{i}') for i in range(ag.MAX_TOOL_ROUNDS + 2)])
    out = ag.converse(s3, 'loop forever')
    check('runaway tool loop is stopped',
          out['ok'] and len(out['tools_used']) <= ag.MAX_TOOL_ROUNDS,
          len(out['tools_used']))

    ag._call_api = lambda m: (None, 'Assistant unreachable.')
    t4 = ag.new_session(design(), 'z.png')
    out = ag.converse(ag.get_session(t4), 'hello')
    check('API failure returns a message, not a crash',
          out['ok'] is False and 'unreachable' in out['reply'].lower(), out)
    check('failed turn is not left in history',
          not ag.get_session(t4)['history'], ag.get_session(t4)['history'])

    print('\nDeterminism')
    a = ag.new_session(img, 'd.png')
    b = ag.new_session(img, 'd.png')
    for tok in (a, b):
        ss = ag.get_session(tok)
        ag.run_tool('convert', {'pins': 300}, ss)
        ag.run_tool('generate_files', {'shuttle_count': 2}, ss)
    fa, fb = ag.get_session(a)['files'], ag.get_session(b)['files']
    check('same image and pins give identical files',
          all(fa[k] == fb[k] for k in fa) and set(fa) == set(fb))

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
