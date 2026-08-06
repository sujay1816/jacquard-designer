"""
Tests for the conversational agent.

Runs entirely offline. The API call is replaced with a scripted responder, so
the tool loop, the validation, and the session handling are all exercised
without a key — which is where every safety property lives.

Run:  python tools/test_agent.py
"""
import os
import sys

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
