"""
Hardening tests: the assistant should never crash, whatever it is handed.

Everything here is an adversarial case rather than a feature. The model can and
does emit malformed tool calls, users paste 100,000 characters, sessions expire
mid-turn, providers die between rounds. None of that may raise — a traceback
reaching the weaver is worse than any wrong answer, because there is nothing
they can do with it.

Run:  python tools/test_hardening.py
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_engine as ag                                   # noqa: E402
import app as flask_app                                     # noqa: E402
import llm                                                  # noqa: E402
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


class Seq(llm.LLMProvider):
    name = 'seq'

    def __init__(self, *replies, vision=True):
        self.q = list(replies)
        self.supports_vision = vision

    def complete(self, *a, **k):
        v = self.q.pop(0) if self.q else Reply(text='done')
        if isinstance(v, Exception):
            raise v
        return v

    def is_available(self):
        return True


def blank():
    return ag.get_session(ag.new_session(None, 't'))


def T(name, args):
    return Reply(tool_calls=[ToolCall('x', name, args)])


# Argument shapes a confused model actually produces.
HOSTILE = [
    {}, None, {'operation': ''}, {'operation': 'nope'}, {'operation': 123},
    {'pins': 'abc'}, {'pins': -5}, {'pins': 10 ** 9},
    {'operation': 'crop', 'box': [1]},
    {'operation': 'crop', 'box': ['a', 'b', 'c', 'd']},
    {'operation': 'crop', 'box': [0, 0, 0, 0]},
    {'operation': 'crop', 'box': [999, 999, 1, 1]},
    {'operation': 'paste', 'x': 'q', 'y': None},
    {'operation': 'stamp', 'motif': None, 'width_threads': None},
    {'operation': 'tile', 'cols': -3, 'rows': 10 ** 7},
    {'operation': 'weave_fill', 'pattern': None, 'n': -1},
    {'operation': 'extend', 'left': -9, 'right': 'x'},
    {'operation': 'move', 'dx': None, 'wrap': 'yes'},
    {'action': 'restore', 'name': None}, {'action': None, 'name': 'v'},
    {'index': -1}, {'index': 10 ** 6}, {'change': 'nonsense'},
    {'steps': 'not a list'}, {'done': 'x'},
    {'assignments': {'99': 'zari'}}, {'assignments': {}},
    {'shuttle_count': 0}, {'shuttle_count': 'three'},
    {'shuttle': 'bogus', 'pattern': 'satin'}, {'amount': -99},
    {'width_in': 0}, {'width_in': -4, 'reed': 0}, {'reed': -1},
]

# Excluded only because each renders for seconds; they are covered elsewhere.
SLOW = {'auto_design', 'explore_designs', 'design', 'generate_design', 'convert'}

LEAKS = ('has no attribute', 'not subscriptable', 'KeyError', 'IndexError',
         'unhashable', 'Traceback', 'positional argument', 'NoneType',
         'invalid literal', 'cannot unpack')


def main():
    print('\nEvery tool, every malformed argument')
    llm.set_provider(Seq(vision=True))
    for label, s in (('blank', blank()), ('with a design', _designed())):
        problems, calls = [], 0
        for tool in sorted(ag._DISPATCH):
            if tool in SLOW:
                continue
            for args in HOSTILE:
                calls += 1
                try:
                    r = ag.run_tool(tool, args, s)
                    if not isinstance(r, dict):
                        problems.append((tool, 'returned %s' % type(r).__name__))
                    elif 'error' in r and any(x in str(r['error']) for x in LEAKS):
                        # A refusal is fine; a Python internal reaching the
                        # weaver is not — they can do nothing with it, and the
                        # model cannot correct from it either.
                        problems.append((tool, str(r['error'])[:70]))
                except Exception as e:
                    problems.append((tool, 'RAISED %s: %s' % (type(e).__name__, e)))
        check(f'{calls} hostile calls on a {label} session leak nothing',
              not problems, problems[:4])

    print('\nMalformed model output')
    cases = [
        ('a tool that does not exist', [T('not_a_tool', {}), Reply(text='ok')]),
        ('a tool that was not offered', [T('canvas', {'operation': 'trim'}), Reply(text='ok')]),
        ('arguments as a string', [T('loom_geometry', '{"pins":400}'), Reply(text='ok')]),
        ('arguments as a list', [T('loom_geometry', [1, 2, 3]), Reply(text='ok')]),
        ('an empty tool name', [T('', {}), Reply(text='ok')]),
        ('an empty reply', [Reply(text='')]),
        ('five tools in one round',
         [Reply(tool_calls=[ToolCall(str(i), 'list_motifs', {}) for i in range(5)]),
          Reply(text='ok')]),
        ('two calls sharing one id',
         [Reply(tool_calls=[ToolCall('x', 'list_motifs', {}),
                            ToolCall('x', 'list_motifs', {})]), Reply(text='ok')]),
    ]
    for label, seq in cases:
        llm.set_provider(Seq(*seq))
        try:
            out = ag.converse(blank(), 'go')
            check(label, isinstance(out.get('reply'), str) and out['reply'], out)
        except Exception as e:
            check(label, False, f'RAISED {type(e).__name__}: {e}')

    print('\nBackend failures')
    for label, err in (('an unexpected exception', RuntimeError('boom')),
                       ('a provider error', llm.ProviderError('rate limited'))):
        llm.set_provider(Seq(err))
        s = blank()
        out = ag.converse(s, 'go')
        check(f'{label} returns a message, not a crash',
              out['ok'] is False and out['reply'], out)
        # A turn that failed part-way would otherwise leave assistant turns
        # holding tool calls nothing answered, which the next request rejects.
        check(f'{label} leaves no half-finished turn', not s['history'], s['history'])

    llm.set_provider(Seq(T('list_motifs', {}), RuntimeError('gone')))
    s = blank()
    out = ag.converse(s, 'go')
    check('a backend that dies mid-loop rolls the whole turn back',
          out['ok'] is False and not s['history'], (out['ok'], len(s['history'])))

    print('\nHostile user input')
    for label, msg in (('empty', ''), ('100,000 characters', 'x' * 100000),
                       ('null bytes', 'a\x00b'), ('heavy unicode', 'हाथकरघा 🧵' * 200),
                       ('something shaped like a system message', '{"role":"system"}')):
        llm.set_provider(Seq(Reply(text='ok')))
        try:
            ag.converse(blank(), msg)
            check(label, True)
        except Exception as e:
            check(label, False, f'RAISED {type(e).__name__}: {e}')

    print('\nConcurrency and expiry')
    llm.set_provider(Seq(*[Reply(text='a')] * 30))
    s = blank()
    errs = []

    def run():
        try:
            ag.converse(s, 'hello')
        except Exception as e:      # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=run) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check('six parallel turns on one session do not crash', not errs, errs[:1])
    check('and leave the history well formed',
          all(isinstance(m, dict) and 'role' in m for m in s['history']))

    tok = ag.new_session(None, 't')
    orphan = ag.get_session(tok)
    ag._sessions.pop(tok, None)
    check('an expired token returns nothing', ag.get_session(tok) is None)
    llm.set_provider(Seq(Reply(text='ok')))
    check('a session dropped mid-conversation still completes its turn',
          ag.converse(orphan, 'still there?')['ok'])

    print('\nHistory stays bounded and well formed')
    p = Seq(*([T('list_motifs', {}), Reply(text='a' * 600)] * 60))
    llm.set_provider(p)
    s = blank()
    for i in range(40):
        ag.converse(s, f'turn {i} ' + 'z' * 300)
    check('stored history is capped', len(s['history']) <= ag.MAX_HISTORY * 4 + 4,
          len(s['history']))
    sent = p.last_messages if hasattr(p, 'last_messages') else None
    win = ag._trim(s['history'])
    check('only a window is ever sent', len(win) <= ag.MAX_HISTORY, len(win))
    # A tool_results whose assistant turn was trimmed away is rejected by BOTH
    # wire formats, so the window must start at a clean boundary.
    check('the window never starts on an orphaned tool result',
          not win or win[0]['role'] != 'tool_results', win[0]['role'] if win else None)
    check('no tool result is orphaned anywhere in the window',
          all(win[i]['role'] != 'tool_results' or win[i - 1]['role'] == 'assistant'
              for i in range(1, len(win))))

    print('\nHTTP surface')
    c = flask_app.app.test_client()
    llm.set_provider(Seq(Reply(text='ok')))
    tok = c.post('/api/agent/blank').get_json()['token']
    for route in ('/api/agent/download', '/api/agent/preview', '/api/agent/state',
                  '/api/agent/file'):
        check(f'{route} refuses a bad token',
              c.get(f'{route}?token=nope&name=x').status_code == 400)
        check(f'{route} refuses a missing token',
              c.get(route).status_code == 400)
    check('an over-long message is refused',
          c.post('/api/agent/message',
                 json={'token': tok, 'message': 'y' * 100000}).status_code == 400)
    check('an empty message is refused',
          c.post('/api/agent/message', json={'token': tok, 'message': ''}).status_code == 400)
    check('a message with no body is refused',
          c.post('/api/agent/message', json={}).status_code == 400)
    # The status route once asked for an Anthropic key specifically, hiding a
    # working local backend behind a notice telling the user to buy one.
    body = json.dumps(c.get('/agent').get_data(as_text=True))
    check('no route tells a local-model user to get an API key',
          'needs an API key' not in body)

    print('\nStreaming failures')
    llm.set_provider(Seq(T('canvas', {'operation': 'trim'}), Reply(text='could not')))
    tok = c.post('/api/agent/blank').get_json()['token']
    evs = [json.loads(x[6:]) for x in
           c.get(f'/api/agent/stream?token={tok}&message=trim')
           .get_data(as_text=True).split('\n\n') if x.startswith('data: ')]
    check('a failing tool is marked failed, not hidden',
          any(e['type'] == 'tool_done' and e['ok'] is False for e in evs), evs)
    check('and the stream still ends cleanly', evs[-1]['type'] == 'done', evs[-1])

    llm.set_provider(Seq(T('list_motifs', {}), RuntimeError('gone')))
    tok = c.post('/api/agent/blank').get_json()['token']
    evs = [json.loads(x[6:]) for x in
           c.get(f'/api/agent/stream?token={tok}&message=hi')
           .get_data(as_text=True).split('\n\n') if x.startswith('data: ')]
    check('a backend dying mid-stream still terminates the stream',
          evs[-1]['type'] == 'done' and evs[-1]['ok'] is False, evs[-1])

    llm.reset()
    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


def _designed():
    llm.set_provider(Seq(vision=True))
    s = blank()
    ag.run_tool('auto_design', {'pins': 200, 'reed': 80, 'effort': 1}, s)
    ag.run_tool('checkpoint', {'action': 'save', 'name': 'v'}, s)
    return s


if __name__ == '__main__':
    sys.exit(main())
