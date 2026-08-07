"""
Tests for the assistant page's backend.

Runs offline behind a scripted model. What is checked is that the page can show
the design, that progress arrives while the work happens rather than after it,
and that a local-model setup is not told to go and get an API key.

Run:  python tools/test_agent_page.py
"""
import json
import os
import sys

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


class Scripted(llm.LLMProvider):
    name = 'scripted'

    def __init__(self, *replies, vision=True, available=True):
        self.seq = list(replies)
        self.supports_vision = vision
        self._available = available

    def complete(self, system, messages, tools, max_tokens=1400):
        return self.seq.pop(0) if self.seq else Reply(text='Done.')

    def is_available(self):
        return self._available

    def describe(self):
        return 'scripted:test'


def client():
    return flask_app.app.test_client()


def events(resp):
    return [json.loads(chunk[6:])
            for chunk in resp.get_data(as_text=True).split('\n\n')
            if chunk.startswith('data: ')]


def main():
    c = client()

    print('\nAvailability is asked of the provider, not of Anthropic')
    llm.set_provider(Scripted(available=True))
    d = c.get('/api/assistant-status').get_json()
    # The bug this guards: this route called assistant_engine.is_available(),
    # which only looks for an ANTHROPIC_API_KEY. A mill running a local Llama
    # was shown "the assistant needs an API key" over a working backend.
    check('a configured backend reports available', d['available'], d)
    check('the backend is named for the UI', d.get('backend') == 'scripted:test', d)
    check('vision capability is reported', d.get('vision') is True, d)

    llm.set_provider(Scripted(available=False))
    check('no backend reports unavailable',
          not c.get('/api/assistant-status').get_json()['available'])

    llm.set_provider(Scripted(vision=False))
    check('a text-only backend says so',
          c.get('/api/assistant-status').get_json()['vision'] is False)

    print('\nDesigning from scratch needs no fake upload')
    llm.set_provider(Scripted())
    tok = c.post('/api/agent/blank').get_json().get('token')
    check('a blank session opens', bool(tok))
    s = ag.get_session(tok)
    # Previously the page posted an 8x8 white PNG so it could reuse the upload
    # path, which showed the weaver a white square as their "design".
    check('it carries no source image', s['image'] is None, s['image'])
    check('tools needing an image refuse clearly, not with a traceback',
          'no design in this session' in ag.run_tool('inspect_design', {}, s)['error'],
          ag.run_tool('inspect_design', {}, s))

    print('\nThe preview')
    check('there is nothing to preview before designing',
          c.get(f'/api/agent/preview?token={tok}').status_code == 400)
    check('an expired token is refused',
          c.get('/api/agent/preview?token=nope').status_code == 400)

    ag.run_tool('auto_design', {'pins': 320, 'reed': 80, 'effort': 1}, s)
    r = c.get(f'/api/agent/preview?token={tok}')
    check('the design can be seen once it exists', r.status_code == 200, r.status_code)
    check('it comes back as an image', r.headers['Content-Type'] == 'image/png',
          r.headers.get('Content-Type'))
    check('it is a real png', r.data[:4] == b'\x89PNG', r.data[:4])
    # A cached preview would show the previous design and make a refinement
    # look like it did nothing at all.
    check('it is never cached', r.headers.get('Cache-Control') == 'no-store',
          r.headers.get('Cache-Control'))

    print('\nProgress arrives while the work happens')
    llm.set_provider(Scripted(
        Reply(tool_calls=[ToolCall('p', 'plan_work',
                                   {'steps': ['Work out the width',
                                              'Design the panel']})]),
        Reply(tool_calls=[ToolCall('a', 'auto_design',
                                   {'pins': 320, 'reed': 80, 'effort': 1})]),
        Reply(tool_calls=[ToolCall('d', 'plan_work', {'done': [0, 1]})]),
        Reply(text='Three paisley across, four inches at reed 80.')))
    tok2 = c.post('/api/agent/blank').get_json()['token']
    evs = events(c.get(f'/api/agent/stream?token={tok2}&message=800 pins at reed 80'))

    kinds = [e['type'] for e in evs]
    check('tool starts are streamed', 'tool' in kinds, kinds)
    check('tool completions are streamed', 'tool_done' in kinds, kinds)
    check('the turn ends with done', kinds[-1] == 'done', kinds)
    check('progress precedes the answer',
          kinds.index('tool') < kinds.index('done'), kinds)

    labels = [e['label'] for e in evs if e['type'] == 'tool']
    # 'auto_design · plan_work' is a stack trace, not something a weaver reads.
    check('progress is in words, not function names',
          all('_' not in l for l in labels), labels)
    check('the design step is narrated',
          any('design' in l.lower() for l in labels), labels)

    done = evs[-1]
    check('the final reply is carried', bool(done['reply']), done)
    check('the plan comes back ticked off',
          done['plan'] and all(p['done'] for p in done['plan']), done.get('plan'))
    check('the page is told a design now exists', done['has_design'], done)

    print('\nStreaming failures are reported, not hung on')
    evs = events(c.get('/api/agent/stream?token=expired&message=hello'))
    check('an expired session ends the stream cleanly',
          evs and evs[0]['type'] == 'error', evs)
    tok3 = c.post('/api/agent/blank').get_json()['token']
    evs = events(c.get(f'/api/agent/stream?token={tok3}&message='))
    check('an empty message is refused', evs and evs[0]['type'] == 'error', evs)

    llm.set_provider(Scripted(available=False))
    tok4 = c.post('/api/agent/blank').get_json()
    check('a blank session is refused with no backend', not tok4.get('success'), tok4)

    print('\nThe page itself')
    llm.set_provider(Scripted())
    html = c.get('/agent').get_data(as_text=True)
    check('the page still renders', 'Jacquard Designer' in html)
    check('it has a design view', 'id="preview"' in html)
    check('it has a plan panel', 'id="planSteps"' in html)
    check('it streams rather than polls', 'EventSource' in html)
    check('a turn can be stopped', 'stopTurn' in html)
    check('the offline notice covers local models',
          'llm_provider' in html and 'ollama' in html)
    check('the session survives a reload', 'sessionStorage' in html)
    check('the preview is cache-busted',
          'Date.now()' in html)

    llm.reset()
    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
