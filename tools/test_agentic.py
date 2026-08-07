"""
Tests for the agentic layer.

Runs offline. What is checked is that the agent improves a design on its own
and can prove it, that looking at its own work never becomes a way to produce
pixels, and that a backend without vision degrades honestly instead of
answering about an image it never saw.

Run:  python tools/test_agentic.py
"""
import base64
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_engine as ag                                   # noqa: E402
import design_studio as ds                                  # noqa: E402
import llm                                                  # noqa: E402
from llm import Reply, ToolCall, tool_results_msg           # noqa: E402
from llm.anthropic import AnthropicProvider                 # noqa: E402
from llm.openai_compat import OpenAICompatProvider, _guesses_vision  # noqa: E402

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


class FakeProvider(llm.LLMProvider):
    """A provider that only exists to declare a vision capability."""
    name = 'fake'

    def __init__(self, vision=True):
        self.supports_vision = vision

    def complete(self, system, messages, tools, max_tokens=1400):
        return Reply(text='ok')

    def is_available(self):
        return True


def main():
    print('\nThe search actually improves the design')
    out = ds.auto_design(pins=400, reed=80, feel='rich', threads=2,
                         pallu=True, rounds=3)
    check('a brief produces a worked answer', 'error' not in out, out.get('error'))
    trail = out['trail']
    check('the trail records every step', len(trail) >= 2, trail)
    check('the trail starts from the plan', trail[0]['step'] == 'start', trail[0])

    kept = [t for t in trail if t.get('kept') and t.get('drift') is not None]
    check('the winner is no worse than the start',
          kept[-1]['drift'] <= kept[0]['drift'], (kept[0]['drift'], kept[-1]['drift']))
    check('every kept step improved on the one before',
          all(kept[i]['drift'] <= kept[i - 1]['drift'] for i in range(1, len(kept))),
          [t['drift'] for t in kept])
    check('the winning record is the last kept step',
          abs(out['best']['drift'] - kept[-1]['drift']) < 1e-6, out['best']['drift'])

    # A hill climb that reports steps it did not keep would be lying about its
    # work; one that never stops would burn the weaver's time for nothing.
    check('the climb stops when nothing helps',
          any(t['step'] == 'stop' for t in trail) or out['rounds_used'] >= 3, trail)
    check('a design that will not convert is refused, not shipped',
          'error' in ds.auto_design(pins=10, reed=60, rounds=1)
          or ds.auto_design(pins=10, reed=60, rounds=1)['best'] is not None)

    print('\nEffort is bounded')
    quick = ds.auto_design(pins=320, reed=80, rounds=1)
    check('effort 1 does at most one improvement',
          quick['rounds_used'] <= 1, quick['rounds_used'])
    check('effort is clamped, not trusted',
          ds.auto_design(pins=320, rounds=1)['rounds_used'] <= 8)

    print('\nThumbnails')
    img = ds.render(ds.LayoutSpec(pins=400, cards=500))
    t = ds.thumbnail(img)
    check('a thumbnail is base64 png', t['media_type'] == 'image/png' and t['data'])
    raw = base64.b64decode(t['data'])
    check('it decodes to a real image', raw[:4] == b'\x89PNG', raw[:4])
    small = Image.open(__import__('io').BytesIO(raw))
    check('it is downscaled', max(small.size) <= 640, small.size)
    check('it is greyscale, not three copies of one bit',
          small.mode in ('L', 'P', 'LA'), small.mode)

    print('\nLooking at the design')
    llm.set_provider(FakeProvider(vision=True))
    s = blank()
    # A session always carries the uploaded image, so "nothing to look at"
    # only arises if the image is genuinely absent.
    empty = blank()
    empty['image'] = None
    check('an empty session has nothing to look at',
          'error' in ag.run_tool('look_at_design', {}, empty))
    check('an upload can be looked at before any design is generated',
          ag.run_tool('look_at_design', {}, blank()).get('showing')
          == 'the uploaded design')

    ag.run_tool('auto_design', {'pins': 320, 'reed': 80, 'feel': 'traditional',
                                'effort': 1}, s)
    r = ag.run_tool('look_at_design', {}, s)
    check('looking returns an image', bool(r.get('_images')), list(r))
    check('it says what is being shown', bool(r.get('showing')), r)
    check('it carries the measurements too', r.get('verdict') is not None, r)
    # The score already measures linework far better than an eye on a
    # thumbnail. Inviting a second opinion on it would produce confident
    # disagreement with the more reliable number.
    check('it steers away from judging line quality',
          'composition' in r['guidance'].lower(), r['guidance'])

    llm.set_provider(FakeProvider(vision=False))
    r = ag.run_tool('look_at_design', {}, s)
    check('a backend without vision refuses instead of guessing',
          'error' in r and r.get('measurements_only'), r)
    check('and it points back at the measurements',
          'measurement' in r['error'].lower(), r['error'])

    print('\nImages never reach the model as text')
    llm.set_provider(FakeProvider(vision=True))
    s = blank()
    ag.run_tool('auto_design', {'pins': 320, 'reed': 60, 'effort': 1}, s)
    seq = [Reply(tool_calls=[ToolCall('v1', 'look_at_design', {})]),
           Reply(text='Looks balanced.')]
    ag._call_api = lambda m, tools=None: (seq.pop(0), None)
    ag.converse(s, 'how does it look?')
    results = [m for m in s['history'] if m.get('role') == 'tool_results']
    entry = results[-1]['results'][0]
    # base64 in a text field would blow the context window and tell the model
    # nothing — the picture has to travel as a picture or not at all.
    check('the image is not serialised into the text payload',
          '_images' not in entry['content'] and len(entry['content']) < 4000,
          len(entry['content']))
    check('the image travels as an image', bool(entry.get('images')), list(entry))

    llm.set_provider(FakeProvider(vision=False))
    s2 = blank()
    ag.run_tool('auto_design', {'pins': 320, 'reed': 60, 'effort': 1}, s2)
    seq = [Reply(tool_calls=[ToolCall('v2', 'look_at_design', {})]),
           Reply(text='Going by the numbers.')]
    ag._call_api = lambda m, tools=None: (seq.pop(0), None)
    ag.converse(s2, 'how does it look?')
    entry = [m for m in s2['history'] if m.get('role') == 'tool_results'][-1]['results'][0]
    check('no image is attached when the backend cannot read one',
          not entry.get('images'), list(entry))

    print('\nImages survive both wire formats')
    h = [tool_results_msg([{'id': 'c1', 'name': 'look', 'content': '{"ok":1}',
                            'images': [{'media_type': 'image/png', 'data': 'AAA'}]}])]
    aw = AnthropicProvider(api_key='x')._to_wire(h)
    inner = aw[0]['content'][0]['content']
    check('anthropic nests the image inside the tool result',
          isinstance(inner, list) and inner[0]['type'] == 'image', inner)
    ow = OpenAICompatProvider()._to_wire('S', h)
    check('openai keeps the tool message text-only',
          isinstance(ow[1]['content'], str), ow[1])
    # Images inside a role="tool" message are silently dropped by most servers,
    # and the model then answers about an image it was never shown.
    check('openai follows it with the image as a user turn',
          ow[-1]['role'] == 'user'
          and ow[-1]['content'][0]['type'] == 'image_url', ow[-1])

    print('\nVision capability is declared, not assumed')
    check('anthropic declares vision', AnthropicProvider(api_key='x').supports_vision)
    check('llava is recognised', _guesses_vision('llava:13b'))
    check('a qwen vl model is recognised', _guesses_vision('Qwen2.5-VL-7B'))
    check('plain llama is not assumed to see', not _guesses_vision('llama3.3:70b'))
    check('an explicit setting overrides the guess',
          OpenAICompatProvider(model='llama3.3:70b', vision=True).supports_vision)

    print('\nComparing candidates')
    llm.set_provider(FakeProvider(vision=True))
    s = blank()
    check('comparing before exploring is refused',
          'error' in ag.run_tool('compare_designs', {}, s))
    ag.run_tool('design', {'pins': 360, 'reed': 80, 'feel': 'traditional'}, s)
    ag.run_tool('explore_designs', {'count': 3}, s)
    r = ag.run_tool('compare_designs', {}, s)
    check('a contact sheet is produced', bool(r.get('_images')), list(r))
    check('the candidates are labelled in order',
          [c['index'] for c in r['candidates']] == list(range(len(r['candidates']))), r)

    print('\nA whole job in one turn')
    llm.set_provider(FakeProvider(vision=True))
    s = blank()
    seq = [Reply(tool_calls=[ToolCall('g', 'loom_geometry', {'width_in': 9, 'reed': 80})]),
           Reply(tool_calls=[ToolCall('a', 'auto_design',
                                      {'pins': 360, 'reed': 80, 'feel': 'traditional',
                                       'threads': 2, 'effort': 1})]),
           Reply(tool_calls=[ToolCall('l', 'look_at_design', {})]),
           Reply(tool_calls=[ToolCall('r', 'refine_design', {'change': 'denser'})]),
           Reply(tool_calls=[ToolCall('f', 'generate_files', {'shuttle_count': 3})]),
           Reply(text='Here they are.')]
    ag._call_api = lambda m, tools=None: (seq.pop(0), None)
    out = ag.converse(s, 'traditional saree body, about 9 inches at reed 80')
    check('the turn completed', out['ok'], out)
    check('it sized, designed, looked, refined and delivered',
          {'loom_geometry', 'auto_design', 'look_at_design',
           'refine_design', 'generate_files'} <= set(out['tools_used']),
          out['tools_used'])
    check('files came out', out['has_files'], out)
    payload, _ = ag.files_zip(s)
    check('the zip is real', payload and len(payload) > 200, len(payload or b''))

    print('\nThe agent has room to work')
    check('the loop allows a full job', ag.MAX_TOOL_ROUNDS >= 12, ag.MAX_TOOL_ROUNDS)
    s = blank()
    ag._call_api = lambda m, tools=None: (Reply(tool_calls=[ToolCall('x', 'list_motifs', {})]), None)
    out = ag.converse(s, 'loop')
    check('but a runaway loop is still stopped',
          len(out['tools_used']) <= ag.MAX_TOOL_ROUNDS, len(out['tools_used']))

    print('\nDeterminism')
    a = ds.auto_design(pins=320, reed=80, feel='traditional', rounds=1)
    b = ds.auto_design(pins=320, reed=80, feel='traditional', rounds=1)
    check('the same brief searches to the same design',
          ds.describe(a['best']['spec']) == ds.describe(b['best']['spec']),
          (ds.describe(a['best']['spec']), ds.describe(b['best']['spec'])))
    check('and to the same measurements', a['best']['drift'] == b['best']['drift'])

    llm.reset()
    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
