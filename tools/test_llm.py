"""
Tests for the swappable LLM layer.

Runs entirely offline — no key, no model server. What is being checked is the
translation, not the model: that a conversation stored once in neutral form
replays correctly into either wire format, and that the argument repair which
makes weaker models usable does not quietly change what was asked for.

Run:  python tools/test_llm.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import Reply, ToolCall, assistant_msg, tool_results_msg, user_msg  # noqa: E402
from llm.anthropic import AnthropicProvider                                # noqa: E402
from llm.openai_compat import (OpenAICompatProvider, _parse_args,          # noqa: E402
                               _recover_inline_call)
from llm.registry import ALIASES, resolve_name                             # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


# A conversation with every message kind in it: plain user turn, assistant turn
# that calls a tool, the results coming back, and a final user turn.
HISTORY = [
    user_msg('convert this at 480 pins'),
    assistant_msg(Reply(text='Let me look.',
                        tool_calls=[ToolCall(id='c1', name='inspect_design',
                                             args={}),
                                    ToolCall(id='c2', name='convert',
                                             args={'pins': 480})])),
    tool_results_msg([{'id': 'c1', 'name': 'inspect_design', 'content': '{"ok":1}'},
                      {'id': 'c2', 'name': 'convert', 'content': '{"verdict":"PASS"}'}]),
    user_msg('good, make the files'),
]


def main():
    print('\nAnthropic wire format')
    wire = AnthropicProvider(api_key='x')._to_wire(HISTORY)
    check('assistant tool calls become tool_use blocks',
          any(b.get('type') == 'tool_use' for b in wire[1]['content']), wire[1])
    check('tool results go back in a USER turn',
          wire[2]['role'] == 'user'
          and wire[2]['content'][0]['type'] == 'tool_result', wire[2])
    check('tool_use_id is preserved',
          wire[2]['content'][0]['tool_use_id'] == 'c1', wire[2])
    check('assistant text survives alongside the calls',
          wire[1]['content'][0]['text'] == 'Let me look.', wire[1])

    ap = AnthropicProvider(api_key='x')
    reply = ap._from_wire({
        'content': [{'type': 'text', 'text': 'Converting.'},
                    {'type': 'tool_use', 'id': 'z1', 'name': 'convert',
                     'input': {'pins': 600}}],
        'usage': {'input_tokens': 12, 'output_tokens': 3,
                  'cache_read_input_tokens': 3000}})
    check('reply text parsed', reply.text == 'Converting.', reply.text)
    check('reply tool call parsed',
          reply.tool_calls[0].name == 'convert'
          and reply.tool_calls[0].args['pins'] == 600, reply.tool_calls)
    check('cache reads are reported', reply.usage['cache_read'] == 3000, reply.usage)

    print('\nPrompt caching')
    tools = [{'name': 'a', 'description': 'x', 'input_schema': {'type': 'object'}},
             {'name': 'b', 'description': 'y', 'input_schema': {'type': 'object'}}]
    cached = AnthropicProvider(api_key='x', cache_prompt=True)._tools_wire(tools)
    check('cache breakpoint lands on the last tool',
          'cache_control' in cached[-1] and 'cache_control' not in cached[0], cached)
    plain = AnthropicProvider(api_key='x', cache_prompt=False)._tools_wire(tools)
    check('caching can be turned off',
          all('cache_control' not in t for t in plain), plain)

    print('\nOpenAI-compatible wire format')
    ow = OpenAICompatProvider()._to_wire('SYSTEM', HISTORY)
    check('system prompt becomes a message',
          ow[0]['role'] == 'system' and ow[0]['content'] == 'SYSTEM', ow[0])
    check('tool calls become a tool_calls array',
          len(ow[2]['tool_calls']) == 2, ow[2])
    check('arguments are serialised to a JSON string',
          isinstance(ow[2]['tool_calls'][0]['function']['arguments'], str), ow[2])
    tool_msgs = [m for m in ow if m['role'] == 'tool']
    check('each result is its own tool message', len(tool_msgs) == 2, tool_msgs)
    check('tool_call_id links the result to the call',
          {m['tool_call_id'] for m in tool_msgs} == {'c1', 'c2'}, tool_msgs)

    ot = OpenAICompatProvider._tools_wire(tools)
    check('tools are wrapped as functions',
          ot[0]['type'] == 'function' and 'parameters' in ot[0]['function'], ot[0])

    print('\nBoth formats replay the same conversation')
    a_calls = [b['name'] for m in wire if m['role'] == 'assistant'
               for b in m['content'] if b.get('type') == 'tool_use']
    o_calls = [c['function']['name'] for m in ow if m.get('tool_calls')
               for c in m['tool_calls']]
    check('same tool calls in the same order', a_calls == o_calls, (a_calls, o_calls))

    print('\nReading an OpenAI-shaped reply')
    r = OpenAICompatProvider._from_wire({
        'choices': [{'message': {'content': None, 'tool_calls': [
            {'id': 'k1', 'type': 'function',
             'function': {'name': 'convert', 'arguments': '{"pins": 480}'}}]}}],
        'usage': {'prompt_tokens': 9, 'completion_tokens': 2}})
    check('tool call decoded from the arguments string',
          r.tool_calls[0].args == {'pins': 480}, r.tool_calls)
    check('usage decoded', r.usage['input'] == 9, r.usage)

    missing_id = OpenAICompatProvider._from_wire({'choices': [{'message': {
        'tool_calls': [{'function': {'name': 'convert', 'arguments': '{}'}}]}}]})
    check('a missing call id is synthesised, not left null',
          bool(missing_id.tool_calls[0].id), missing_id.tool_calls)

    print('\nArgument repair (what weaker models actually emit)')
    cases = [
        ('clean json',        '{"pins": 480}',                  {'pins': 480}),
        ('markdown fenced',   '```json\n{"pins": 480}\n```',    {'pins': 480}),
        ('trailing comma',    '{"pins": 480,}',                 {'pins': 480}),
        ('single quotes',     "{'pins': 480}",                  {'pins': 480}),
        ('already a dict',    {'pins': 480},                    {'pins': 480}),
        ('prose around it',   'Sure! {"pins": 480} ok?',        {'pins': 480}),
        ('empty string',      '',                               {}),
        ('total garbage',     'no idea',                        {}),
        ('nested object kept', '{"satin": {"n": 8}}',           {'satin': {'n': 8}}),
    ]
    for label, raw, want in cases:
        got = _parse_args(raw)
        check(f'{label} -> {want}', got == want, got)

    check('a json array is not accepted as arguments',
          _parse_args('[1,2,3]') == {}, _parse_args('[1,2,3]'))

    print('\nInline tool call recovery')
    rec = _recover_inline_call('{"name": "convert", "arguments": {"pins": 300}}')
    check('a call written as text is recovered',
          rec and rec.name == 'convert' and rec.args == {'pins': 300}, rec)
    rec2 = _recover_inline_call('{"tool": "convert", "arguments": "{\\"pins\\": 300}"}')
    check('doubly-encoded arguments are recovered',
          rec2 and rec2.args == {'pins': 300}, rec2)
    check('ordinary prose is not mistaken for a call',
          _recover_inline_call('That will need about 480 pins.') is None)
    check('a json object with no tool name is not a call',
          _recover_inline_call('{"pins": 480}') is None)

    print('\nProvider selection')
    check('default is anthropic', resolve_name({}, {}) == 'anthropic')
    check('env wins over config',
          resolve_name({'llm_provider': 'anthropic'},
                       {'JQ_LLM_PROVIDER': 'ollama'}) == 'openai_compat')
    for alias, target in ALIASES.items():
        check(f'alias {alias!r} -> {target}',
              resolve_name({'llm_provider': alias}, {}) == target)
    check('an unknown name does not crash selection',
          isinstance(resolve_name({'llm_provider': 'typo'}, {}), str))

    print('\nAvailability')
    check('anthropic without a key is unavailable',
          not AnthropicProvider(api_key='').is_available())
    check('anthropic with a key is available',
          AnthropicProvider(api_key='sk-test').is_available())
    check('a local server needs no key',
          OpenAICompatProvider(base_url='http://localhost:11434/v1').is_available())

    print('\nNo vendor names leak above the llm package')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, 'agent_engine.py'), encoding='utf-8').read()
    # Quoted literals only. A bare word may legitimately appear in a comment
    # explaining why the neutral format exists; what must not appear is the
    # agent actually constructing or reading a vendor field.
    leaks = [lit for lit in ('"x-api-key"', "'x-api-key'",
                             '"anthropic-version"', "'anthropic-version'",
                             '"tool_use"', "'tool_use'",
                             '"tool_result"', "'tool_result'",
                             # NOTE: input_schema is deliberately absent here.
                             # It is the NEUTRAL tool-spec field name defined
                             # in llm/base.py (chosen to match the shape
                             # agent_engine.TOOLS already used), so both
                             # adapters translate it outward. It is not a leak.
                             'chat/completions', 'api.anthropic.com')
             if lit in src]
    check('agent_engine constructs no wire-format fields', not leaks, leaks)

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
