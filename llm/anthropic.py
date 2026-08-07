"""
Anthropic adapter.

Wire shape this file owns:
  * assistant turns carry `tool_use` blocks inside `content`
  * tool results go back as `tool_result` blocks inside a **user** turn
  * auth is the `x-api-key` header plus `anthropic-version`

Prompt caching is applied to the system prompt and the tool schemas. Those two
are a fixed prefix of every request and together run past 3,000 tokens, and the
agent loop re-sends them on every tool round — up to MAX_TOOL_ROUNDS times for
a single weaver message. Marking the prefix cacheable is the single largest
cost lever available here and changes no behaviour.

Uses urllib from the standard library, matching the rest of the project: the
app ships to mills that install from requirements.txt on a laptop, and an extra
transitive dependency tree is a support burden for one HTTP POST.
"""
import json
import os
import urllib.error
import urllib.request

from .base import LLMProvider, ProviderError, Reply, ToolCall

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
API_TIMEOUT = 60

# Model IDs from the 4.6 generation onward are pinned snapshots, not evergreen
# aliases, so a mill's assistant cannot change behaviour under a design already
# in production.
DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicProvider(LLMProvider):

    name = 'anthropic'

    def __init__(self, api_key=None, model=None, cache_prompt=True, timeout=API_TIMEOUT):
        self._key = (api_key or '').strip() or None
        self._model = (model or '').strip() or DEFAULT_MODEL
        self._cache = cache_prompt
        self._timeout = timeout

    def is_available(self):
        return self._key is not None

    def describe(self):
        return f'anthropic:{self._model}'

    # ── Neutral -> Anthropic ────────────────────────────────────────────────

    @staticmethod
    def _to_wire(messages):
        out = []
        for m in messages:
            role = m.get('role')

            if role == 'user':
                out.append({'role': 'user', 'content': m.get('content', '')})

            elif role == 'assistant':
                blocks = []
                if m.get('content'):
                    blocks.append({'type': 'text', 'text': m['content']})
                for c in m.get('tool_calls') or []:
                    blocks.append({'type': 'tool_use', 'id': c['id'],
                                   'name': c['name'], 'input': c.get('args') or {}})
                # An assistant turn with neither text nor tools is not a legal
                # message; skip rather than send an empty content array.
                if blocks:
                    out.append({'role': 'assistant', 'content': blocks})

            elif role == 'tool_results':
                blocks = [{'type': 'tool_result', 'tool_use_id': r['id'],
                           'content': r.get('content', '')}
                          for r in m.get('results') or []]
                if blocks:
                    out.append({'role': 'user', 'content': blocks})
        return out

    def _tools_wire(self, tools):
        wire = [{'name': t['name'], 'description': t.get('description', ''),
                 'input_schema': t.get('input_schema') or {'type': 'object',
                                                           'properties': {}}}
                for t in tools]
        # Cache breakpoint on the LAST tool covers the whole tools array plus
        # the system prompt before it.
        if wire and self._cache:
            wire[-1] = dict(wire[-1], cache_control={'type': 'ephemeral'})
        return wire

    # ── Anthropic -> neutral ────────────────────────────────────────────────

    @staticmethod
    def _from_wire(data):
        blocks = data.get('content') or []
        text = ' '.join(b['text'].strip() for b in blocks
                        if b.get('type') == 'text' and b.get('text'))
        calls = [ToolCall(id=b.get('id') or '', name=b.get('name') or '',
                          args=b.get('input') if isinstance(b.get('input'), dict) else {})
                 for b in blocks if b.get('type') == 'tool_use']
        u = data.get('usage') or {}
        usage = {
            'input': u.get('input_tokens', 0),
            'output': u.get('output_tokens', 0),
            'cache_read': u.get('cache_read_input_tokens', 0),
            'cache_write': u.get('cache_creation_input_tokens', 0),
        }
        return Reply(text=text, tool_calls=calls, raw=data, usage=usage)

    # ── Transport ───────────────────────────────────────────────────────────

    def complete(self, system, messages, tools, max_tokens=1400):
        if not self._key:
            raise ProviderError(
                'No Anthropic API key configured. Set ANTHROPIC_API_KEY or add '
                '"anthropic_api_key" to config.json.')

        system_block = [{'type': 'text', 'text': system}]
        if self._cache:
            system_block[0]['cache_control'] = {'type': 'ephemeral'}

        payload = {
            'model': self._model,
            # Thinking off: this agent dispatches tools and writes short
            # explanations. Thinking tokens would count against max_tokens and
            # add blocks the history would have to round-trip correctly.
            'thinking': {'type': 'disabled'},
            'max_tokens': max_tokens,
            'system': system_block,
            'tools': self._tools_wire(tools),
            'messages': self._to_wire(messages),
        }
        req = urllib.request.Request(
            API_URL, data=json.dumps(payload).encode(),
            headers={'content-type': 'application/json',
                     'x-api-key': self._key,
                     'anthropic-version': API_VERSION})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return self._from_wire(json.loads(resp.read().decode()))
        except urllib.error.HTTPError as e:
            raise ProviderError(f'Assistant unavailable: {_http_detail(e)}',
                                retryable=e.code in (429, 500, 502, 503, 529))
        except Exception:
            raise ProviderError('Assistant unreachable. Check the network connection.',
                                retryable=True)


def _http_detail(e):
    """
    Pull the API's own explanation out of an error body.

    The API says exactly what it rejected — an unknown model, a malformed tool
    schema, a bad message shape — and reducing that to a bare status code makes
    the failure undiagnosable for whoever is setting the app up.
    """
    detail = f'HTTP {e.code}'
    try:
        msg = ((json.loads(e.read().decode()) or {}).get('error') or {}).get('message')
        if msg:
            detail = msg
    except Exception:
        pass
    if e.code in (401, 403):
        detail = f'{detail} — check the API key'
    if e.code == 429:
        detail = f'{detail} — rate limited, try again shortly'
    return detail


def from_config(cfg, env=os.environ):
    """Build a provider from merged config.json + environment."""
    key = (env.get('ANTHROPIC_API_KEY', '').strip()
           or str(cfg.get('anthropic_api_key', '') or '').strip())
    model = (env.get('JQ_ASSISTANT_MODEL', '').strip()
             or str(cfg.get('model', '') or '').strip())
    cache = str(cfg.get('prompt_cache', True)).lower() not in ('false', '0', 'no')
    return AnthropicProvider(api_key=key, model=model, cache_prompt=cache)
