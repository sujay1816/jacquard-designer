"""
OpenAI-compatible adapter — one file, most of the open-model world.

Ollama, vLLM, llama.cpp's server, Together, Groq, Fireworks, DeepInfra and
OpenRouter all expose /v1/chat/completions with the same tool-calling shape, so
running Llama locally and falling back to a hosted endpoint are the same code
path with a different base_url.

Wire shape this file owns, and where it differs from Anthropic:
  * the system prompt is a MESSAGE with role="system", not a top-level field
  * tools are wrapped: {"type": "function", "function": {..., "parameters": …}}
  * tool calls live in a `tool_calls` array on the assistant message
  * each result is its own message with role="tool" and a tool_call_id
  * **arguments arrive as a JSON string, not an object**

That last point is the one that bites. A frontier model emits clean JSON; a
7B–70B open model emits arguments wrapped in markdown fences, with trailing
commas, with single quotes, or as a bare object when the schema wanted a
string. _parse_args below repairs the common cases rather than letting a
recoverable formatting slip surface to the weaver as "the assistant failed".

What it deliberately does NOT do is guess at *meaning*. A repaired call still
goes through the same run_tool dispatch and the same range checks as any other,
so a model that asks for 9,000 pins is still refused. Repair is about syntax.
"""
import json
import os
import re
import urllib.error
import urllib.request

from .base import LLMProvider, ProviderError, Reply, ToolCall

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible port
DEFAULT_MODEL = "llama3.3:70b"
API_TIMEOUT = 120        # local inference on CPU/consumer GPU is slow


class OpenAICompatProvider(LLMProvider):

    name = 'openai_compat'

    def __init__(self, base_url=None, api_key=None, model=None,
                 timeout=API_TIMEOUT, temperature=0.0):
        self._base = (base_url or DEFAULT_BASE_URL).rstrip('/')
        self._key = (api_key or '').strip() or None
        self._model = (model or '').strip() or DEFAULT_MODEL
        self._timeout = timeout
        # Deterministic by default. The pixels are already reproducible; the
        # explanations around them should not wobble between identical runs.
        self._temperature = temperature

    def is_available(self):
        # A local server needs no key, so a base_url is the only hard
        # requirement. Whether it is actually listening is discovered on call —
        # probing here would add a round trip to every page load.
        return bool(self._base)

    def describe(self):
        host = re.sub(r'^https?://', '', self._base).split('/')[0]
        return f'openai-compat:{self._model} @ {host}'

    # ── Neutral -> OpenAI ───────────────────────────────────────────────────

    @staticmethod
    def _to_wire(system, messages):
        out = [{'role': 'system', 'content': system}]
        for m in messages:
            role = m.get('role')

            if role == 'user':
                out.append({'role': 'user', 'content': m.get('content', '')})

            elif role == 'assistant':
                msg = {'role': 'assistant', 'content': m.get('content') or None}
                calls = m.get('tool_calls') or []
                if calls:
                    msg['tool_calls'] = [
                        {'id': c['id'], 'type': 'function',
                         'function': {'name': c['name'],
                                      'arguments': json.dumps(c.get('args') or {})}}
                        for c in calls]
                if msg['content'] or msg.get('tool_calls'):
                    out.append(msg)

            elif role == 'tool_results':
                # One message per result, unlike Anthropic's single user turn.
                for r in m.get('results') or []:
                    out.append({'role': 'tool', 'tool_call_id': r['id'],
                                'name': r.get('name', ''),
                                'content': r.get('content', '')})
        return out

    @staticmethod
    def _tools_wire(tools):
        return [{'type': 'function',
                 'function': {'name': t['name'],
                              'description': t.get('description', ''),
                              'parameters': t.get('input_schema')
                              or {'type': 'object', 'properties': {}}}}
                for t in tools]

    # ── OpenAI -> neutral ───────────────────────────────────────────────────

    @classmethod
    def _from_wire(cls, data):
        choices = data.get('choices') or []
        if not choices:
            raise ProviderError('Model returned no reply.')
        msg = choices[0].get('message') or {}

        text = (msg.get('content') or '').strip()
        calls = []
        for i, tc in enumerate(msg.get('tool_calls') or []):
            fn = tc.get('function') or {}
            name = fn.get('name') or ''
            if not name:
                continue
            calls.append(ToolCall(
                # Some servers omit the id. A tool_call_id is mandatory on the
                # way back, so synthesise a stable one rather than send null.
                id=tc.get('id') or f'call_{i}',
                name=name,
                args=_parse_args(fn.get('arguments'))))

        # Fallback: several local builds ignore the tools array and emit the
        # call as text. Recovering it is the difference between the agent
        # working on Llama and not.
        if not calls and text:
            recovered = _recover_inline_call(text)
            if recovered:
                calls.append(recovered)
                text = ''

        u = data.get('usage') or {}
        return Reply(text=text, tool_calls=calls, raw=data,
                     usage={'input': u.get('prompt_tokens', 0),
                            'output': u.get('completion_tokens', 0),
                            'cache_read': 0, 'cache_write': 0})

    # ── Transport ───────────────────────────────────────────────────────────

    def complete(self, system, messages, tools, max_tokens=1400):
        payload = {
            'model': self._model,
            'max_tokens': max_tokens,
            'temperature': self._temperature,
            'messages': self._to_wire(system, messages),
        }
        if tools:
            payload['tools'] = self._tools_wire(tools)
            payload['tool_choice'] = 'auto'

        headers = {'content-type': 'application/json'}
        if self._key:
            headers['authorization'] = f'Bearer {self._key}'

        req = urllib.request.Request(
            f'{self._base}/chat/completions',
            data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return self._from_wire(json.loads(resp.read().decode()))
        except urllib.error.HTTPError as e:
            raise ProviderError(f'Assistant unavailable: {_http_detail(e)}',
                                retryable=e.code in (429, 500, 502, 503))
        except ProviderError:
            raise
        except urllib.error.URLError:
            raise ProviderError(
                f'Cannot reach the model server at {self._base}. '
                'Is it running?', retryable=True)
        except Exception:
            raise ProviderError('Assistant unreachable. Check the network connection.',
                                retryable=True)


# ── Argument repair ─────────────────────────────────────────────────────────

_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$')
_TRAILING_COMMA = re.compile(r',\s*([}\]])')


def _parse_args(raw):
    """
    Turn a model's `arguments` field into a dict, tolerating common mangling.

    Returns {} rather than raising: an empty argument set reaches run_tool,
    which applies the tool's own defaults and validation. A malformed call
    should degrade to a defaulted call the weaver can correct, not a traceback.
    """
    if isinstance(raw, dict):
        return raw                      # some servers already decode it
    if not isinstance(raw, str) or not raw.strip():
        return {}

    text = _FENCE.sub('', raw.strip())
    for attempt in (text,
                    _TRAILING_COMMA.sub(r'\1', text),
                    _TRAILING_COMMA.sub(r'\1', text.replace("'", '"'))):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            continue

    # Last resort: the first balanced {...} inside a chattier response.
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except (ValueError, TypeError):
                    break
    return {}


def _recover_inline_call(text):
    """
    Recover a tool call a model wrote as prose instead of using the tool API.

    Only fires on the shapes weak models actually produce — a JSON object with
    a name plus arguments/parameters/input — so ordinary prose containing a
    stray brace is not mistaken for a call.
    """
    args = _parse_args(text)
    name = args.get('name') or args.get('tool') or args.get('function')
    if not isinstance(name, str) or not name:
        return None
    for k in ('arguments', 'parameters', 'input', 'args'):
        inner = args.get(k)
        if isinstance(inner, dict):
            return ToolCall(id='call_recovered', name=name, args=inner)
        if isinstance(inner, str):
            return ToolCall(id='call_recovered', name=name, args=_parse_args(inner))
    return ToolCall(id='call_recovered', name=name, args={})


def _http_detail(e):
    detail = f'HTTP {e.code}'
    try:
        payload = json.loads(e.read().decode()) or {}
        err = payload.get('error')
        msg = err.get('message') if isinstance(err, dict) else (err or payload.get('message'))
        if msg:
            detail = str(msg)
    except Exception:
        pass
    if e.code in (401, 403):
        detail = f'{detail} — check the API key'
    if e.code == 404:
        detail = f'{detail} — check the model name and base URL'
    return detail


def from_config(cfg, env=os.environ):
    return OpenAICompatProvider(
        base_url=(env.get('JQ_LLM_BASE_URL', '').strip()
                  or str(cfg.get('llm_base_url', '') or '').strip()),
        api_key=(env.get('JQ_LLM_API_KEY', '').strip()
                 or str(cfg.get('llm_api_key', '') or '').strip()),
        model=(env.get('JQ_LLM_MODEL', '').strip()
               or str(cfg.get('llm_model', '') or '').strip()))
