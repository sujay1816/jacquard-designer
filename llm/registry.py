"""
Provider selection.

One place decides which backend the agent talks to, so no other module needs a
vendor name in it. Resolution order is environment first, then config.json,
then a default — matching how the rest of the app already resolves settings, so
an operator who knows how to set the API key does not need to learn a second
convention.

Selecting a provider is deliberately cheap and does no network I/O: the
Generator page asks whether the assistant is available on every load, and a
health probe there would put a model server's latency in front of a page that
does not need the model at all.
"""
import json
import os

from . import anthropic as _anthropic
from . import openai_compat as _openai
from .base import LLMProvider, ProviderError, Reply, ToolCall  # re-exported

PROVIDERS = {
    'anthropic': _anthropic.from_config,
    'openai_compat': _openai.from_config,
}

# Convenience aliases for whoever is editing config.json at 2am in a mill.
ALIASES = {
    'claude': 'anthropic',
    'llama': 'openai_compat',
    'ollama': 'openai_compat',
    'openai': 'openai_compat',
    'local': 'openai_compat',
    'vllm': 'openai_compat',
    'together': 'openai_compat',
    'groq': 'openai_compat',
    'openrouter': 'openai_compat',
}

DEFAULT_PROVIDER = 'anthropic'


def load_config(path=None):
    """Read config.json next to the project root, or {} if absent."""
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'config.json')
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def resolve_name(cfg=None, env=os.environ):
    cfg = load_config() if cfg is None else cfg
    raw = (env.get('JQ_LLM_PROVIDER', '').strip()
           or str(cfg.get('llm_provider', '') or '').strip()
           or DEFAULT_PROVIDER).lower()
    return ALIASES.get(raw, raw)


def get_provider(cfg=None, env=os.environ):
    """
    Build the configured provider.

    An unknown name falls back to the default rather than raising: a typo in
    config.json should not take the whole app down, and the assistant panel
    will report itself unavailable if the fallback has no key either.
    """
    cfg = load_config() if cfg is None else cfg
    factory = PROVIDERS.get(resolve_name(cfg, env)) or PROVIDERS[DEFAULT_PROVIDER]
    return factory(cfg, env)


# The agent calls this on every turn. Cached so a turn does not re-read
# config.json, but resettable so tests and a settings change can swap backends
# without a restart.
_cached = None


def provider():
    global _cached
    if _cached is None:
        _cached = get_provider()
    return _cached


def set_provider(p):
    """Install a provider explicitly — used by tests and by runtime switching."""
    global _cached
    _cached = p


def reset():
    global _cached
    _cached = None


def is_available():
    try:
        return provider().is_available()
    except Exception:
        return False


def describe():
    try:
        return provider().describe()
    except Exception:
        return 'unavailable'
