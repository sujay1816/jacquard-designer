"""
Provider-neutral LLM interface.

The agent must not know which model vendor it is talking to. Everything above
this layer — the tool functions, the fidelity scoring, the BMP writer — is
deterministic code that never sees a wire format, and that separation is what
lets the model be swapped without re-auditing loom safety.

So this module defines one neutral vocabulary and nothing else:

  * ToolCall  — the model wants a tool run, by name, with arguments.
  * Reply     — one model turn: some text, zero or more tool calls.
  * Msg       — one entry of conversation history, in OUR format, not a vendor's.

Adapters translate Msg/Reply to and from their wire format. They are the only
files allowed to contain vendor-specific field names.

Why a neutral history rather than storing raw vendor blocks: the two major
wire formats disagree about where tool results live. Anthropic puts tool_use
blocks inside the assistant turn and tool_result blocks inside a *user* turn;
OpenAI-compatible APIs use a tool_calls array on the assistant message and a
separate role="tool" message per result. History stored in either shape cannot
be replayed to the other, which would mean a provider switch silently loses
the conversation. Stored neutrally, it replays to both.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProviderError(Exception):
    """
    A provider could not complete a turn.

    Carries a message already suitable for showing a weaver — adapters are
    responsible for turning HTTP codes and vendor error payloads into
    something actionable ("check the API key") rather than a status number.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


@dataclass
class ToolCall:
    """A request from the model to run one tool."""
    id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reply:
    """
    One assistant turn, normalised.

    `text` may be empty when the model only called tools. `raw` keeps the
    vendor payload for logging and cost accounting; nothing above this layer
    should read it.
    """
    text: str = ''
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Optional[Any] = None
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ── Neutral message helpers ─────────────────────────────────────────────────
# History is a list of plain dicts so it stays JSON-serialisable: sessions may
# outlive a process, and a dataclass history would need a custom encoder.

def user_msg(text: str, images: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    A user turn, optionally carrying images.

    Each image is {'media_type': 'image/png', 'data': '<base64>'}. Adapters
    translate to their own shape — Anthropic wants a source block, OpenAI wants
    a data URI — so callers never encode either.
    """
    msg: Dict[str, Any] = {'role': 'user', 'content': text}
    if images:
        msg['images'] = images
    return msg


def assistant_msg(reply: Reply) -> Dict[str, Any]:
    return {
        'role': 'assistant',
        'content': reply.text or '',
        'tool_calls': [{'id': c.id, 'name': c.name, 'args': c.args}
                       for c in reply.tool_calls],
    }


def tool_results_msg(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    `results` entries are {'id', 'name', 'content'}, content a string, plus an
    optional 'images' list in the same shape user_msg takes.

    Images on a tool result are how the agent looks at its own work: a tool
    renders the design and hands the picture back, and the model reads it in
    the same turn. The alternative — a tool that calls the model itself —
    would nest one conversation inside another and put a second, unlogged
    model turn outside the loop that counts rounds and tokens.
    """
    return {'role': 'tool_results', 'results': results}


class LLMProvider(ABC):
    """
    One model backend.

    Implementations translate the neutral types above to their wire format and
    back. They must not mutate `messages`, and must raise ProviderError rather
    than returning a sentinel — the caller distinguishes "the model replied"
    from "the backend failed" and cannot do that from a Reply alone.
    """

    name = 'provider'

    # Whether this backend can read images. False makes the agent skip its
    # self-critique step rather than send an image that will be ignored, which
    # would burn a round and return a confident opinion about nothing.
    supports_vision = False

    @abstractmethod
    def complete(self, system: str, messages: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]], max_tokens: int = 1400) -> Reply:
        """
        Run one turn.

        system   : system prompt string.
        messages : neutral history (see helpers above).
        tools    : neutral tool specs — {'name', 'description', 'input_schema'},
                   which is the shape agent_engine.TOOLS already uses.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """True when this provider is configured well enough to try a call."""

    def describe(self) -> str:
        """One line for the UI and the /api/build endpoint."""
        return self.name
