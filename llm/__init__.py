"""
Swappable LLM backends for the jacquard assistant.

Nothing outside this package should contain a vendor's field names. The agent
imports the neutral types and the registry; the adapters own the wire formats.
"""
from .base import (LLMProvider, ProviderError, Reply, ToolCall,
                   assistant_msg, tool_results_msg, user_msg)
from .registry import (describe, get_provider, is_available, provider, reset,
                       set_provider)

__all__ = [
    'LLMProvider', 'ProviderError', 'Reply', 'ToolCall',
    'user_msg', 'assistant_msg', 'tool_results_msg',
    'provider', 'get_provider', 'set_provider', 'reset',
    'is_available', 'describe',
]
