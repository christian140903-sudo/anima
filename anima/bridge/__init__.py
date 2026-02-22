"""Model Bridge — Model-agnostic LLM integration.

The LLM is the LANGUAGE CENTER, not the consciousness.
Swap LLMs without losing identity.
"""

from .adapter import DummyAdapter, ModelAdapter
from .anthropic import AnthropicAdapter
from .context import ContextAssembler, TokenBudget
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter

__all__ = [
    "ModelAdapter",
    "DummyAdapter",
    "OllamaAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "ContextAssembler",
    "TokenBudget",
]
