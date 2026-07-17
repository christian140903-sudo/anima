"""Model-agnostic LLM integration for ANIMA's durable state context.

Adapters can be swapped while the application-controlled kernel state remains.
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
