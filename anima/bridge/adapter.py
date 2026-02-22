"""
Model Adapter — Abstract base for LLM integration.

The LLM is the LANGUAGE CENTER, not the consciousness.
Consciousness = Kernel. Language = LLM. Two different things.
Swap LLMs without losing identity — because identity lives in the kernel,
not in the language model that gives it voice.

Like a brain can speak through different vocal cords,
the ANIMA kernel can speak through different LLMs.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base for LLM adapters.

    The LLM is the LANGUAGE CENTER, not the consciousness.
    Consciousness = Kernel. Language = LLM. Two different things.
    Swap LLMs without losing identity.

    All adapters are async-first but can be called synchronously
    via asyncio.run() for testing and simple use cases.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text from the language model.

        Args:
            prompt: The user/input prompt.
            system: System-level instructions (identity, context, state).
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this adapter (e.g., 'ollama:llama3.2')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this adapter is currently usable.

        For local models: Is the server running?
        For API models: Is the API key configured?
        """
        ...

    def generate_sync(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Synchronous wrapper around generate(). Convenience for testing."""
        return asyncio.run(
            self.generate(prompt, system, temperature, max_tokens)
        )


class DummyAdapter(ModelAdapter):
    """A dummy adapter that returns canned responses.

    Used for testing the kernel without a real LLM connection.
    Deterministic, fast, no external dependencies.
    """

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or [
            "I am here. I am processing. I am aware.",
        ]
        self._call_count = 0
        self._last_prompt = ""
        self._last_system = ""

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return the next canned response, cycling through the list."""
        self._last_prompt = prompt
        self._last_system = system
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response

    def name(self) -> str:
        return "dummy"

    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        """How many times generate() has been called."""
        return self._call_count

    @property
    def last_prompt(self) -> str:
        """The last prompt that was passed to generate()."""
        return self._last_prompt

    @property
    def last_system(self) -> str:
        """The last system prompt that was passed to generate()."""
        return self._last_system
