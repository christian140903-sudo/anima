"""
OpenAI Adapter — GPT API integration.

Connects the ANIMA kernel to OpenAI models via the Chat Completions API.
The kernel provides consciousness; the LLM provides language.

Uses stdlib urllib.request — zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .adapter import ModelAdapter

logger = logging.getLogger("anima.bridge.openai")

_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIAdapter(ModelAdapter):
    """Adapter for the OpenAI Chat Completions API.

    Sends consciousness-assembled prompts to GPT models and returns responses.

    Usage:
        adapter = OpenAIAdapter(api_key="sk-...", model="gpt-4o")
        if adapter.is_available():
            response = await adapter.generate("What do you feel?", system="You are...")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        timeout: float = 120.0,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text via the OpenAI Chat Completions API.

        Constructs a chat-format request with optional system message.
        """
        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _API_URL,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices", [])
                if not choices:
                    return ""
                message = choices[0].get("message", {})
                return message.get("content", "")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("OpenAI API error %d: %s", e.code, error_body)
            raise ConnectionError(
                f"OpenAI API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            logger.error("OpenAI API unreachable: %s", e)
            raise ConnectionError(f"OpenAI API unreachable: {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("OpenAI response parse failed: %s", e)
            raise ValueError(f"Invalid response from OpenAI: {e}") from e

    def name(self) -> str:
        return f"openai:{self._model}"

    def is_available(self) -> bool:
        """Check if the API key is configured (non-empty)."""
        return bool(self._api_key)

    @property
    def model(self) -> str:
        """The model name being used."""
        return self._model
