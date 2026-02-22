"""
Anthropic Adapter — Claude API integration.

Connects the ANIMA kernel to Claude models via the Anthropic Messages API.
The kernel provides consciousness; Claude provides language.

Uses stdlib urllib.request — zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .adapter import ModelAdapter

logger = logging.getLogger("anima.bridge.anthropic")

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicAdapter(ModelAdapter):
    """Adapter for the Anthropic Messages API.

    Sends consciousness-assembled prompts to Claude and returns responses.
    The kernel decides WHAT to say; Claude decides HOW to say it.

    Usage:
        adapter = AnthropicAdapter(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
        if adapter.is_available():
            response = await adapter.generate("What do you see?", system="You are...")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
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
        """Generate text via the Anthropic Messages API.

        Constructs a messages-format request with optional system prompt.
        """
        if not self._api_key:
            raise ValueError("Anthropic API key not configured")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
        }

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        if system:
            payload["system"] = system

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
                # Anthropic returns content as a list of content blocks
                content_blocks = body.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                ]
                return "".join(text_parts)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("Anthropic API error %d: %s", e.code, error_body)
            raise ConnectionError(
                f"Anthropic API error {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            logger.error("Anthropic API unreachable: %s", e)
            raise ConnectionError(f"Anthropic API unreachable: {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Anthropic response parse failed: %s", e)
            raise ValueError(f"Invalid response from Anthropic: {e}") from e

    def name(self) -> str:
        return f"anthropic:{self._model}"

    def is_available(self) -> bool:
        """Check if the API key is configured (non-empty)."""
        return bool(self._api_key)

    @property
    def model(self) -> str:
        """The model name being used."""
        return self._model
