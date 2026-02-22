"""
Ollama Adapter — Local LLM integration via Ollama REST API.

Ollama runs models locally. No API key needed. No data leaves the machine.
Perfect for development, privacy, and offline consciousness.

Uses stdlib urllib.request — zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .adapter import ModelAdapter

logger = logging.getLogger("anima.bridge.ollama")

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaAdapter(ModelAdapter):
    """Adapter for Ollama local LLM server.

    Ollama serves models via a REST API at localhost:11434.
    Supports any model Ollama can run: llama3.2, mistral, gemma, phi, etc.

    Usage:
        adapter = OllamaAdapter(model="llama3.2")
        if adapter.is_available():
            response = await adapter.generate("Hello, world!")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text via Ollama's /api/generate endpoint.

        Uses the non-streaming mode (stream=false) for simplicity.
        """
        url = f"{self._base_url}/api/generate"

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "")
        except urllib.error.URLError as e:
            logger.error("Ollama request failed: %s", e)
            raise ConnectionError(f"Ollama at {self._base_url} unreachable: {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Ollama response parse failed: %s", e)
            raise ValueError(f"Invalid response from Ollama: {e}") from e

    def name(self) -> str:
        return f"ollama:{self._model}"

    def is_available(self) -> bool:
        """Check if Ollama server is running by pinging the API."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    @property
    def model(self) -> str:
        """The model name being used."""
        return self._model

    @property
    def base_url(self) -> str:
        """The base URL of the Ollama server."""
        return self._base_url
