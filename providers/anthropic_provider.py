"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

import anthropic

import config
from providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Calls Claude via the Anthropic Python SDK."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self._model = model or config.DEFAULT_MODEL
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message to Claude and return the response text."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Anthropic"
