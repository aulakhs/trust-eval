"""OpenAI (GPT) provider implementation."""

from __future__ import annotations

from providers.base import BaseProvider

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(BaseProvider):
    """Calls GPT models via the OpenAI Python SDK."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        if not HAS_OPENAI:
            raise ImportError(
                "The openai package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            )
        self._model = model or "gpt-4o"
        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message to a GPT model and return the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
