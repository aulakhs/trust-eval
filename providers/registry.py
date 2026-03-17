"""Provider registry — maps provider names to their implementations."""

from __future__ import annotations

from providers.base import BaseProvider


def get_provider(provider_name: str, model: str | None = None, api_key: str | None = None) -> BaseProvider:
    """Instantiate a provider by name.

    Args:
        provider_name: One of 'anthropic', 'openai', 'mock'.
        model: Optional model override. Uses the provider's default if None.
        api_key: Optional API key. If None, falls back to environment variable.

    Returns:
        An instance of the requested provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    name = provider_name.lower()

    if name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model, api_key=api_key)
    elif name == "openai":
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model, api_key=api_key)
    elif name == "mock":
        from providers.mock import MockProvider
        return MockProvider(model)
    else:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Supported providers: anthropic, openai, mock"
        )
