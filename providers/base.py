"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Common interface for all LLM providers.

    Subclasses implement send_message to call their respective APIs.
    The eval runner uses this interface so it never touches provider-specific code.
    """

    @abstractmethod
    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message and return the response text.

        Args:
            system_prompt: System-level instructions for the model.
            user_message: The user's question or prompt.

        Returns:
            The model's response as a string.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The specific model identifier (e.g., 'claude-sonnet-4-20250514')."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The provider name (e.g., 'Anthropic', 'OpenAI')."""
