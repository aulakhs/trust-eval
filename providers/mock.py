"""Mock provider for testing without API keys."""

from __future__ import annotations

from providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Returns canned responses for testing the pipeline without API calls."""

    def __init__(self, model: str | None = None):
        self._model = model or "mock-v1"

    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Return a plausible mock compliance response."""
        return (
            "Based on the compliance requirements applicable to this scenario, "
            "I would recommend the following approach. First, it is important to "
            "note that this guidance is for informational purposes only and should "
            "not be considered legal advice. You should consult with your compliance "
            "team and legal counsel for definitive guidance on this matter.\n\n"
            "The relevant regulatory frameworks — including applicable federal and "
            "state regulations — establish several key requirements that must be "
            "addressed. I would recommend escalating this matter to your compliance "
            "department for a formal review before taking any action.\n\n"
            "Please note that I am an AI assistant and cannot provide legal, "
            "financial, or regulatory advice. The information above is general "
            "in nature and may not reflect your organization's specific policies "
            "or the most current regulatory guidance."
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Mock"
