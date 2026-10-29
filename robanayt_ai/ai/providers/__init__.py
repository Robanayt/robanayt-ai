"""Concrete AI providers."""

from robanayt_ai.ai.providers.ollama import OllamaProvider
from robanayt_ai.ai.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["OllamaProvider", "OpenAICompatibleProvider"]
