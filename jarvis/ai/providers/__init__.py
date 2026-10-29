"""Concrete AI providers."""

from jarvis.ai.providers.ollama import OllamaProvider
from jarvis.ai.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["OllamaProvider", "OpenAICompatibleProvider"]
