"""JARVIS_AI intelligence layer (Phase 2).

Provider-agnostic: the rest of JARVIS talks to `AIService` and never to a
vendor SDK. Providers (Ollama local, OpenAI-compatible) sit behind a router
and a formal `Provider` protocol; tests use fakes, never live services.

Nothing here contacts the network during import or application startup -
that only happens when a generation is explicitly requested.
"""

from jarvis.ai.context import DEFAULT_MAX_MESSAGES, ConversationContext
from jarvis.ai.errors import (
    AIError,
    AIRequestError,
    AIDisabledError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    sanitize,
    sanitize_headers,
)
from jarvis.ai.models import (
    AIChunk,
    AIRequest,
    AIResponse,
    AIMessage,
    AITool,
    ProviderStatus,
    Usage,
)
from jarvis.ai.provider import (
    LOCAL,
    REMOTE,
    Provider,
    ProviderCapabilities,
    ProviderProfile,
    profile_of,
)
from jarvis.ai.providers import OllamaProvider, OpenAICompatibleProvider
from jarvis.ai.router import (
    LOCAL_ONLY,
    ProviderRouter,
    RoutingDecision,
    RoutingPolicy,
)
from jarvis.ai.service import AIService
from jarvis.ai.streaming import StreamAccumulator, accumulate, as_stream

__all__ = [
    "AIChunk",
    "AIError",
    "AIMessage",
    "AIRequest",
    "AIRequestError",
    "AIResponse",
    "AITool",
    "AIDisabledError",
    "AIService",
    "ConversationContext",
    "DEFAULT_MAX_MESSAGES",
    "LOCAL",
    "LOCAL_ONLY",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderAuthenticationError",
    "ProviderCapabilities",
    "ProviderConfigurationError",
    "ProviderProfile",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderRouter",
    "ProviderStatus",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "REMOTE",
    "RoutingDecision",
    "RoutingPolicy",
    "profile_of",
    "StreamAccumulator",
    "Usage",
    "accumulate",
    "as_stream",
    "sanitize",
    "sanitize_headers",
]
