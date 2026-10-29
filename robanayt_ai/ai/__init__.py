"""JARVIS_AI intelligence layer (Phase 2).

Provider-agnostic: the rest of JARVIS talks to `AIService` and never to a
vendor SDK. Providers (Ollama local, OpenAI-compatible) sit behind a router
and a formal `Provider` protocol; tests use fakes, never live services.

Nothing here contacts the network during import or application startup -
that only happens when a generation is explicitly requested.
"""

from robanayt_ai.ai.context import DEFAULT_MAX_MESSAGES, ConversationContext
from robanayt_ai.ai.errors import (
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
from robanayt_ai.ai.models import (
    AIChunk,
    AIRequest,
    AIResponse,
    AIMessage,
    AITool,
    ProviderStatus,
    Usage,
)
from robanayt_ai.ai.provider import (
    LOCAL,
    REMOTE,
    Provider,
    ProviderCapabilities,
    ProviderProfile,
    profile_of,
)
from robanayt_ai.ai.providers import OllamaProvider, OpenAICompatibleProvider
from robanayt_ai.ai.router import (
    LOCAL_ONLY,
    ProviderRouter,
    RoutingDecision,
    RoutingPolicy,
)
from robanayt_ai.ai.service import AIService
from robanayt_ai.ai.streaming import StreamAccumulator, accumulate, as_stream

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
