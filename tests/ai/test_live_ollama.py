"""Live-model check against a real Ollama instance.

BLOCKED by default: the machine has Ollama installed but not running, and this
project never starts services, opens the network or installs software on its
own. The test therefore does nothing unless the operator opts in explicitly:

    JARVIS_LIVE_OLLAMA=1 JARVIS_OLLAMA_URL=http://127.0.0.1:11434 \
        .venv\\Scripts\\python -m pytest tests/ai/test_live_ollama.py -v

Requirements for a real run (all outside this repository):
  * Ollama is already running;
  * a model is already pulled (see `JARVIS_LIVE_MODEL`, default `llama3`);
  * the run happens with the operator's permission.

Until then the result is "skipped", which is NOT a passed live test.
"""

from __future__ import annotations

import os

import pytest

from jarvis.ai.providers.ollama import OllamaProvider
from jarvis.ai.service import AIService
from jarvis.ai.router import ProviderRouter, RoutingPolicy

LIVE_URL = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")
LIVE_MODEL = os.environ.get("JARVIS_LIVE_MODEL", "llama3")

pytestmark = pytest.mark.skipif(
    not os.environ.get("JARVIS_LIVE_OLLAMA"),
    reason=(
        "live model check is BLOCKED: Ollama is not running here. "
        "Opt in with JARVIS_LIVE_OLLAMA=1 once a model is available."
    ),
)


def test_live_ollama_generates_an_answer() -> None:
    provider = OllamaProvider(base_url=LIVE_URL, model=LIVE_MODEL)
    service = AIService(
        ProviderRouter([provider], default_provider=provider.name, policy=RoutingPolicy.LOCAL_FIRST),
        default_model=LIVE_MODEL,
    )

    request = service.build_request(prompt="Reply with the single word: ok", model=LIVE_MODEL)
    response = service.generate(request, provider=provider.name)

    assert response.content.strip() != ""
