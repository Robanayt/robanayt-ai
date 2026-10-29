"""Voice provider selection - deliberately separate from AI routing.

Phase 7 decides which *model* answers; this router decides which *ears* listen
and which *mouth* speaks. They are different resources with different costs and
different privacy consequences, so a policy that is cheap for a model says
nothing about a speech backend.

Selection never invents availability: a provider is usable only when it is
configured **and** reports itself available. Privacy filters are applied here,
once, so neither a session nor the UI can accidentally send audio off-machine.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.voice.errors import VoiceConfigurationError, VoiceUnavailableError
from jarvis.voice.models import LOCAL, REMOTE, VoiceSelection, VoiceState
from jarvis.voice.provider import SpeechToTextProvider, TextToSpeechProvider

logger = logging.getLogger(__name__)

# Both selection failures mean "cannot use this provider"; keeping them in one
# tuple lets `explain()` report a failed selection without duplicating logic.
_SELECTION_ERRORS = (VoiceUnavailableError, VoiceConfigurationError)

AUTO = "auto"
NONE = "none"

LOCAL_ONLY = "local_only"
ANY_PRIVACY = "any"
VALID_PRIVACY_MODES: frozenset[str] = frozenset({LOCAL_ONLY, ANY_PRIVACY})


class VoiceRouter:
    """Chooses STT and TTS providers. Owns no device and performs no I/O."""

    def __init__(
        self,
        *,
        stt: tuple[SpeechToTextProvider, ...] = (),
        tts: tuple[TextToSpeechProvider, ...] = (),
        privacy: str = LOCAL_ONLY,
        allow_remote: bool = False,
    ) -> None:
        self._stt: dict[str, SpeechToTextProvider] = {}
        self._tts: dict[str, TextToSpeechProvider] = {}
        for provider in stt:
            self._stt[str(provider.id)] = provider
        for provider in tts:
            self._tts[str(provider.id)] = provider
        self.privacy = normalize_privacy(privacy)
        self.allow_remote = bool(allow_remote)

    # -- registration -------------------------------------------------------

    def register_stt(self, provider: SpeechToTextProvider) -> None:
        self._stt[str(provider.id)] = provider

    def register_tts(self, provider: TextToSpeechProvider) -> None:
        self._tts[str(provider.id)] = provider

    @property
    def stt_ids(self) -> tuple[str, ...]:
        return tuple(self._stt)

    @property
    def tts_ids(self) -> tuple[str, ...]:
        return tuple(self._tts)

    # -- selection ----------------------------------------------------------

    def select_stt(
        self,
        name: str = AUTO,
        *,
        probe: bool = True,
        language: str = "",
    ) -> VoiceSelection:
        """Pick a speech-to-text provider, or raise with the real reason."""
        return self._select(self._stt, name, probe=probe, kind="stt")

    def select_tts(self, name: str = AUTO, *, probe: bool = True) -> VoiceSelection:
        """Pick a text-to-speech provider, or raise with the real reason."""
        return self._select(self._tts, name, probe=probe, kind="tts")

    # -- reporting ----------------------------------------------------------

    def status(self, *, probe: bool = False) -> dict[str, list[dict[str, Any]]]:
        """Availability of every provider of both kinds. No side effects."""
        return {
            "stt": [provider.status(probe=probe).as_dict() for provider in self._stt.values()],
            "tts": [provider.status(probe=probe).as_dict() for provider in self._tts.values()],
        }

    def explain(self, *, probe: bool = True) -> dict[str, Any]:
        """What would be used, why, and why the rest would not. No audio."""
        summary: dict[str, Any] = {
            "privacy": self.privacy,
            "allow_remote": self.allow_remote,
            "stt": self._explain_side(self._stt, "stt", probe=probe),
            "tts": self._explain_side(self._tts, "tts", probe=probe),
        }
        return summary

    def _explain_side(
        self, providers: dict[str, Any], kind: str, *, probe: bool
    ) -> dict[str, Any]:
        entries = []
        for provider in providers.values():
            # One probe per provider, not one per field: a probe is a real
            # system call (a helper process on Windows), so asking the same
            # question four times would quadruple the cost of diagnostics.
            state = provider.status(probe=probe)
            entries.append(
                {
                    "provider": provider.id,
                    "kind": provider.kind,
                    "configured": state.configured,
                    "available": state.available,
                    "message": state.message,
                    "tested_on_hardware": state.tested_on_hardware,
                    "capabilities": {
                        "streaming": provider.capabilities.streaming,
                        "local": provider.capabilities.local,
                        "cancel_supported": provider.capabilities.cancel_supported,
                    },
                }
            )
        side: dict[str, Any] = {"providers": entries}
        try:
            selection = self._select(providers, AUTO, probe=probe, kind=kind)
        except _SELECTION_ERRORS as exc:
            side.update({"selected": None, "reason": str(exc), "rejected": []})
            return side
        side.update(
            {
                "selected": selection.name or None,
                "reason": selection.reason,
                "rejected": [
                    {"provider": provider, "reason": reason}
                    for provider, reason in selection.rejections
                ],
            }
        )
        return side

    # -- internals ----------------------------------------------------------

    def _select(
        self,
        providers: dict[str, Any],
        name: str,
        *,
        probe: bool,
        kind: str,
    ) -> VoiceSelection:
        wanted = (name or AUTO).strip().lower()
        if wanted not in {AUTO, NONE} and wanted not in providers:
            known = ", ".join(providers) or "нет зарегистрированных провайдеров"
            raise VoiceConfigurationError(
                f"unknown {kind} provider {wanted!r}; known: {known}"
            )

        if wanted == NONE:
            raise VoiceUnavailableError(f"{kind}: провайдер отключён настройкой '{NONE}'")

        candidates = (
            [providers[wanted]]
            if wanted != AUTO
            else list(providers.values())
        )
        explicit = wanted != AUTO

        rejections: list[tuple[str, str]] = []
        usable: list[Any] = []
        for candidate in candidates:
            state = candidate.status(probe=probe)
            blocked = self._blocked(candidate, state)
            if blocked is None:
                usable.append(candidate)
            else:
                rejections.append((str(candidate.id), blocked))

        if not usable:
            details = "; ".join(f"{item}: {reason}" for item, reason in rejections)
            target = f"provider {wanted!r}" if explicit else "подходящий провайдер"
            raise VoiceUnavailableError(
                f"{kind}: {target} недоступен ({details})" if details
                else f"{kind}: провайдер не зарегистрирован"
            )

        chosen = usable[0]
        reason = (
            f"{kind}: выбран '{chosen.id}' ({chosen.kind})"
            if explicit
            else f"{kind}: выбран '{chosen.id}' ({chosen.kind}, приватность: {self.privacy})"
        )
        return VoiceSelection(
            provider=chosen,
            reason=reason,
            explicit=explicit,
            rejections=tuple(rejections),
        )

    def _blocked(self, provider: Any, state: Any) -> str | None:
        """Why a provider cannot be used, or `None` when it can."""
        if not state.configured and not state.available:
            return state.message or "не настроен"
        if getattr(provider, "kind", LOCAL) == REMOTE and not self.allow_remote:
            return "удалённые провайдеры запрещены (voice.allow_remote=false)"
        if self.privacy == LOCAL_ONLY and getattr(provider, "kind", LOCAL) == REMOTE:
            return "приватность local_only запрещает удалённые провайдеры"
        if not state.available:
            return state.message or "недоступен"
        return None


def normalize_privacy(value: str | None) -> str:
    """Validate a privacy mode instead of silently ignoring a typo."""
    if value is None:
        return LOCAL_ONLY
    candidate = str(value).strip().lower()
    if candidate not in VALID_PRIVACY_MODES:
        raise VoiceConfigurationError(
            f"unknown voice privacy mode {value!r}; "
            f"allowed: {', '.join(sorted(VALID_PRIVACY_MODES))}"
        )
    return candidate


# Local alias so the helper above can stay readable.
VoiceError_ = (VoiceUnavailableError, VoiceConfigurationError)


def state_for(is_available: bool) -> str:
    """Map availability onto the honest state name."""
    return VoiceState.IDLE.value if is_available else VoiceState.UNAVAILABLE.value
