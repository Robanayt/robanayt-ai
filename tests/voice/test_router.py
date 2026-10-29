"""Voice provider selection: privacy and availability actually restrict it."""

from __future__ import annotations

import pytest

from robanayt_ai.voice.errors import VoiceConfigurationError, VoiceUnavailableError
from robanayt_ai.voice.router import ANY_PRIVACY, AUTO, LOCAL_ONLY, NONE, VoiceRouter, normalize_privacy
from tests.voice.fakes import RemoteSTT, ScriptedSTT, ScriptedTTS, UnavailableSTT


def _router(*, privacy: str = LOCAL_ONLY, allow_remote: bool = False, **kwargs):
    router = VoiceRouter(
        stt=kwargs.get("stt", ()), tts=kwargs.get("tts", ()),
        privacy=privacy, allow_remote=allow_remote,
    )
    return router


# -- registration -----------------------------------------------------------


def test_providers_are_registered_with_their_ids() -> None:
    router = _router(stt=(ScriptedSTT(),), tts=(ScriptedTTS(),))

    assert router.stt_ids == ("scripted_stt",)
    assert router.tts_ids == ("scripted_tts",)


def test_selection_picks_the_configured_provider() -> None:
    router = _router(stt=(ScriptedSTT(),), tts=(ScriptedTTS(),))

    assert router.select_stt().name == "scripted_stt"
    assert router.select_tts().name == "scripted_tts"


def test_unknown_explicit_provider_is_rejected() -> None:
    router = _router(stt=(ScriptedSTT(),))

    with pytest.raises(VoiceConfigurationError, match="unknown stt provider"):
        router.select_stt("does_not_exist")


def test_explicit_provider_is_never_replaced_by_another() -> None:
    second = ScriptedSTT(provider_id="second_stt")
    router = _router(stt=(UnavailableSTT(), second))

    # An explicit request for the broken provider must fail, not silently
    # hand over to another one.
    with pytest.raises(VoiceUnavailableError, match="микрофон"):
        router.select_stt("unavailable_stt")
    assert router.select_stt("second_stt").name == "second_stt"


def test_explicit_none_disables_the_side() -> None:
    router = _router(stt=(ScriptedSTT(),))

    with pytest.raises(VoiceUnavailableError, match="отключён"):
        router.select_stt(NONE)


# -- availability -----------------------------------------------------------


def test_unavailable_provider_is_skipped_with_its_own_reason() -> None:
    usable = ScriptedSTT(provider_id="usable_stt")
    router = _router(stt=(UnavailableSTT("микрофон отключён"), usable))

    selection = router.select_stt()

    assert selection.name == "usable_stt"
    assert ("unavailable_stt", "микрофон отключён") in selection.rejections


def test_nothing_usable_reports_every_reason() -> None:
    router = _router(stt=(UnavailableSTT("микрофон отключён"),))

    with pytest.raises(VoiceUnavailableError, match="микрофон отключён"):
        router.select_stt()


def test_unregistered_side_is_reported_clearly() -> None:
    router = _router()

    with pytest.raises(VoiceUnavailableError, match="не зарегистрирован"):
        router.select_tts()


# -- privacy ----------------------------------------------------------------


def test_local_only_excludes_remote_stt() -> None:
    # Remote is permitted by the process, refused by privacy: no hidden
    # fallback to the cloud even though it would technically work.
    router = _router(
        stt=(RemoteSTT(), ScriptedSTT()), privacy=LOCAL_ONLY, allow_remote=True
    )

    selection = router.select_stt()

    assert selection.name == "scripted_stt"
    assert "local_only" in dict(selection.rejections)["remote_stt"]


def test_local_only_without_a_local_provider_fails_instead_of_using_cloud() -> None:
    router = _router(stt=(RemoteSTT(),), privacy=LOCAL_ONLY, allow_remote=True)

    with pytest.raises(VoiceUnavailableError, match="local_only"):
        router.select_stt()


def test_remote_provider_is_used_only_when_explicitly_allowed() -> None:
    router = _router(stt=(RemoteSTT(),), privacy=ANY_PRIVACY, allow_remote=True)

    assert router.select_stt().name == "remote_stt"


def test_remote_blocked_even_with_permissive_privacy() -> None:
    router = _router(stt=(RemoteSTT(),), privacy=ANY_PRIVACY, allow_remote=False)

    with pytest.raises(VoiceUnavailableError, match="allow_remote"):
        router.select_stt()


def test_unknown_privacy_mode_is_rejected() -> None:
    with pytest.raises(VoiceConfigurationError, match="unknown voice privacy mode"):
        normalize_privacy("classified")


def test_privacy_defaults_to_local_only() -> None:
    assert normalize_privacy(None) == LOCAL_ONLY
    assert normalize_privacy("ANY") == ANY_PRIVACY


# -- reporting --------------------------------------------------------------


def test_explain_reports_selection_capabilities_and_rejections() -> None:
    router = _router(
        stt=(UnavailableSTT("нет микрофона"), ScriptedSTT()),
        tts=(ScriptedTTS(),),
        privacy=LOCAL_ONLY,
    )

    report = router.explain(probe=False)

    assert report["privacy"] == LOCAL_ONLY
    assert report["stt"]["selected"] == "scripted_stt"
    assert report["tts"]["selected"] == "scripted_tts"
    rejected = {item["provider"]: item["reason"] for item in report["stt"]["rejected"]}
    assert rejected["unavailable_stt"] == "нет микрофона"
    capabilities = report["stt"]["providers"][1]["capabilities"]
    assert capabilities["local"] is True
    assert capabilities["streaming"] is False


def test_explain_of_an_empty_router_says_why() -> None:
    router = _router()

    report = router.explain(probe=False)

    assert report["stt"]["selected"] is None
    assert "не зарегистрирован" in report["stt"]["reason"]


def test_auto_is_the_default_selection_name() -> None:
    router = _router(stt=(ScriptedSTT(),))

    assert router.select_stt(AUTO).explicit is False
    assert router.select_stt("scripted_stt").explicit is True


def test_explain_probes_each_provider_once_per_question() -> None:
    """A probe is a real system call - the report must not repeat it per field."""

    class CountingSTT(ScriptedSTT):
        def __init__(self) -> None:
            super().__init__()
            self.probes = 0

        def status(self, *, probe: bool = False):
            if probe:
                self.probes += 1
            return super().status(probe=probe)

    provider = CountingSTT()
    router = _router(stt=(provider,))

    router.explain(probe=True)

    # Once for the report, once for the selection the report shows - not four
    # times for four fields.
    assert provider.probes == 2
