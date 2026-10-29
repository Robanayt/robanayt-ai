"""Voice in the real pipeline: the dialogue layer is not duplicated.

Every test here uses the genuine `DialogService`, tool system and memory layer
from Phase 3-5. Only the microphone/speaker boundary is scripted, because no
real audio backend exists in this environment.
"""

from __future__ import annotations

from pathlib import Path

from jarvis.core.config import ToolsConfig
from jarvis.orchestration.models import DialogStatus
from jarvis.tools import build_tool_system
from jarvis.voice.models import VoiceState
from jarvis.voice.router import LOCAL_ONLY, VoiceRouter
from jarvis.voice.session import VoiceSession
from tests.orchestration.dialog_fakes import (
    answer,
    make_dialog,
    register_tool,
    structured_call,
)
from tests.voice.fakes import ScriptedSTT, ScriptedTTS


def _voice(tmp_path: Path, replies: list, *, transcript: str = "запусти проверку"):
    dialog, provider, memory, tools, _clock = make_dialog(tmp_path, replies)
    router = VoiceRouter(
        stt=(ScriptedSTT(transcript),), tts=(ScriptedTTS(),), privacy=LOCAL_ONLY
    )
    session = VoiceSession(router, dialog=dialog)
    return dialog, provider, memory, tools, session


def test_transcript_travels_through_the_existing_dialogue_layer(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, session = _voice(
        tmp_path, [answer("готово")], transcript="привет джарвис"
    )

    turn = session.listen()

    assert turn.transcript == "привет джарвис"
    assert turn.answer == "готово"
    # One model request for one spoken phrase: no second AI pipeline.
    assert len(provider.requests) == 1
    sent = [message.content for message in provider.requests[0].messages]
    assert "привет джарвис" in sent
    dialog.close()


def test_voice_stores_no_more_history_than_a_typed_turn(tmp_path: Path) -> None:
    """Voice transcripts are not extracted into extra long-term memory."""
    typed_dialog, _typed_provider, typed_memory, _tools, _clock = make_dialog(
        tmp_path / "text", [answer("ответ")]
    )
    typed_result = typed_dialog.send("привет джарвис")
    typed_count = len(
        typed_memory.conversations.get_messages(typed_result.conversation_id or "x")
    )

    dialog, _provider, voice_memory, _tools2, session = _voice(
        tmp_path / "voice", [answer("ответ")], transcript="привет джарвис"
    )
    session.listen()
    conversations = voice_memory.conversations.list_conversations(limit=5)
    assert len(conversations) == 1
    voice_count = len(voice_memory.conversations.get_messages(conversations[0].id))

    assert voice_count == typed_count == 2
    typed_memory.close()
    voice_memory.close()


def test_a_spoken_tool_request_still_waits_for_approval(tmp_path: Path) -> None:
    """A built-in tool that always requires approval stays unanswered here."""
    # echo_text is a built-in tool whose policy always requires approval.
    dialog, provider, _memory, tools, session = _voice(
        tmp_path,
        [structured_call("echo_text", {"text": "привет"}), answer("не должно дойти")],
    )

    turn = session.listen()

    # "отправь..." said aloud is not permission: nothing ran, nothing moved.
    assert turn.pending is True
    assert turn.pending_tool == "echo_text"
    assert turn.pending_id
    assert turn.answer == ""
    # Nothing ran: the audit shows it stopped at approval, not an execution.
    records = tools.audit.records()
    assert len(records) == 1
    assert records[0].status == "approval_required"
    assert len(provider.requests) == 1
    dialog.close()


def test_spoken_request_for_an_unknown_tool_is_denied(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, session = _voice(
        tmp_path, [structured_call("ghost_tool")]
    )

    turn = session.listen()

    assert turn.pending is False
    assert turn.state == VoiceState.ERROR.value
    assert turn.answer == ""
    dialog.close()


def test_the_dialogue_status_is_preserved_for_voice(tmp_path: Path) -> None:
    from jarvis.ai.errors import ProviderUnavailableError

    dialog, _provider, _memory, _tools, session = _voice(
        tmp_path, [ProviderUnavailableError("модель недоступна")]
    )

    turn = session.listen()

    assert turn.state == VoiceState.ERROR.value
    assert turn.error
    dialog.close()


def test_one_tool_policy_for_spoken_and_typed_requests(tmp_path: Path) -> None:
    """Same registry, same decision: voice neither widens nor narrows it."""
    executed: list[dict] = []
    dialog, _provider, _memory, tools, session = _voice(
        tmp_path, [structured_call("harmless"), answer("выполнено")]
    )
    register_tool(
        tools,
        "harmless",
        lambda arguments, context: executed.append(dict(arguments)) or {"ok": True},
    )

    typed = dialog.send("выполни harmless")
    assert typed.status is DialogStatus.SUCCESS
    spoken_result = None

    dialog2, _provider2, _memory2, tools2, session2 = _voice(
        tmp_path / "second", [structured_call("harmless"), answer("выполнено")]
    )
    register_tool(
        tools2,
        "harmless",
        lambda arguments, context: executed.append(dict(arguments)) or {"ok": True},
    )
    turn = session2.listen()
    spoken_result = turn.answer

    assert spoken_result == "выполнено"
    assert len(executed) == 2  # once typed, once spoken - both through the policy
    assert [record.status for record in tools2.audit.records()] == ["success"]
    dialog.close()
    dialog2.close()
