"""Application-level tool behaviour: nothing runs unless asked, in writing."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from robanayt_ai.app import EXIT_OK, EXIT_TOOL_ERROR, main


def test_startup_registers_but_never_executes(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main([]) == EXIT_OK
    output = capsys.readouterr().out

    assert "tools:" in output
    assert "executed:   nothing" in output
    assert "list_tools" in output


def test_startup_makes_no_network_call(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def _record(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _record)

    assert main([]) == EXIT_OK
    assert calls == []


def test_tools_list_prints_registered_tools(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--tools-list"]) == EXIT_OK
    output = capsys.readouterr().out

    assert "current_time" in output
    assert "add_numbers" in output
    assert "permission=safe" in output


def test_tools_run_executes_a_safe_tool(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--tools-run", "add_numbers", "--tools-args", '{"a": 2, "b": 3}']) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "success"
    assert payload["output"]["sum"] == 5


def test_tools_run_rejects_bad_arguments(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["--tools-run", "add_numbers", "--tools-args", '{"a": "x"}'])

    assert code == EXIT_TOOL_ERROR
    assert "invalid" in json.loads(capsys.readouterr().out)["status"]


def test_tools_run_unknown_tool_fails_cleanly(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["--tools-run", "not_a_tool"])

    assert code == EXIT_TOOL_ERROR
    assert json.loads(capsys.readouterr().out)["status"] == "not_found"


def test_approval_required_tool_waits_for_the_flag(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    blocked = main(["--tools-run", "echo_text", "--tools-args", '{"text": "hi"}'])
    assert blocked == EXIT_TOOL_ERROR
    assert json.loads(capsys.readouterr().out)["status"] == "approval_required"

    capsys.readouterr()
    allowed = main(
        ["--tools-run", "echo_text", "--tools-args", '{"text": "hi"}', "--tools-approve"]
    )
    assert allowed == EXIT_OK
    assert json.loads(capsys.readouterr().out)["output"]["text"] == "hi"


def test_approval_is_bound_to_the_arguments(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(
        ["--tools-run", "echo_text", "--tools-args", '{"text": "a"}', "--tools-approve"]
    ) == EXIT_OK
    capsys.readouterr()

    # Same approval flag, different arguments: still one approval per request,
    # and the text never leaks into the console as an audit value.
    assert main(
        ["--tools-run", "echo_text", "--tools-args", '{"text": "b"}', "--tools-approve"]
    ) == EXIT_OK


def test_audit_is_empty_when_nothing_ran(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--tools-audit"]) == EXIT_OK
    assert "no executions in this run" in capsys.readouterr().out


def test_describe_prints_provider_neutral_schema(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--tools-describe"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert {item["name"] for item in payload} >= {"current_time", "add_numbers"}


def test_tools_do_not_touch_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def _record(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _record)

    assert main(["--tools-run", "add_numbers", "--tools-args", '{"a": 1, "b": 1}']) == EXIT_OK
    assert calls == []
