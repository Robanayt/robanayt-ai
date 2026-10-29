"""JARVIS_AI application entry point.

Phase 5: the core layer, the provider-agnostic intelligence layer, the
local-first memory layer, the permissioned tool system and the dialogue
orchestration that connects them. Normal startup never contacts a provider,
never opens the memory database and never runs a tool - all of them are
lazily initialised. Network use, database creation, tool execution and
dialogue turns happen exclusively when the user explicitly asks for them
(`--ai-status` probes, `--ai-generate` sends, `--memory-*` uses the store,
`--tools-run` executes one tool, `--dialog-send` runs one bounded turn).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from robanayt_ai import __phase__, __version__
from robanayt_ai.ai import (
    AIService,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderRouter,
    RoutingPolicy,
    sanitize,
)
from robanayt_ai.ai.errors import AIError
from robanayt_ai.memory import MemoryService
from robanayt_ai.memory.errors import MemoryLayerError
from robanayt_ai.memory.service import safe_output_path
from robanayt_ai.orchestration import DialogResult, DialogService, DialogStatus
from robanayt_ai.core import (
    APP_RUNNING,
    APP_STARTING,
    APP_STOPPED,
    APP_STOPPING,
    Config,
    ConfigError,
    Event,
    EventBus,
    Lifecycle,
    RunMode,
    SingleInstanceLock,
    load_config,
    setup_logging,
    shutdown_logging,
)
from robanayt_ai.tools import ToolRequest, ToolSystem, build_tool_system
from robanayt_ai.voice import VoiceRouter, VoiceSession
from robanayt_ai.voice.providers import (
    SapiSpeechRecognizer,
    SapiSpeechSynthesizer,
    probe_audio_devices,
)
from robanayt_ai.tools.bridge import tool_call_from_ai, tool_descriptions
from robanayt_ai.tools.errors import ToolError
from robanayt_ai.surfaces import TK_AVAILABLE
from robanayt_ai.tools.models import ToolStatus

STATUS_NO_COMPONENTS = "no components registered"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ALREADY_RUNNING = 2
EXIT_CONFIG_ERROR = 3
EXIT_AI_ERROR = 4
EXIT_MEMORY_ERROR = 5
EXIT_TOOL_ERROR = 6
EXIT_DIALOG_ERROR = 7
EXIT_VOICE_ERROR = 8

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robanayt-ai", description="Robanayt AI assistant"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Robanayt AI {__version__} ({__phase__})",
    )
    parser.add_argument("--config", help="path to a TOML configuration file")
    parser.add_argument(
        "--run-mode",
        dest="run_mode",
        help="normal | safe | diagnostic | development",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        help="CRITICAL | ERROR | WARNING | INFO | DEBUG",
    )
    parser.add_argument(
        "--no-single-instance",
        action="store_true",
        help="skip the single-instance lock (diagnostics only)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="open the desktop interface (tkinter, standard library only)",
    )
    ai = parser.add_argument_group(
        "AI (network is used only when one of these is invoked)"
    )
    ai.add_argument(
        "--ai-status",
        action="store_true",
        help="query provider configuration/availability (probes the network)",
    )
    ai.add_argument(
        "--ai-generate",
        metavar="PROMPT",
        help="send one prompt through the AI service and print the answer",
    )
    ai.add_argument("--ai-provider", help="force a provider by name")
    ai.add_argument(
        "--ai-routing",
        action="store_true",
        help="explain the routing decision: active provider/model and why "
        "others were rejected (no network)",
    )
    ai.add_argument("--ai-model", help="force a model name")
    ai.add_argument("--ai-system", help="system prompt for --ai-generate")

    voice = parser.add_argument_group(
        "Voice (nothing is recorded or played unless one of these is invoked)"
    )
    voice.add_argument(
        "--voice",
        action="store_true",
        help="show the voice subsystem status and why each part is "
        "available or not (read-only: never records, never plays)",
    )
    voice.add_argument(
        "--voice-say",
        metavar="TEXT",
        help="speak one text through the text-to-speech provider (explicit action)",
    )
    voice.add_argument(
        "--voice-listen",
        action="store_true",
        help="listen once, transcribe, and run the transcript through the "
        "dialogue layer as ordinary text (explicit action)",
    )

    memory = parser.add_argument_group(
        "Memory (local database; created only when one of these is invoked)"
    )
    memory.add_argument("--memory-status", action="store_true", help="show memory statistics")
    memory.add_argument("--memory-list", action="store_true", help="list conversations (metadata only)")
    memory.add_argument("--memory-show", metavar="ID", help="print one conversation (explicit request)")
    memory.add_argument("--memory-export", metavar="ID", help="export a conversation to JSON")
    memory.add_argument("--memory-out", metavar="PATH", help="output path for --memory-export")
    memory.add_argument("--memory-search", metavar="QUERY", help="search messages and memories")
    memory.add_argument(
        "--memory-clear",
        action="store_true",
        help="delete every saved memory (requires --yes)",
    )
    memory.add_argument("--yes", action="store_true", help="confirm a destructive operation")

    tools = parser.add_argument_group(
        "Tools (nothing runs unless --tools-run is invoked; unknown tools are denied)"
    )
    tools.add_argument("--tools-list", action="store_true", help="list registered tools")
    tools.add_argument("--tools-describe", action="store_true", help="print tool descriptions for an AI request")
    tools.add_argument("--tools-run", metavar="NAME", help="execute one registered tool")
    tools.add_argument("--tools-args", metavar="JSON", help='arguments for --tools-run, e.g. \'{"a": 1}\'')
    tools.add_argument(
        "--tools-approve",
        action="store_true",
        help="approve this exact invocation (required for non-safe tools)",
    )
    tools.add_argument("--tools-audit", action="store_true", help="print the audit log of this run")

    dialog = parser.add_argument_group(
        "Dialog (one bounded turn per invocation; no autonomous loop)"
    )
    dialog.add_argument("--dialog-send", metavar="TEXT", help="send one message and print the answer")
    dialog.add_argument("--dialog-conversation", metavar="ID", help="continue an existing conversation")
    dialog.add_argument("--dialog-system", metavar="TEXT", help="system prompt for --dialog-send")
    dialog.add_argument(
        "--dialog-approve",
        metavar="PENDING_ID",
        help="approve a pending tool call and finish the turn",
    )
    dialog.add_argument("--dialog-pending", action="store_true", help="list calls waiting for approval")
    return parser


def build_memory_service(config: Config) -> MemoryService:
    """Compose the memory layer. Performs no I/O and opens no database."""
    return MemoryService(config.memory)


def build_tools(config: Config, *, allow_tools: bool = True) -> ToolSystem:
    """Compose the tool subsystem. Registers tools; executes nothing.

    `allow_tools=False` (safe/diagnostic mode) keeps the registry readable but
    disables the executor: nothing can run, whatever a request asks for.
    """
    tools_config = config.tools if allow_tools else replace(config.tools, enabled=False)
    return build_tool_system(tools_config)


def build_dialog(config: Config, ai: AIService, memory: MemoryService, tools: ToolSystem) -> DialogService:
    """Compose the dialogue layer. Performs no I/O and starts no request."""
    return DialogService(ai, memory, tools, config.orchestration)


def build_ai_service(config: Config, *, allow_network: bool = False) -> AIService:
    """Compose the intelligence layer. Performs no I/O.

    `allow_network` comes from the run-mode policy: when the mode forbids
    network access, remote providers stay unavailable even if the
    configuration enables them.
    """
    ai = config.ai if allow_network else replace(config.ai, allow_remote=False)
    ollama = OllamaProvider(
        base_url=ai.ollama_base_url,
        model=ai.resolve_ollama_model(),
        timeout=ai.request_timeout,
    )
    openai_compatible = OpenAICompatibleProvider(
        base_url=ai.openai_base_url,
        model=ai.resolve_openai_model(),
        api_key_env=ai.openai_api_key_env,
        organization=ai.openai_organization,
        allow_remote=ai.allow_remote,
        timeout=ai.request_timeout,
    )
    policy = RoutingPolicy.from_value(ai.routing_policy)
    router = ProviderRouter(
        [ollama, openai_compatible],
        default_provider=ai.default_provider,
        policy=policy,
        # The boundary reaches the router: a blocked remote provider is never
        # selected, not merely reported as unavailable.
        allow_remote=bool(ai.allow_remote),
    )
    return AIService(
        router,
        enabled=ai.enabled,
        default_model=ai.default_model or None,
        timeout=ai.request_timeout,
        max_retries=ai.max_retries,
        retry_delay_seconds=ai.retry_delay_seconds,
        # Failover is opt-in through the routing policy, never implicit.
        allow_fallback=policy is RoutingPolicy.FALLBACK,
    )


def build_voice(
    config: Config,
    *,
    allow_network: bool = False,
    dialog: DialogService | None = None,
    bus: Any | None = None,
    runner: Any | None = None,
) -> VoiceSession:
    """Compose the voice subsystem. Performs no I/O and probes nothing.

    Providers are only *objects* here: their real availability is decided by a
    probe inside the provider itself, never by the fact that the class exists.
    `runner` exists so diagnostics can be tested without spawning PowerShell.
    """
    voice = (
        config.voice if allow_network else replace(config.voice, allow_remote=False)
    )
    recognizer = SapiSpeechRecognizer(
        default_language=voice.language, runner=runner
    )
    synthesizer = SapiSpeechSynthesizer(
        voice=voice.voice,
        max_chars=voice.max_speech_chars,
        timeout_seconds=max(30.0, voice.listen_timeout_seconds * 4),
        runner=runner,
    )
    router = VoiceRouter(
        stt=(recognizer,),
        tts=(synthesizer,),
        privacy=voice.privacy,
        allow_remote=bool(voice.allow_remote),
    )
    return VoiceSession(
        router,
        dialog=dialog,
        bus=bus,
        enabled=voice.enabled,
        stt_provider=voice.stt_provider,
        tts_provider=voice.tts_provider,
        language=voice.language,
        listen_timeout_seconds=voice.listen_timeout_seconds,
        speak_responses=voice.speak_responses,
    )


def launch_ui(
    config: Config,
    mode: RunMode,
    service: AIService,
    memory: MemoryService,
    tools: ToolSystem,
    dialog: DialogService,
    bus: EventBus,
) -> int:
    """Open the desktop window. Returns a CLI exit code when it closes.

    The window owns no core object: it receives a `DialogBridge`, so every
    action goes through the same dialogue and permission paths as the CLI.
    """
    if not TK_AVAILABLE:
        print(
            "Robanayt AI - interface unavailable: tkinter is not available "
            "in this Python installation",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    from robanayt_ai.surfaces.bridge import DialogBridge
    from robanayt_ai.surfaces.ui import launch

    # The voice session probes nothing at construction: availability is decided
    # when the user actually asks to listen or to speak.
    voice_session = build_voice(
        config,
        allow_network=mode.policy.allow_network,
        dialog=dialog,
        bus=bus,
    )
    bridge = DialogBridge(
        dialog,
        ai=service,
        tools=tools,
        memory=memory,
        bus=bus,
        mode=mode,
        voice=voice_session,
        speak_responses=config.voice.speak_responses,
    )
    try:
        launch(bridge)
    except Exception as exc:  # noqa: BLE001 - never crash into a traceback
        print(f"Robanayt AI - interface error: {sanitize(str(exc))}", file=sys.stderr)
        logger.error("desktop interface failed: %s", sanitize(str(exc)), exc_info=exc)
        return EXIT_CONFIG_ERROR
    return EXIT_OK


def build_startup_message(
    config: Config | None = None,
    run_mode: RunMode = RunMode.NORMAL,
    service: AIService | None = None,
    memory: MemoryService | None = None,
    tools: ToolSystem | None = None,
    dialog: DialogService | None = None,
) -> str:
    """Return the human-readable startup banner."""
    config = config or Config()
    log_file = str(config.logging.log_path) if config.logging.to_file else "disabled"
    lines = [
        "Robanayt AI - starting",
        f"phase:    {__phase__}",
        f"version:  {__version__}",
        f"status:   {STATUS_NO_COMPONENTS}",
        f"run mode: {run_mode.value}",
        f"boundaries: network={'allowed' if run_mode.policy.allow_network else 'blocked'}"
        f", tools={'allowed' if run_mode.policy.allow_tools else 'blocked'}",
        f"config:   {config.source}",
        f"log:      {log_file}",
        "core:     config + event bus + lifecycle + logging",
    ]
    lines.extend(_ai_status_lines(config, service))
    lines.extend(_memory_status_lines(config, memory))
    lines.extend(_tools_status_lines(config, tools))
    lines.extend(_dialog_status_lines(config, dialog))
    lines.append(
        "note:     AI backend is configured but not contacted; tools are "
        "registered but not executed; audio, GUI and OS automation are not wired."
    )
    return "\n".join(lines)


def _ai_status_lines(config: Config, service: AIService | None) -> list[str]:
    if service is None or not service.enabled:
        return ["AI:", "  service disabled by configuration", "  network: not contacted"]

    routing = service.explain()
    lines = [
        "AI:",
        "  service initialized",
        f"  routing policy: {routing['policy']} ({routing['policy_note']})",
    ]
    selected = routing.get("selected")
    if selected:
        model = selected.get("model") or "default"
        lines.append(
            f"  active:   {selected['provider']} / model {model} [{selected['kind']}]"
        )
        lines.append(f"  why:      {selected['reason']}")
    else:
        lines.append("  active:   none")
        lines.append(f"  why:      {routing.get('reason', 'no provider available')}")
    for rejection in routing.get("rejected", ())[:4]:
        lines.append(f"  rejected: {rejection['provider']}: {rejection['reason']}")
    chain = routing.get("chain") or ()
    if chain:
        lines.append(f"  failover: {' -> '.join(chain)}")
    lines.append("  network: not contacted")
    return lines


def _memory_status_lines(config: Config, memory: MemoryService | None) -> list[str]:
    if memory is None or not memory.enabled:
        return ["memory:", "  disabled by configuration", "  database: not opened"]
    return [
        "memory:",
        "  service configured",
        f"  database: {memory.database_path}",
        "  database: not opened (lazy; created on first use)",
    ]


def _tools_status_lines(config: Config, tools: ToolSystem | None) -> list[str]:
    if tools is None or not config.tools.enabled:
        return ["tools:", "  disabled by configuration", "  executed: nothing"]
    return [
        "tools:",
        "  registry ready",
        f"  registered: {len(tools.registry)} tools ({', '.join(tools.registry.names())})",
        "  policy:     unknown tools denied, non-safe tools need approval",
        "  executed:   nothing (startup never runs a tool)",
    ]


def _dialog_status_lines(config: Config, dialog: DialogService | None) -> list[str]:
    if dialog is None or not config.orchestration.enabled:
        return ["dialog:", "  disabled by configuration", "  turns: nothing (startup sends nothing)"]
    return [
        "dialog:",
        "  orchestration ready",
        f"  tool call limit: {config.orchestration.max_tool_calls_per_message} per message",
        f"  history:        {'persisted' if config.orchestration.persist_history else 'not persisted'}",
        "  turns:          nothing (startup sends nothing)",
    ]


def send_dialog(dialog: DialogService, args: argparse.Namespace) -> int:
    """Run exactly one bounded dialogue turn and print the result."""
    try:
        result = dialog.send(
            args.dialog_send,
            conversation_id=args.dialog_conversation,
            system_prompt=args.dialog_system,
        )
    except ValueError as exc:
        print(f"Robanayt AI - dialog error: {exc}", file=sys.stderr)
        return EXIT_DIALOG_ERROR
    _print_dialog_result(result)
    return EXIT_OK if result.status is DialogStatus.SUCCESS else EXIT_DIALOG_ERROR


def approve_dialog_call(dialog: DialogService, pending_id: str) -> int:
    """Approve one pending tool call through the trusted CLI path."""
    result = dialog.approve_pending(pending_id)
    _print_dialog_result(result)
    return EXIT_OK if result.status is DialogStatus.SUCCESS else EXIT_DIALOG_ERROR


def list_pending_calls(dialog: DialogService) -> int:
    """List calls waiting for approval (no argument values)."""
    print("Robanayt AI - pending tool calls")
    pending = dialog.pending()
    if not pending:
        print("  (none)")
        return EXIT_OK
    print(json.dumps([item.describe() for item in pending], ensure_ascii=False, indent=2))
    return EXIT_OK


def _print_dialog_result(result: DialogResult) -> None:
    """Print a turn result: never a traceback, never a secret."""
    print("Robanayt AI - dialog")
    print(f"  status:       {result.status.value}")
    print(f"  conversation: {result.conversation_id or '-'}")
    print(f"  tool calls:   {result.tool_call_count}")
    print(f"  persisted:    {result.persisted}")
    if result.pending is not None:
        print("  pending:")
        print(
            json.dumps(
                result.pending.as_dict(include_arguments=True),
                ensure_ascii=False,
                indent=2,
            )
        )
    if result.answer:
        print(f"  answer: {result.answer}")
    if result.error:
        print(f"  error:  {result.error}")


def list_tools(tools: ToolSystem) -> int:
    """List registered tools - metadata only, nothing is executed."""
    print("Robanayt AI - registered tools")
    if not len(tools.registry):
        print("  (none)")
        return EXIT_OK
    for spec in tools.registry.specs():
        print(f"  {spec.name} v{spec.version}  permission={spec.permission.value}")
        print(f"      {spec.description}")
    return EXIT_OK


def describe_tools(tools: ToolSystem) -> int:
    """Print provider-neutral tool descriptions for an AI request."""
    print(json.dumps(list(tool_descriptions(tools.registry)), ensure_ascii=False, indent=2))
    return EXIT_OK


def show_tool_audit(tools: ToolSystem) -> int:
    """Print the audit records produced by this run (no argument values)."""
    print("Robanayt AI - tool audit")
    records = tools.audit.records() if tools.audit is not None else ()
    if not records:
        print("  (no executions in this run)")
        return EXIT_OK
    print(json.dumps([record.as_dict() for record in records], ensure_ascii=False, indent=2))
    return EXIT_OK


def run_tool(tools: ToolSystem, name: str, raw_arguments: str | None, approve: bool) -> int:
    """Execute exactly one tool through the executor."""
    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(arguments, dict):
            print("Robanayt AI - --tools-args must be a JSON object", file=sys.stderr)
            return EXIT_TOOL_ERROR
        request = tool_call_from_ai({"name": name, "arguments": arguments}, caller="cli")
    except (ValueError, ToolError) as exc:
        print(f"Robanayt AI - tool error: {exc}", file=sys.stderr)
        logger.error("tool request rejected: %s", exc)
        return EXIT_TOOL_ERROR

    approval = None
    if approve:
        tool = tools.registry.get(request.tool)
        if tool is None:
            print(f"Robanayt AI - unknown tool: {request.tool}", file=sys.stderr)
            return EXIT_TOOL_ERROR
        approval = tools.policy.grant(tool.spec, request.arguments_digest())

    result = tools.executor.execute(request, approval=approval)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    if result.status is not ToolStatus.SUCCESS:
        logger.error("tool %s -> %s: %s", result.tool, result.status.value, result.reason)
        return EXIT_TOOL_ERROR
    return EXIT_OK


def report_memory_status(memory: MemoryService) -> int:
    """Print memory statistics. Opens (and creates) the database on request."""
    print("Robanayt AI - memory status")
    try:
        stats = memory.stats()
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    print(f"  database:       {stats.database_path}")
    print(f"  schema version: {stats.schema_version}")
    print(f"  conversations:  {stats.conversations}")
    print(f"  messages:       {stats.messages}")
    print(f"  saved memories: {stats.memories}")
    print(f"  search backend: {stats.search_backend}")
    print(f"  size:           {stats.database_bytes} bytes")
    return EXIT_OK


def list_conversations(memory: MemoryService) -> int:
    """List conversations with metadata only - never prints message text."""
    print("Robanayt AI - conversations")
    try:
        conversations = memory.conversations.list_conversations()
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    if not conversations:
        print("  (none)")
        return EXIT_OK
    for conversation in conversations:
        title = conversation.title or "(untitled)"
        print(f"  {conversation.id}  {title}  messages={memory.conversations.count_messages(conversation.id)}")
    return EXIT_OK


def show_conversation(memory: MemoryService, conversation_id: str) -> int:
    """Print one conversation - only because the caller named it explicitly."""
    try:
        conversation = memory.conversations.require_conversation(conversation_id)
        messages = memory.conversations.get_messages(conversation_id)
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    print(f"Robanayt AI - conversation {conversation.id} ({conversation.title or 'untitled'})")
    if not messages:
        print("  (no messages)")
    for message in messages:
        print(f"  [{message.role}] {message.content}")
    return EXIT_OK


def export_conversation(memory: MemoryService, conversation_id: str, out: str | None) -> int:
    """Export one conversation to a validated JSON path."""
    try:
        target = Path(out) if out else Path(memory.config.export_directory) / f"{conversation_id}.json"
        safe_output_path(target)
        path = memory.export_conversation_to_file(conversation_id, target)
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    print(f"Robanayt AI - exported conversation {conversation_id} to {path}")
    return EXIT_OK


def search_memory(memory: MemoryService, query: str) -> int:
    """Search messages and memories locally; prints snippets only."""
    try:
        results = memory.recall(query)
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    print(f"Robanayt AI - search results for {query!r}")
    if not results:
        print("  (no matches)")
    for result in results:
        location = result.conversation_id or result.category or "-"
        print(f"  [{result.kind}] {result.id} ({location}) {result.snippet}")
    return EXIT_OK


def clear_memories(memory: MemoryService, confirmed: bool) -> int:
    """Delete every saved memory - requires explicit confirmation."""
    if not confirmed:
        print(
            "Robanayt AI - refusing to clear saved memories without --yes",
            file=sys.stderr,
        )
        return EXIT_MEMORY_ERROR
    try:
        removed = memory.memories.clear_all(confirm=True)
    except MemoryLayerError as exc:
        return _report_memory_error(exc)
    print(f"Robanayt AI - cleared {removed} saved memories")
    return EXIT_OK


def _report_memory_error(exc: MemoryLayerError) -> int:
    print(f"Robanayt AI - memory error: {exc}", file=sys.stderr)
    logger.error("memory operation failed: %s", exc)
    return EXIT_MEMORY_ERROR


def report_ai_status(service: AIService, *, probe: bool) -> int:
    """Print provider statuses. `probe=True` is the only network usage."""
    statuses = service.status(probe=probe)
    print("Robanayt AI - AI provider status")
    for status in statuses:
        state = "ready" if status.ok else "unavailable"
        print(
            f"  {status.provider}: configured={status.configured} "
            f"available={status.available} [{state}] - {status.message}"
        )
    # Only report network use when a provider actually answered a probe.
    contacted = any(status.checked_at is not None for status in statuses)
    print(f"  network: {'contacted (explicit probe)' if contacted else 'not contacted'}")
    return EXIT_OK


def report_ai_routing(service: AIService) -> int:
    """Explain the routing decision without contacting anything."""
    routing = service.explain()
    print("Robanayt AI - AI routing")
    print(f"  policy:   {routing['policy']} ({routing['policy_note']})")
    print(f"  remote:   {'allowed' if routing['allow_remote'] else 'blocked'}")
    print(f"  failover: {'on' if routing['allow_fallback'] else 'off'}")
    selected = routing.get("selected")
    if selected:
        print(f"  active:   {selected['provider']} / {selected['model'] or 'default'}")
        print(f"  why:      {selected['reason']}")
    else:
        print("  active:   none")
        print(f"  why:      {routing.get('reason', 'no provider available')}")
    for provider in routing.get("providers", ()):
        profile = provider["profile"]
        print(
            f"  provider: {provider['name']} [{provider['kind']}] "
            f"cost={profile['cost']} quality={profile['quality']} "
            f"latency={profile['latency_hint_ms']}"
        )
    for rejection in routing.get("rejected", ()):
        print(f"  rejected: {rejection['provider']}: {rejection['reason']}")
    chain = routing.get("chain") or ()
    if chain:
        print(f"  chain:    {' -> '.join(chain)}")
    print("  network: not contacted")
    return EXIT_OK


def run_generation(service: AIService, args: argparse.Namespace) -> int:
    """Send one explicit prompt. The only path that may call a provider."""
    try:
        response = service.generate(
            prompt=args.ai_generate,
            provider=args.ai_provider,
            model=args.ai_model,
            system=args.ai_system,
        )
    except AIError as exc:
        print(f"Robanayt AI - AI error: {sanitize(str(exc))}", file=sys.stderr)
        logger.error("AI request failed: %s", sanitize(str(exc)))
        return EXIT_AI_ERROR
    print(response.content)
    logger.info(
        "generation complete (provider=%s, model=%s, chars=%s)",
        response.provider,
        response.model,
        len(response.content),
    )
    return EXIT_OK


def report_voice_status(session: VoiceSession, *, probe: bool = True) -> int:
    """Voice diagnostics: what exists, what works, and why not. Read-only."""
    summary = session.explain(probe=probe)
    print("Robanayt AI - voice status")
    print(f"  subsystem:  {'enabled' if summary['enabled'] else 'disabled'}")
    print(
        f"  privacy:    {summary['privacy']} "
        f"({'удалённые разрешены' if summary['allow_remote'] else 'удалённые запрещены'})"
    )
    print(f"  state:      {summary['state']}")
    for side in ("stt", "tts"):
        block = summary[side]
        label = "speech-to-text" if side == "stt" else "text-to-speech"
        print(f"  {label} ({side}):")
        selected = block.get("selected")
        print(f"    selected:  {selected or 'нет доступного провайдера'}")
        if selected:
            print(f"    reason:    {block.get('reason', '')}")
        for entry in block.get("providers", ()):
            flag = "доступен" if entry["available"] else "недоступен"
            hardware = (
                "проверено на оборудовании"
                if entry["tested_on_hardware"]
                else "на оборудовании не проверялось"
            )
            capabilities = entry["capabilities"]
            print(
                f"    provider:  {entry['provider']} [{entry['kind']}] {flag} "
                f"({hardware})"
            )
            print(f"               причина: {entry['message'] or '—'}")
            print(
                "               возможности: streaming="
                f"{capabilities['streaming']}, local={capabilities['local']}, "
                f"cancel={capabilities['cancel_supported']}"
            )
        for rejection in block.get("rejected", ()):
            print(f"    rejected:  {rejection['provider']}: {rejection['reason']}")

    devices = probe_audio_devices()
    if devices:
        print(f"  devices:    {', '.join(devices)}")
    else:
        print("  devices:    список устройств недоступен (только чтение не удалось)")
    print("  hint:       listening and speaking are explicit actions only - "
          "nothing records or plays by itself")
    return EXIT_OK


def speak_text(session: VoiceSession, text: str) -> int:
    """Speak one explicit text. Returns EXIT_VOICE_ERROR when it cannot."""
    if session.speak(text):
        print("Robanayt AI - spoken successfully")
        return EXIT_OK
    message = session.last_error or "голос недоступен"
    print(f"Robanayt AI - voice error: {message}", file=sys.stderr)
    return EXIT_VOICE_ERROR


def listen_once(session: VoiceSession) -> int:
    """Listen once and run the transcript through the dialogue layer."""
    turn = session.listen()
    if turn.transcript_empty:
        print("Robanayt AI - речь не распознана: ход не отправлен")
        return EXIT_OK
    if turn.transcript:
        # Heard something: say what, before saying what happened next.
        print(f"распознано: {turn.transcript}")
    if turn.error:
        print(f"Robanayt AI - voice error: {turn.error}", file=sys.stderr)
        return EXIT_VOICE_ERROR
    if turn.pending:
        # A spoken request stops at the same gate a typed one does.
        print(f"требуется подтверждение: {turn.pending_tool} (id: {turn.pending_id})")
        print(f"одобрить: python -m robanayt_ai --dialog-approve {turn.pending_id}")
    if turn.answer:
        print(f"ответ: {turn.answer}")
    if turn.spoken:
        print("(ответ озвучен)")
    return EXIT_OK


def run(argv: Sequence[str] | None = None) -> int:
    """Run the application once and return the process exit code."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        mode = RunMode.from_value(args.run_mode or config.runtime.run_mode)
    except (ConfigError, ValueError) as exc:
        print(f"Robanayt AI - configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.log_level:
        level = args.log_level
    elif args.run_mode:
        level = mode.policy.log_level
    else:
        level = config.logging.level
    logging_config = (
        config.logging
        if level == config.logging.level
        else replace(config.logging, level=level)
    )

    setup_logging(logging_config)
    # Run-mode boundaries are applied here, once: configuration may enable a
    # capability, but the mode decides whether this process may use it.
    policy = mode.policy
    service = build_ai_service(config, allow_network=policy.allow_network)
    memory = build_memory_service(config)
    tools = build_tools(config, allow_tools=policy.allow_tools)
    dialog = build_dialog(config, service, memory, tools)
    logger.debug("AI service composed: %s", service.describe())
    logger.debug("memory service composed: %s", memory.describe())
    logger.debug("tool subsystem composed: %s", tools.describe())
    logger.debug("dialog service composed: %s", dialog.describe())

    if args.ai_status:
        return report_ai_status(service, probe=True)
    if args.ai_routing:
        return report_ai_routing(service)
    if args.ai_generate is not None:
        return run_generation(service, args)

    # Voice commands are explicit: only they touch the microphone or speakers.
    if args.voice or args.voice_say is not None or args.voice_listen:
        voice = build_voice(config, allow_network=policy.allow_network, dialog=dialog)
        try:
            if args.voice:
                return _finish(memory, report_voice_status(voice))
            if args.voice_say is not None:
                return _finish(memory, speak_text(voice, args.voice_say))
            return _finish(memory, listen_once(voice))
        finally:
            voice.close()

    # Memory commands are explicit: only they open/create the database.
    if args.memory_status:
        return _finish(memory, report_memory_status(memory))
    if args.memory_list:
        return _finish(memory, list_conversations(memory))
    if args.memory_show is not None:
        return _finish(memory, show_conversation(memory, args.memory_show))
    if args.memory_export is not None:
        return _finish(
            memory, export_conversation(memory, args.memory_export, args.memory_out)
        )
    if args.memory_search is not None:
        return _finish(memory, search_memory(memory, args.memory_search))
    if args.memory_clear:
        return _finish(memory, clear_memories(memory, args.yes))

    # Tool commands are explicit: only --tools-run executes anything.
    if args.tools_list:
        return _finish(memory, list_tools(tools))
    if args.tools_describe:
        return _finish(memory, describe_tools(tools))
    if args.tools_audit:
        return _finish(memory, show_tool_audit(tools))
    if args.tools_run is not None:
        return _finish(
            memory, run_tool(tools, args.tools_run, args.tools_args, args.tools_approve)
        )

    # Dialogue commands: one bounded turn per invocation, never a loop.
    if args.dialog_send is not None:
        return _finish(memory, send_dialog(dialog, args))
    if args.dialog_approve is not None:
        return _finish(memory, approve_dialog_call(dialog, args.dialog_approve))
    if args.dialog_pending:
        return _finish(memory, list_pending_calls(dialog))

    bus = EventBus(
        record_failures=config.bus.record_failures,
        max_recorded_failures=config.bus.max_recorded_failures,
    )
    lifecycle = Lifecycle()
    lock = SingleInstanceLock(config.runtime.lock_path)

    try:
        lifecycle.start()
        bus.publish(Event(APP_STARTING, {"run_mode": mode.value, "config": config.source}))

        if config.runtime.single_instance and not args.no_single_instance:
            if not lock.acquire():
                message = (
                    f"Robanayt AI is already running (lock: {lock.path}); "
                    "this instance exits without touching the other one"
                )
                print(message, file=sys.stderr)
                logger.error(message)
                lifecycle.fail("another instance is already running")
                lifecycle.shutdown_cleanly()
                return EXIT_ALREADY_RUNNING
            logger.debug("single-instance lock acquired: %s", lock.path)

        lifecycle.mark_running()
        bus.publish(Event(APP_RUNNING, mode.describe()))
        logger.info(
            "core ready (mode=%s, config=%s, ai=%s, components=none)",
            mode.value,
            config.source,
            config.ai.default_provider,
        )
        if args.ui:
            return launch_ui(config, mode, service, memory, tools, dialog, bus)

        print(build_startup_message(config, mode, service, memory, tools, dialog))
        return EXIT_OK
    finally:
        dialog.close()
        tools.close()
        memory.close()
        bus.publish(Event(APP_STOPPING, {"state": lifecycle.state.value}))
        lifecycle.shutdown_cleanly()
        bus.publish(
            Event(
                APP_STOPPED,
                {"state": lifecycle.state.value, "bus_failures": len(bus.failures)},
            )
        )
        lock.release()
        shutdown_logging()


def _finish(memory: MemoryService, code: int) -> int:
    """Release the database connection before returning an exit code."""
    memory.close()
    shutdown_logging()
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by `python -m robanayt_ai` and the `robanayt_ai` script."""
    return run(argv)


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    sys.exit(main())
