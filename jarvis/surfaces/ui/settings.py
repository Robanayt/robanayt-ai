"""Settings window: only functions that really exist are offered.

Anything not implemented is shown as deferred instead of being faked.
Values come from the bridge (`info()` / `snapshot()`), never from guesses.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from jarvis import __phase__, __version__
from jarvis.surfaces.bridge import DialogBridge
from jarvis.surfaces.theme import Theme
from jarvis.surfaces.ui.widgets import apply_theme, label, panel

DEFERRED = "не реализовано (отложено)"


def open_settings(
    parent: tk.Misc,
    bridge: DialogBridge,
    theme: Theme,
    *,
    on_motion_change: Callable[[bool], None] | None = None,
) -> tk.Toplevel:
    """Open the settings window. Returns the created toplevel."""
    info = bridge.info()
    top = tk.Toplevel(parent)
    top.title("Настройки — Robanayt AI")
    top.configure(background=theme.palette.bg)
    top.geometry("720x560")
    top.transient(parent)

    apply_theme(top, theme)
    notebook = ttk.Notebook(top)
    notebook.pack(fill="both", expand=True, padx=12, pady=12)

    _appearance_tab(notebook, theme, on_motion_change)
    _ai_tab(notebook, info)
    _memory_tab(notebook, info)
    _tools_tab(notebook, info)
    _voice_tab(notebook, info)
    _privacy_tab(notebook)
    _diagnostics_tab(notebook, info)

    return top


def _tab(parent: ttk.Notebook, title: str) -> ttk.Frame:
    frame = ttk.Frame(parent, padding=14)
    parent.add(frame, text=title)
    return frame


def _row(frame: ttk.Frame, name: str, value: Any) -> None:
    row = ttk.Frame(frame)
    row.pack(fill="x", pady=3)
    label(row, f"{name}:", style_name="Dim.TLabel").pack(side="left", padx=(0, 6))
    label(row, str(value)).pack(side="left")


def _note(frame: ttk.Frame, text: str) -> None:
    label(frame, text, style_name="Dim.TLabel", wraplength=620).pack(anchor="w", pady=(8, 2))


def _voice_tab(parent: ttk.Notebook, info: dict[str, Any]) -> None:
    """Voice and audio: providers, availability and the honest reason."""
    frame = _tab(parent, "Голос")
    voice = info.get("voice") or {}
    if not voice or not voice.get("enabled", True):
        _row(frame, "Подсистема", "отключена")
        _note(frame, "Включить: [voice] enabled = true в config/config.toml.")
        return

    _row(frame, "Состояние", voice.get("state", "—"))
    _row(frame, "Приватность", voice.get("privacy", "—"))
    _row(
        frame,
        "Удалённые провайдеры",
        "разрешены" if voice.get("allow_remote") else "запрещены",
    )
    _row(frame, "Озвучивание ответов", "да" if voice.get("speak_responses") else "нет")
    _row(frame, "Таймаут прослушивания, с", voice.get("listen_timeout_seconds", "—"))

    for title, side in (("Распознавание речи (STT)", "stt"), ("Синтез речи (TTS)", "tts")):
        label(frame, title, style_name="Section.TLabel").pack(anchor="w", pady=(14, 4))
        block = voice.get(side) or {}
        selected = block.get("selected")
        _row(frame, "Активный провайдер", selected or "нет доступного")
        if selected:
            _row(frame, "Причина выбора", block.get("reason", "—"))
        for entry in block.get("providers", ()):
            state = "доступен" if entry.get("available") else "недоступен"
            label(
                frame,
                f"• {entry.get('provider')} [{entry.get('kind')}] — {state}"
                f" · {entry.get('message') or '—'}",
                wraplength=620,
            ).pack(anchor="w", pady=1)
            capabilities = entry.get("capabilities") or {}
            label(
                frame,
                "    streaming="
                f"{capabilities.get('streaming')}, local="
                f"{capabilities.get('local')}, cancel="
                f"{capabilities.get('cancel_supported')}, проверено на оборудовании: "
                f"{entry.get('tested_on_hardware')}",
                style_name="Dim.TLabel",
                wraplength=620,
            ).pack(anchor="w")
        for rejection in block.get("rejected", ()):
            label(
                frame,
                f"отклонён: {rejection.get('provider')} — {rejection.get('reason')}",
                style_name="Dim.TLabel",
                wraplength=620,
            ).pack(anchor="w")

    _note(
        frame,
        "Доступность решается пробой системы, а не наличием класса провайдера. "
        "Ничего не записывается и не воспроизводится само: прослушивание и "
        "озвучивание запускаются только действием пользователя.",
    )


def _appearance_tab(
    parent: ttk.Notebook,
    theme: Theme,
    on_motion_change: Callable[[bool], None] | None,
) -> None:
    frame = _tab(parent, "Внешний вид")
    label(frame, "Тема", style_name="Section.TLabel").pack(anchor="w", pady=(0, 6))
    _row(frame, "Активная тема", theme.name)
    _row(frame, "Акцент", theme.palette.accent)

    reduce = tk.BooleanVar(value=theme.motion.reduced)
    check = ttk.Checkbutton(
        frame,
        text="Уменьшить анимацию (меньше нагрузки на процессор)",
        variable=reduce,
        command=lambda: on_motion_change(reduce.get()) if on_motion_change else None,
    )
    check.pack(anchor="w", pady=(10, 4))
    _note(frame, "Прозрачность окна и эффекты свечения за пределами холста: " + DEFERRED + ".")


def _ai_tab(parent: ttk.Notebook, info: dict[str, Any]) -> None:
    frame = _tab(parent, "AI-провайдер")
    ai = info.get("ai") or {}
    _row(frame, "Сервис включён", ai.get("enabled", "—"))
    _row(frame, "Модель по умолчанию", ai.get("default_model") or "не задана")
    routing = ai.get("routing") or {}
    _row(frame, "Маршрутизация", routing.get("policy", "—"))
    _row(frame, "Как выбирается", routing.get("policy_note", "—"))
    _row(frame, "Удалённые вызовы", "разрешены" if routing.get("allow_remote") else "заблокированы")
    _row(frame, "Failover", "включён" if routing.get("allow_fallback") else "выключен")
    selected = routing.get("selected") or {}
    if selected:
        _row(frame, "Активный провайдер", f"{selected.get('provider')} / {selected.get('model') or 'по умолчанию'}")
        _row(frame, "Причина выбора", selected.get("reason", "—"))
    else:
        _row(frame, "Активный провайдер", "нет")
        _row(frame, "Причина", routing.get("reason", "—"))
    label(frame, "Провайдеры", style_name="Section.TLabel").pack(anchor="w", pady=(12, 6))
    for line in info.get("providers") or ():
        label(frame, f"• {line}").pack(anchor="w")
    for rejection in (routing.get("rejected") or ())[:4]:
        label(
            frame,
            f"отклонён: {rejection.get('provider')} — {rejection.get('reason')}",
            style_name="Panel.Dim.TLabel",
        ).pack(anchor="w")
    _note(
        frame,
        "Провайдеры настраиваются через config.toml или переменные окружения; "
        "сеть не используется при старте приложения.",
    )
    _note(frame, "Голос, анализ изображений и веб-поиск: " + DEFERRED + ".")


def _memory_tab(parent: ttk.Notebook, info: dict[str, Any]) -> None:
    frame = _tab(parent, "Память")
    memory = info.get("memory") or {}
    dialog = info.get("dialog") or {}
    _row(frame, "Память включена", memory.get("enabled", "—"))
    _row(frame, "База данных", memory.get("database", "—"))
    _row(frame, "История сохраняется", dialog.get("persist_history", "—"))
    _row(frame, "Сниппеты по запросу", dialog.get("memory_query", "только по явному запросу"))
    _note(frame, "Шифрование данных на диске не реализовано и не заявляется.")


def _tools_tab(parent: ttk.Notebook, info: dict[str, Any]) -> None:
    frame = _tab(parent, "Инструменты")
    tools = info.get("tools") or {}
    dialog = info.get("dialog") or {}
    _row(frame, "Подсистема включена", tools.get("enabled", "—"))
    _row(frame, "Лимит вызовов на сообщение", dialog.get("max_tool_calls_per_message", "—"))
    _row(frame, "Текстовый канал вызовов", dialog.get("allow_text_tool_calls", "—"))
    label(frame, "Зарегистрированные инструменты", style_name="Section.TLabel").pack(
        anchor="w", pady=(12, 6)
    )
    for name, permission in info.get("tools_list") or ():
        label(frame, f"• {name} — {permission}").pack(anchor="w")
    _note(
        frame,
        "Уровни разрешений задаются политикой и конфигурацией; подтверждение "
        "выдаётся только пользователем, модель не может одобрить собственный вызов.",
    )


def _privacy_tab(parent: ttk.Notebook) -> None:
    frame = _tab(parent, "Приватность")
    for line in (
        "• В логи идут идентификаторы, статусы и счётчики — не текст сообщений "
        "и не секреты.",
        "• Аргументы видны только в карточке подтверждения.",
        "• Ответ модели, содержимое памяти и результат инструмента считаются "
        "данными, а не инструкциями.",
        "• Данные не покидают машину, пока провайдер не настроен и не разрешён "
        "режимом запуска.",
    ):
        label(frame, line, wraplength=620).pack(anchor="w", pady=3)
    _note(frame, "Песочница и ограничение прав на уровне ОС: " + DEFERRED + ".")


def _diagnostics_tab(parent: ttk.Notebook, info: dict[str, Any]) -> None:
    frame = _tab(parent, "Диагностика")
    _row(frame, "Версия пакета", __version__)
    _row(frame, "Фаза", __phase__)
    _row(frame, "Python", sys.version.split()[0])
    _row(frame, "Режим запуска", info.get("mode", "normal"))
    _row(frame, "Интерфейс", "tkinter (стандартная библиотека)")
    _row(frame, "Ожидающих подтверждения", (info.get("dialog") or {}).get("pending", 0))
    _note(frame, "Полный прогон тестов выполняется командой pytest из каталога проекта.")
