# Robanayt AI

*Пользовательское название проекта — **Robanayt AI**. Каталог исходников и имя
пакета — `robanayt_ai`. Технические имена данных (`jarvis.db`, `jarvis.log`,
`jarvis.lock`) и переменные окружения `JARVIS_*` сохранены для совместимости.*

Локальный голосовой ассистент. Проект находится на **Фазе 8** — реализованы слой
ядра (конфигурация, шина событий, жизненный цикл, логирование, режимы запуска,
защита от второго экземпляра), слой интеллекта (provider-agnostic AI-ядро с
роутером провайдеров, `AIService`, контекстом и стримингом), слой памяти
(SQLite: диалоги, сообщения, явные долговременные воспоминания, локальный поиск),
система инструментов с разрешениями (реестр, политика, исполнитель, аудит),
оркестрация диалога, которая соединяет их в одном ограниченном ходе,
голографический интерфейс (Фаза 6), гибридная маршрутизация AI (Фаза 7) и
голосовой слой (Фаза 8).

Старт приложения **не обращается к сети**, **не создаёт базу данных** и **не
запускает инструменты**: сеть используется только при явном вызове
`--ai-status` / `--ai-generate`, БД создаётся лениво, инструмент выполняется
только по `--tools-run` и только после проверки политики, а ход диалога —
только по `--dialog-send`.

Что не реализовано: зрение, управление ОС и автономные агенты. Реальное
распознавание речи — **BLOCKED окружением**: слой готов, но в системе нет ни
одного распознавателя (`InstalledRecognizers=0`); озвучивание работает
(`windows_sapi_tts`, найдено голосов: 2).

Корень проекта — эта папка (`.../New Jarvis AI`). Вложенной папки `JARVIS_AI` нет
и создавать её не нужно.

## Требования

- Python 3.10+ (проверено на 3.14.10)
- Внешних зависимостей у рантайма нет; для тестов нужен `pytest`

## Запуск

```bash
python -m robanayt_ai
```

Ожидаемый вывод:

```
Robanayt AI - starting
phase:    phase-8
version:  0.5.0
status:   no components registered
run mode: normal
config:   defaults
log:      runtime\logs\jarvis.log
core:     config + event bus + lifecycle + logging
AI:
  service initialized
  provider configured: ollama
  network: not contacted
memory:
  service configured
  database: runtime\memory\jarvis.db
  database: not opened (lazy; created on first use)
tools:
  registry ready
  registered: 4 tools (add_numbers, current_time, echo_text, list_tools)
  policy:     unknown tools denied, non-safe tools need approval
  executed:   nothing (startup never runs a tool)
dialog:
  orchestration ready
  tool call limit: 3 per message
  history:        persisted
  turns:          nothing (startup sends nothing)
note:     AI backend is configured but not contacted; tools are registered but not executed; audio, GUI and OS automation are not wired.
```

Блок выше — ожидаемый вывод сборки: баннер печатает название продукта
`Robanayt AI` и его метаданные (`phase-8`). Способ запуска —
`python -m robanayt_ai`; переменные окружения по-прежнему имеют префикс
`JARVIS_*`, а файлы данных называются `jarvis.db` / `jarvis.log`.

Полезные варианты запуска:

```bash
python -m robanayt_ai --run-mode diagnostic   # DEBUG-логи, без инструментов
python -m robanayt_ai --run-mode safe         # инструменты отключены
python -m robanayt_ai --no-single-instance    # пропустить проверку второго экземпляра
python -m robanayt_ai --version
```

Коды выхода: `0` — норма, `2` — уже запущен другой экземпляр,
`3` — ошибка конфигурации, `4` — ошибка AI, `5` — ошибка памяти,
`6` — инструмент не выполнен, `7` — ход диалога не завершён успешно.

## Тесты

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Проверено 2026-10-29 в `.venv` (Python 3.14.10): **732 passed / 0 failed /
1 skipped**, ~6 s, без сети и без реальных моделей. Единственный пропуск —
live-проверка Ollama (модель не запущена): `skipped` ≠ `passed`.

## Интерфейс (Фаза 6 — ACCEPTED / DONE, принято 2026-10-29)

## Голос (Фаза 8: слой DONE, реальный STT — BLOCKED)

```
микрофон → STT-провайдер → transcript → DialogService → ответ → TTS-провайдер
```

Голос — это **ввод и вывод**, а не второй AI-конвейер: transcript попадает ровно
в тот же `DialogService`, что и набранный текст (те же инструменты, те же
подтверждения, те же отказы). Voice-роутер отделён от AI-роутера Phase 7:
цена/качество модели не влияют на выбор ушей и голоса.

```bash
.venv\Scripts\python -m robanayt_ai --voice          # статус: без записи и без БД
.venv\Scripts\python -m robanayt_ai --voice-listen   # явно: послушать и отправить transcript в диалог
.venv\Scripts\python -m robanayt_ai --voice-say "привет"   # явно: произнести текст
```

Настройки (`[voice]`, env `JARVIS_VOICE_*`): `enabled`, `stt_provider`,
`tts_provider` (`auto` | имя | `none`), `voice`, `language`, `privacy`
(`local_only` по умолчанию), `listen_timeout_seconds`, `speak_responses`
(false), `max_speech_chars`, `allow_remote` (false).

Что правда, а что нет:

- **TTS доступен**: Windows SAPI, в этой системе найдено 2 голоса; синтез
  проверен рендером в память (WAV, без вывода на динамики).
- **STT недоступен**: движок есть, но ни один распознаватель речи не установлен
  — `InstalledRecognizers=0`; `--voice` показывает это как причину, а доступ
  даёт `--voice-listen` отказать честно, а не «тихо слушать».
- Класс провайдера ≠ доступность: `status(probe=False)` отвечает «не проверено»,
  проба выполняется явно (`--voice`, настройки UI, перед действием).
- Отмена кооперативная: между шагами и завершением своего процесса-помощника.
  Уже произнесённое не отменяется; аппаратного «мгновенного прерывания» не обещаем.
- Приватность: audio не сохраняется, не логируется и не уходит по сети;
  `local_only` запрещает remote-провайдеры без скрытого fallback; secrets не логируются.
- Никакого автоматического прослушивания/озвучивания — только по явному действию.

```bash
.venv\Scripts\python -m robanayt_ai --ui
```

Окно строится на `tkinter` (стандартная библиотека) — новых зависимостей нет.
Интерфейс не владеет ядром: он работает только через `robanayt_ai.surfaces.bridge`
(`send`, `approve`, `cancel`, `snapshot`), поэтому диалог, память, инструменты
и разрешения идут тем же путём, что и в CLI.

Что работает:

- центральное ядро с реальными состояниями: `idle`, `processing`, `tool`,
  `approval_required`, `answering`, `error`, `cancelled`;
- транскрипт, ввод с отправкой по Enter;
- карточка подтверждения с аргументами вызова (подтверждение выдаёт только
  ядро, модель не подтверждает свои действия);
- правая панель: провайдеры, инструменты, ожидающие подтверждения;
- **отмена хода** — кооперативная: ход останавливается между шагами; уже
  запущенный инструмент не прерывается, и результат честно перечисляет
  выполненные вызовы;
- **экспорт диалога** в JSON через существующий `MemoryService` (в фоновом
  потоке, ошибка показывается явно);
- **история в левой панели** — постранично (`HISTORY_PAGE_SIZE = 20`),
  кнопка «Показать ещё», выбор разговора;
- окно настроек: внешний вид, AI, память, инструменты, приватность,
  диагностика.

Интерфейс не блокируется во время работы ядра: ход выполняется в отдельном
потоке, результаты приходят через очередь.

### Оговорки приёмки (не снимаются принятием Фазы 6)

- **Live-тест модели — BLOCKED.** `tests/ai/test_live_ollama.py` пропускается:
  Ollama установлен, но не запущен. `1 skipped` в отчёте pytest — это **не**
  пройденный live-тест. Ничего не устанавливалось и не запускалось.
- **`pyproject.toml` приведён к пакету 2026-10-29** (шаг 4 Git-аудита): версия
  `0.1.0` → `0.5.0`, описание «Phase 0 skeleton» → «Phase 8». Техническое имя
  `name = "robanayt_ai"` и `requires-python >=3.10` не менялись.
- **Косметические эффекты вне canvas — deferred** (прозрачность окна и
  эффекты вне холста); помечены в интерфейсе как «не реализовано (отложено)».
- **Отмена хода кооперативная:** останавливает ход между шагами и не прерывает
  уже запущенный инструмент; результат перечисляет выполненные вызовы.
- **Тяжёлая параллельная запись SQLite — неподдерживаемый сценарий:**
  соединение на поток, файловую конкурентность обеспечивает SQLite (WAL,
  один писатель). Поддерживается один ход за раз.

Что отложено и помечено в интерфейсе: голос, зрение, автоматизация Windows,
веб-исследования, прозрачность окна.

Если `tkinter` недоступен, приложение печатает понятное сообщение и завершается
с кодом `3` — окно не подменяется заглушкой.

## AI-слой (Фаза 2 + Фаза 7: гибридная маршрутизация)

```
robanayt_ai/ai/
    models.py       AIMessage, AIRequest, AIResponse, AIChunk, ProviderStatus
    errors.py       AIError и типы ошибок провайдеров + санитайзеры секретов
    http.py         транспорт на urllib (stdlib), инжектируется в тестах
    provider.py     протокол Provider + ProviderProfile (цена/качество/задержка)
    router.py       ProviderRouter, RoutingPolicy, explain()
    service.py      AIService — единая точка входа для остального JARVIS
    context.py      ConversationContext (system + ограниченная история)
    streaming.py    накопление AIChunk, фолбэк без стриминга
    providers/
        ollama.py              локальный Ollama (/api/chat, /api/tags)
        openai_compatible.py   любой OpenAI-style /chat/completions
```

Пример использования:

```python
from robanayt_ai.ai import AIService, OllamaProvider, ProviderRouter

service = AIService(ProviderRouter([OllamaProvider(model="qwen3.5:4b")]))
response = service.generate(prompt="Привет, JARVIS")
print(response.provider, response.content)
```

### Маршрутизация (Фаза 7)

Провайдер выбирается политикой `ai.routing_policy`
(`JARVIS_AI_ROUTING_POLICY`):

| Политика | Поведение |
|---|---|
| `local_first` (по умолчанию) | локальные провайдеры первыми, затем удалённые |
| `cloud_first` | удалённые первыми, затем локальные |
| `cheapest` | по **объявленной** цене профиля, при равенстве — локальный |
| `strongest` | по **объявленному** уровню качества; не объявившие — последними |
| `fastest` | по **объявленным** подсказкам задержки; без подсказок — деградация до `local_first`, причина так и говорит |
| `fallback` | локальный первый + цепочка failover (каждый альтернативный провайдер — один раз, только для повторяемых ошибок) |

Приватность: `privacy="local_only"` (или `metadata["privacy"]`) исключает
удалённые провайдеры; в диалоге — `DialogService.send(..., privacy="local_only")`.
Граница `allow_remote=false` (в том числе из режима запуска) **не позволяет
выбрать** удалённый провайдер, а не только помечает его недоступным.

Увидеть решение без обращения к сети:

```bash
.venv\Scripts\python -m robanayt_ai --ai-routing
```

Причины выбора и отказа видны также в стартовом баннере и во вкладке
«AI-провайдер» окна настроек. Ничего не измеряется: цена, качество и задержка —
**объявленные** свойства провайдера (`ProviderProfile`), а не телеметрия.

## Память (Фаза 3)

```
robanayt_ai/memory/
    models.py              Conversation, Message, SavedMemory, SearchResult,
                           MemoryStats, PreparedContext + валидация
    errors.py              типы ошибок (базовый класс — не MemoryError)
    database.py            sqlite3: ленивое открытие, версия схемы, миграции
    conversation_store.py  диалоги/сообщения, пагинация, экспорт/импорт
    memory_store.py        явные долговременные воспоминания
    search.py              FTS5 (bm25, сниппеты) + LIKE-фолбэк
    service.py             MemoryService: контекст, run_turn, экспорт/импорт
```

База — обычный SQLite-файл `runtime/memory/jarvis.db` (схема версии 2,
миграции инкрементальные). Старт приложения базу **не** создаёт и не мигрирует.

```python
from robanayt_ai.core.config import MemoryConfig
from robanayt_ai.memory.service import MemoryService

service = MemoryService(MemoryConfig(directory="runtime/memory"))
conversation = service.create_conversation("утро")
service.add_message(conversation.id, "user", "запомни: встреча в 10:00")
service.remember("встречи начинаются вовремя", category="preferences")

context = service.prepare_context(conversation.id, system_prompt="Ты — JARVIS")
request = context.to_ai_request(model="qwen3.5:4b")   # совместимо с Фазой 2

for result in service.recall("встреча"):
    print(result.kind, result.snippet)
service.close()
```

Автоматического запоминания нет: воспоминание создаётся только явным `save()`
/`remember()`. Извлечённое из памяти считается данными, а не инструкциями.

## Инструменты (Фаза 4)

```
robanayt_ai/tools/
    models.py       ToolSpec, ArgumentSpec, ToolRequest, ToolResult, ToolStatus,
                    PermissionLevel, ApprovalToken, AuditRecord
    errors.py       ToolError и типы ошибок
    registry.py     реестр: уникальность имён, явная замена и удаление
    policy.py       политика и подтверждения, привязанные к аргументам
    executor.py     единственный путь выполнения: валидация → политика → запуск
    audit.py        аудит без значений аргументов и без секретов
    builtins.py     current_time, add_numbers, echo_text, list_tools
    bridge.py       описания для AI и разбор предложенного вызова
    system.py       build_tool_system()
```

Жизненный цикл одного вызова:

```
ToolRequest -> реестр -> валидация аргументов -> политика -> запуск с дедлайном
           -> ToolResult (success | failed | timeout | denied |
                          approval_required | invalid | not_found)
           -> запись аудита
```

Пример:

```python
from robanayt_ai.core.config import ToolsConfig
from robanayt_ai.tools import ToolRequest, build_tool_system

system = build_tool_system(ToolsConfig())

safe = system.executor.execute(
    ToolRequest(tool="add_numbers", arguments={"a": 2, "b": 40})
)
print(safe.status, safe.output)          # success {'sum': 42.0, ...}

request = ToolRequest(tool="echo_text", arguments={"text": "hi"})
print(system.executor.execute(request).status)          # approval_required

approval = system.policy.grant(
    system.registry.require("echo_text").spec, request.arguments_digest()
)
print(system.executor.execute(request, approval=approval).status)   # success
```

Демонстрационные инструменты не имеют внешних побочных эффектов: чтение часов,
сложение чисел, чтение реестра и возврат переданного текста. Произвольного
выполнения Python или командной строки, удаления файлов, управления браузером
и системными настройками в этой фазе нет.

## Диалог (Фаза 5)

```
robanayt_ai/orchestration/
    models.py       DialogStatus, DialogResult, ToolCallRecord, PendingApproval
    errors.py       ошибки слоя оркестрации
    protocol.py     два явных канала вызова инструмента
    prompts.py      описание инструментов для системной подсказки
    service.py      DialogService: ограниченный ход, подтверждение, сохранение
```

Один ход — одна цепочка:

```
send(message) -> диалог и контекст -> AIService -> (вызов инструмента через
ToolExecutor с проверкой политики) -> результат модели -> сохранение
```

Ограничения: максимум 3 вызова инструмента на сообщение (0..10), не более
`limit + 1` запросов к модели, остановка при `tool_limit_reached`,
отсутствие автоматических повторов. Вызов инструмента приходит либо
структурированно (`AIResponse.metadata["tool_call"]`), либо — если включён
`allow_text_tool_calls` — единственным блоком ```` ```json ```` с ключом
`tool_call`; проза вызовом не считается.

Структурированный канал заполняется нативно: провайдеры Ollama и
OpenAI-compatible отправляют описания инструментов в поле `tools`, а своё
поле `tool_calls` приводят к `metadata["tool_call"]`
(`robanayt_ai.ai.tool_calls`). Из ответа провайдера берутся только `name` и
`arguments`: идентификаторы, `approved`, `permission` и любые другие поля
отбрасываются, поэтому модель не может передать разрешение через payload.
Невалидные аргументы, неподдерживаемый тип вызова и более одного вызова в
ответе отклоняются — инструмент не запускается, обычный текстовый ответ
сохраняется. Потоковая выдача (`stream()`) собирает только текст, вызовы
инструментов в ней не агрегируются.

```python
from robanayt_ai.core.config import MemoryConfig, OrchestrationConfig, ToolsConfig
from robanayt_ai.memory import MemoryService
from robanayt_ai.orchestration import DialogService
from robanayt_ai.tools import build_tool_system

dialog = DialogService(
    ai_service,                       # AIService из Фазы 2
    MemoryService(MemoryConfig()),
    build_tool_system(ToolsConfig()),
    OrchestrationConfig(),
)

result = dialog.send("Сколько будет 2+40?")
print(result.status.value, result.answer, result.tool_call_count)

if result.status.value == "approval_required":
    print(result.pending.tool, result.pending.arguments)   # для согласия
    result = dialog.approve_pending(result.pending.pending_id)
```

Статусы: `success`, `approval_required`, `tool_limit_reached`,
`invalid_tool_call`, `ai_error`, `memory_error`, `tool_error`,
`empty_response`, `pending_expired`, `disabled`. Успехом считается только
`success`; остальные статусы объясняют причину и не выдают пустой ответ
за результат.

## Структура (Фаза 8)

```
pyproject.toml      манифест, конфиг pytest, точка входа robanayt-ai
README.md           этот файл
PLAN.md             план фаз и переноса ядра
STATUS.md           снимок текущего состояния
docs/               отчёты сессий (docs/night/) и архитектурные предложения
                    (docs/architecture/)
.gitignore
.gitattributes     переносы строк: LF в репозитории, CRLF для .ps1/.bat
robanayt_ai/
    __init__.py     метаданные пакета (0.5.0, phase-8)
    __main__.py     python -m robanayt_ai
    app.py          CLI, композиция ядра, AI-слоя, памяти и инструментов
    core/
        config.py          типизированный конфиг: defaults + TOML + env
        bus.py             шина событий (publish / publish_async)
        lifecycle.py       состояния NEW/STARTING/RUNNING/STOPPING/STOPPED/FAILED
        log_config.py      логирование на stdlib (консоль + файл с ротацией)
        run_mode.py        normal / safe / diagnostic / development
        single_instance.py блокировка файла с PID, безопасный отказ
    ai/
        models.py, errors.py, http.py, provider.py, router.py, service.py,
        context.py, streaming.py, tool_calls.py,
        providers/{ollama,openai_compatible}.py
    memory/
        models.py, errors.py, database.py, conversation_store.py,
        memory_store.py, search.py, service.py
    tools/
        models.py, errors.py, registry.py, policy.py, executor.py,
        audit.py, builtins.py, bridge.py, system.py
    orchestration/
        models.py, errors.py, protocol.py, prompts.py, service.py
    surfaces/
        protocol.py, bridge.py, theme.py     слой интерфейса (зависит от ядра)
        ui/
            window.py      главное окно
            core_widget.py центральное ядро и его состояния
            approval.py    карточка подтверждения вызова
            settings.py    окно настроек
            widgets.py     тематические элементы управления
    voice/
        errors.py, models.py, provider.py, router.py, session.py
        providers/{powershell,sapi}.py   PowerShell-хост и Windows SAPI
tests/
    surfaces/      тесты моста, моделей представлений, smoke-тесты окна
    test_smoke.py         старт, сообщение, код выхода
    test_app.py           интеграция: режимы, лог, второй экземпляр
    test_ai_app.py        AI-команды CLI без сети
    test_memory_app.py    память: БД создаётся только по явной команде
    test_tools_app.py     инструменты: реестр, запуск, аудит
    test_dialog_app.py    диалог: один ход, pending, approve, коды выхода
    test_voice_app.py     голос: статус, озвучивание, прослушивание
    core/           тесты шины, конфига, lifecycle, логов, режимов, блокировки
    ai/             тесты AI-слоя (фейковый транспорт, без сети)
    memory/         тесты памяти (временные SQLite-базы)
    tools/          тесты инструментов (фейк-часы, фейк-инструменты)
    orchestration/  тесты диалога (фейк-провайдер, временные БД)
    voice/          тесты голоса (фейк-провайдеры, PowerShell-хост)
```

## Конфигурация

Значения по умолчанию подходят для старта, файл конфигурации не обязателен.
При необходимости — `config/config.toml`:

```toml
[logging]
level = "INFO"          # CRITICAL|ERROR|WARNING|INFO|DEBUG
directory = "runtime/logs"

[runtime]
run_mode = "normal"     # normal|safe|diagnostic|development
single_instance = true
```

Переменные окружения: `JARVIS_RUN_MODE`, `JARVIS_LOG_LEVEL`,
`JARVIS_SINGLE_INSTANCE`, `JARVIS_LOG_TO_FILE`, `JARVIS_LOCK_PATH`.

### AI (`[ai]`)

```toml
[ai]
enabled = true
default_provider = "ollama"     # или "openai_compatible"
default_model = ""              # модель не навязывается и не скачивается
routing_policy = "local_first"
request_timeout = 30.0
max_retries = 0
ollama_base_url = "http://127.0.0.1:11434"
openai_base_url = ""            # пусто => провайдер не настроен
openai_api_key_env = "JARVIS_OPENAI_API_KEY"
allow_remote = false            # удалённые вызовы выключены по умолчанию
```

Ключ API **не хранится** в конфиге — только имя переменной окружения
(`openai_api_key_env`). Значение читается провайдером в момент запроса и не
попадает в логи, ошибки, статусы или `describe()`.

Явные AI-операции (единственные, кто использует сеть):

```bash
python -m robanayt_ai --ai-status                  # конфигурация/доступность провайдеров
python -m robanayt_ai --ai-generate "Привет"       # один запрос, ответ в stdout
```

Без модели `ollama` сообщает `configured=False`, а `--ai-generate` завершается
с кодом `4` и понятным сообщением — приложение не падает.

### Память (`[memory]`)

```toml
[memory]
enabled = true
directory = "runtime/memory"
filename = "jarvis.db"
database_timeout = 5.0
max_recent_messages = 20
default_search_limit = 20
max_context_chars = 8000
export_directory = "runtime/exports"
```

Env: `JARVIS_MEMORY_ENABLED`, `JARVIS_MEMORY_DIR`. База не обязана существовать
на старте; при `enabled = false` обычный старт работает, а вызовы API памяти
дают явную ошибку `MemoryDisabledError`.

Команды памяти (каждая — явный вызов; БД создаётся только здесь):

```bash
python -m robanayt_ai --memory-status               # схема, счётчики, бэкенд поиска
python -m robanayt_ai --memory-list                 # только метаданные диалогов
python -m robanayt_ai --memory-show ID              # содержимое по явному запросу
python -m robanayt_ai --memory-search "запрос"      # сниппеты, локально
python -m robanayt_ai --memory-export ID --memory-out path.json
python -m robanayt_ai --memory-clear --yes          # только с подтверждением
```

`--memory-clear` без `--yes` завершается кодом `5` и ничего не удаляет.

### Инструменты (`[tools]`)

```toml
[tools]
enabled = true
default_timeout_seconds = 5.0
approval_ttl_seconds = 300.0
audit_enabled = true
audit_to_file = false          # файл аудита не создаётся при старте
audit_directory = "runtime/logs"
audit_filename = "tools.jsonl"
```

Env: `JARVIS_TOOLS_ENABLED`, `JARVIS_TOOLS_TIMEOUT`, `JARVIS_TOOLS_APPROVAL_TTL`,
`JARVIS_TOOLS_AUDIT`.

```bash
python -m robanayt_ai --tools-list                      # инструменты и их уровень
python -m robanayt_ai --tools-describe                  # описания для AI-запроса
python -m robanayt_ai --tools-run current_time
python -m robanayt_ai --tools-run echo_text --tools-args '{"text": "hi"}'             # approval_required, exit 6
python -m robanayt_ai --tools-run echo_text --tools-args '{"text": "hi"}' --tools-approve
python -m robanayt_ai --tools-audit                     # аудит этого запуска
```

### Диалог (`[orchestration]`)

```toml
[orchestration]
enabled = true
max_tool_calls_per_message = 3        # 0..10
allow_text_tool_calls = false         # не угадывать вызовы из прозы
persist_history = true
pending_ttl_seconds = 300.0
```

Env: `JARVIS_ORCHESTRATION_ENABLED`, `JARVIS_ORCHESTRATION_MAX_TOOL_CALLS`,
`JARVIS_ORCHESTRATION_ALLOW_TEXT_CALLS`, `JARVIS_ORCHESTRATION_PERSIST`.

```bash
python -m robanayt_ai --dialog-send "Сколько будет 2+40?"
python -m robanayt_ai --dialog-send "ещё" --dialog-conversation ID
python -m robanayt_ai --dialog-pending                 # ожидают подтверждения
python -m robanayt_ai --dialog-approve PENDING_ID      # подтвердить конкретный вызов
```

Один вызов CLI — один ограниченный ход; интерактивного цикла нет. Если
инструмент требует подтверждения, ход останавливается и печатает
`pending_id` и аргументы: без `--dialog-approve` ничего не выполняется.

Модель разрешений: `safe` выполняется сразу (побочных эффектов нет),
`approval_required` — только с подтверждением, привязанным к имени, версии и
дайджесту аргументов (одноразовое, с TTL), `denied` — никогда. Инструмент,
которого нет в политике, считается неизвестным и отклоняется. Ответ модели,
описание инструмента или текст из памяти **не** являются разрешением.

### Приватность

Данные памяти хранятся локально и никуда не передаются. Логи содержат только
идентификаторы и счётчики — не текст сообщений и не текст воспоминаний.
Метаданные с ключами вида `api_key`, `authorization`, `token`, `secret`,
`password` отклоняются при сохранении.

**Шифрование at-rest отсутствует**: SQLite-файл лежит на диске открытым
текстом. Это известное ограничение Фазы 3; самодельная криптография не
применяется, а внешние зависимости под неё не устанавливались.

Аудит инструментов не содержит значений аргументов и секретов: только
идентификатор, имя инструмента, время, статус, решение политики, дайджест
аргументов и очищенное сообщение об ошибке. Логи диалога содержат
идентификаторы, статусы и счётчики — не текст сообщений.

**Границы оркестрации.** Ход диалога конечен и управляем, но это не защита
ОС и не песочница: подтверждение означает согласие пользователя на один
конкретный вызов внутри процесса JARVIS и не ограничивает другие программы.
Автономного планирования, фоновых агентов и самоутверждения моделью
собственных вызовов нет.

**Границы защиты инструментов.** Политика работает только внутри процесса
JARVIS: это не песочница и не защита ОС. Подтверждение через `--tools-approve`
означает согласие пользователя на один конкретный вызов, а не разрешение для
других программ; инструмент, уже получивший разрешение, выполняется в том же
процессе. Настоящая изоляция (отдельный процесс, права ОС) в этой фазе
не реализована.

## Принципы (из VISION донора, сохраняем)

1. Local-first: ядро и инференс не требуют сети.
2. Ядро владеет оркестрацией, политиками, историей и маршрутизацией инструментов.
3. Поверхности (UI) — тонкие клиенты, не владеют логикой движка.
4. LLM-бэкенд — сменяемый провайдер вычислений.
5. Инструменты — адаптеры с ограниченными capability-контрактами.
6. Честное неполное состояние вместо имитации успеха.
