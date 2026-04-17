# Strands Agent Backend Adapter — Design Spec

**Date:** 2026-04-17  
**Status:** Approved  
**Scope:** Add a `StrandsAdapter` that implements `AgentProtocol` using the Strands Python SDK as a third agent backend alongside `StubAgent` and `DeepAgentsAdapter`.

---

## Context

The TUI communicates with any agent backend through `AgentProtocol` (`domain/protocol.py`), an async Python Protocol class that emits typed `AgentEvent` objects. The existing `DeepAgentsAdapter` wraps LangGraph via the deepagents library. This spec describes a new `StrandsAdapter` that wraps the [Strands Agents SDK](https://strandsagents.com/docs/) (AWS open-source, not LangGraph-based).

---

## Goals

- Implement `AgentProtocol` fully using the Strands Python SDK.
- Use the `strands-tools` community package for tools (30+ pre-built tools).
- Bridge Strands' sequential re-invocation interrupt model into the TUI's streaming HITL contract.
- Mirror DeepAgents HITL policy: gate `shell`, `file_write`, `editor`, `http_request`, `web_search` on user approval.
- Use OpenAI as the default model provider (reusing existing `OPENAI_API_KEY`).
- Persist conversation history via Strands `FileSessionManager` (full parity with DeepAgents `MemorySaver`).
- Zero breakage to existing backends.

---

## Architecture

### File Layout

```
src/agent_tui/services/strands/
    __init__.py          # exports StrandsAdapter
    adapter.py           # StrandsAdapter — implements AgentProtocol
    event_translator.py  # maps Strands stream events → AgentEvent (pure, no side-effects)
    tools.py             # tool list selection, ApprovalHook definition
```

Existing files modified:
- `src/agent_tui/entrypoints/main.py` — add `--agent=strands` choice
- `src/agent_tui/configurator/settings.py` — add `STRANDS_MODEL` and `STRANDS_PROVIDER` env vars

---

## Components

### `event_translator.py`

A pure function module with `translate(event: dict) -> Iterator[AgentEvent]`. No Strands imports, no state. Unit-testable in isolation.

**Mapping:**

| Strands event key | AgentEvent type |
|---|---|
| `"data"` (non-empty str) | `MESSAGE_CHUNK` |
| `"result"` present | `MESSAGE_END` + optional `TOKEN_UPDATE` (from `result.metrics`) |
| `"current_tool_use"` with complete `name` + `input` | `TOOL_CALL` |
| tool output available | `TOOL_RESULT` |
| `"force_stop"` is `True` | `ERROR` |
| `"init_event_loop"`, `"start_event_loop"` | ignored |

**TOOL_CALL detection:** The translator is stateless. Since Strands resolves the complete tool `input` dict before calling `BeforeToolCallEvent` (and the stream pauses at that point), the `current_tool_use` event received in the stream already carries the complete `name` and `input`. The translator emits `TOOL_CALL` for any event where `current_tool_use` contains both a non-empty `name` and a non-empty `input` dict.

**TOOL_RESULT:** After re-invocation with the approved interrupt response, Strands executes the tool and continues the agent loop. Tool output appears in a subsequent stream event. The exact Strands event key for tool results must be verified during implementation by inspecting the raw stream events. The translator should emit `TOOL_RESULT` when it sees a Strands event containing tool output data (likely as a `tool_result` key or within `current_tool_use` with an `output` field). The unit test for `TOOL_RESULT` will be written against the verified event structure.

---

### `tools.py`

Selects and configures tools from `strands-tools` community package.

**Tool mapping:**

| Function | `strands-tools` tool | HITL gated |
|---|---|---|
| Read file | `file_read` | No |
| Write file | `file_write` | Yes |
| Edit file | `editor` | Yes |
| Find files (glob) | `find_file` | No |
| Search content | `grep` | No |
| List directory | `git_repo_tree` | No |
| Shell execution | `shell` | Yes |
| HTTP request | `http_request` | Yes |
| Web search | custom `web_search` `@tool` wrapping `tavily-python` | Yes |

The web search tool is a custom `@tool`-decorated function in `tools.py` that wraps `tavily-python` (already a project dependency in `pyproject.toml`). It mirrors the `create_web_search_tool()` pattern in `deep_agents/web_tools.py`. If `TAVILY_API_KEY` is not set, the tool returns an error string instead of raising.

**`ApprovalHook`** is a `HookProvider` that unconditionally registers a callback named `_on_before_tool_call` on `BeforeToolCallEvent`. Inside the callback, it checks `event.tool_use["name"]` against the gated-tool set. For gated tools: calls `event.interrupt("strands-approval", reason={"tool_name": name, "tool_args": input})` and examines the response — if the response is not `"approved"`, sets `event.cancel_tool = "User denied tool execution"` to skip the tool. For non-gated tools: returns immediately without calling `interrupt`.

The `event.cancel_tool` property is documented in the [Strands interrupt API](https://strandsagents.com/docs/user-guide/concepts/interrupts/) — setting it to a string causes Strands to skip the tool call and return the string as the tool result to the LLM. The adapter should yield a `TOOL_RESULT` event with `tool_output="User denied tool execution"` after the denial re-invocation stream starts (visible in the next stream's events as the tool output).

`tools.py` also exports `GATED_TOOLS: frozenset[str]`, a set of tool names that require approval:
```python
GATED_TOOLS = frozenset({"shell", "file_write", "editor", "http_request", "web_search"})

def tool_needs_approval(tool_name: str) -> bool:
    return tool_name in GATED_TOOLS
```
This function is imported into `adapter.py` and used in the stream loop.

**Interrupt ID flow:** When Strands stops due to an interrupt, `result.interrupts` contains the raised interrupts. Each interrupt has an `.id` field (a Strands-generated UUID). The adapter reads this id from the stream's terminal `result` event, stores it as `self._pending_interrupt_id`, and uses it when building the `interruptResponse` payload on re-invocation. The `tool_id` field of the yielded `TOOL_CALL` event is set to this Strands interrupt id, so `approve_tool(tool_id, approved)` can match it.

---

### `adapter.py` — `StrandsAdapter`

Implements all 9 `AgentProtocol` methods via lazy-loaded Strands `Agent`.

#### Initialization

```python
class StrandsAdapter:
    def __init__(self, model: str = "gpt-4o", *, api_key: str | None = None,
                 tavily_api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._tavily_api_key = tavily_api_key
        self._agents: dict[str, Any] = {}   # keyed by thread_id for session reuse
        self._translator = StrandsEventTranslator()
        self._cancelled = False
        self._approval_event: asyncio.Event | None = None
        self._approval_result: bool = False
        self._pending_tool_id: str | None = None
        self._answer_event: asyncio.Event | None = None
        self._user_answer: str = ""
        self._pending_interrupt_id: str | None = None  # Strands interrupt UUID
        self._strands_available: bool = self._check_strands_available()

    @classmethod
    def from_settings(cls) -> "StrandsAdapter":
        from agent_tui.configurator.settings import settings
        return cls(
            model=settings.strands_model,
            api_key=settings.openai_api_key,
            tavily_api_key=settings.tavily_api_key,
        )

    def _check_strands_available(self) -> bool:
        try:
            import strands  # noqa: F401
            return True
        except ImportError:
            return False
```

Agent is created lazily via `_ensure_agent(thread_id)`. Sessions are persisted to `~/.agent-tui/strands-sessions/<thread_id>/` via `FileSessionManager`. On `set_model()`, all cached agents are cleared.

#### `stream()` — Restart-Loop HITL Bridge

Strands `stream_async()` emits events in this sequence for a tool-requiring interrupt:
1. `{"data": "some text"}` — text chunks from the LLM
2. `{"current_tool_use": {"name": "shell", "input": {"command": "ls"}}}` — tool call ready
3. `{"result": AgentResult(stop_reason="interrupt", interrupts=[Interrupt(id="uuid", ...)])}` — stream ends

The adapter buffers the `current_tool_use` event and yields the `TOOL_CALL` only after seeing the terminal `result` event (which carries the Strands interrupt id). Then it waits for `approve_tool()` and re-invokes.

```
stream(message, *, thread_id):
  agent = _ensure_agent(thread_id)
  strands_input = message
  pending_tool_call = None    # buffered TOOL_CALL agent_event

  while True:
    async for raw_event in agent.stream_async(strands_input):
      if cancelled: return

      for agent_event in translator.translate(raw_event):
        if agent_event.type == TOOL_CALL:
          pending_tool_call = agent_event   # buffer, don't yield yet

        elif agent_event.type == MESSAGE_END:
          # Check if this is an interrupt termination (result.stop_reason)
          result = raw_event.get("result")
          if result and result.stop_reason == "interrupt" and pending_tool_call:
            # Store Strands interrupt id for re-invocation
            if not result.interrupts:
              yield AgentEvent(type=ERROR, text="Interrupt with no interrupt data")
              return
            self._pending_interrupt_id = result.interrupts[0].id
            pending_tool_call = pending_tool_call._replace(
                tool_id=self._pending_interrupt_id)

            if tool_needs_approval(pending_tool_call.tool_name):
              # Gate on user approval
              self._approval_event = asyncio.Event()
              yield pending_tool_call
              await self._approval_event.wait()
              response = "approved" if self._approval_result else "denied"
              strands_input = [{"interruptResponse": {
                  "interruptId": self._pending_interrupt_id,
                  "response": response,
              }}]
              pending_tool_call = None
              break  # restart stream with interrupt response
            else:
              # Auto-approve non-gated tool (shouldn't interrupt, but be safe)
              strands_input = [{"interruptResponse": {
                  "interruptId": self._pending_interrupt_id,
                  "response": "approved",
              }}]
              pending_tool_call = None
              break
          else:
            yield agent_event  # normal MESSAGE_END → return
            return
        else:
          yield agent_event
      else:
        continue  # inner for loop didn't break; continue async-for
      break  # inner for loop broke; restart outer while
    else:
      return  # async-for exhausted without break → done
```

**Interrupt flow:**
- `ApprovalHook._on_before_tool_call()` raises `event.interrupt("strands-approval", ...)`.
- Strands pauses the tool, terminates `stream_async()` with `result.stop_reason == "interrupt"`.
- Adapter reads `result.interrupts[0].id` as `_pending_interrupt_id`, yields `TOOL_CALL`, waits for `approve_tool()`.
- On approval: re-invokes `agent.stream_async([{"interruptResponse": {"interruptId": id, "response": "approved"}}])`. Strands resumes, `ApprovalHook` sees `"approved"`, tool executes.
- On denial: re-invokes with `"denied"`. `ApprovalHook` sets `event.cancel_tool = "User denied"`, Strands skips the tool and continues the loop.
- Both cases restart the outer `while True` loop with `strands_input` set to the interrupt response.

#### Other Methods

| Method | Implementation |
|---|---|
| `approve_tool(tool_id, approved)` | Sets `_approval_result`, fires `_approval_event` |
| `answer_question(answer)` | Stub: stores `answer` in `_user_answer`, fires `_answer_event`. No Strands ASK_USER mapping in scope for this phase — intentionally stubbed, no-op equivalent. |
| `cancel()` | Sets `_cancelled=True`, fires all pending events (`_approval_event`, `_answer_event`) |
| `get_threads()` | Lists subdirs in `~/.agent-tui/strands-sessions/` |
| `get_models()` | Returns OpenAI model list: `gpt-4o`, `gpt-4o-mini`, `gpt-4o-2024-11-20` |
| `set_model(name)` | Updates `_model`, clears `_agents` cache |
| `get_skills()` | Returns `[]` (Strands has native skills concept; future enhancement) |
| `invoke_skill(name, args)` | No-op (future enhancement) |

---

## Settings

New env vars (added to `configurator/settings.py`):

| Env var | Default | Description |
|---|---|---|
| `STRANDS_MODEL` | `gpt-4o` | Model name passed to Strands OpenAI provider |
| `STRANDS_PROVIDER` | `openai` | Provider hint (future: `anthropic`, `bedrock`) |

Reuses existing `OPENAI_API_KEY` / `AGENT_TUI_OPENAI_API_KEY`.

---

## Entry Point Wiring

`main.py` changes:

```python
parser.add_argument(
    "--agent",
    choices=["stub", "deepagents", "strands"],
    default="stub",
)

# In cli_main():
elif _args.agent == "strands":
    from agent_tui.services.strands import StrandsAdapter
    agent = StrandsAdapter.from_settings()
```

---

## Session Persistence

- `FileSessionManager(session_id=thread_id, storage_dir=~/.agent-tui/strands-sessions/)`
- One session directory per `thread_id`.
- `get_threads()` scans `~/.agent-tui/strands-sessions/` for subdirectories and returns a list of dicts matching the protocol shape:
  ```python
  [{"id": dir_name, "title": dir_name, "updated_at": mtime_isoformat,
    "created_at": ctime_isoformat, "message_count": 0}]
  ```
- On model switch (`set_model()`), cached `Agent` objects are discarded; sessions persist on disk.

---

## Dependency Management

New optional dependency group in `pyproject.toml`:

```toml
[dependency-groups]
strands = [
    "strands-agents>=0.1.0",
    "strands-tools>=0.1.0",
]
```

`StrandsAdapter` uses lazy import (checks availability on init, raises informative `RuntimeError` if not installed). This keeps `uv run agent-tui` working even without Strands installed.

---

## Error Handling

- If `strands-agents` not installed: `stream()` raises `RuntimeError` with install instructions.
- If API key missing: `stream()` catches the Strands auth error and yields `AgentEvent(type=ERROR, text=...)`.
- If `stream_async()` raises: caught and translated to `ERROR` event, stream terminates cleanly.
- Cancellation: `_cancelled` flag is checked at each iteration; `_approval_event` is fired to unblock any waiting gate.

---

## Testing

### Unit Tests

**`tests/unit/strands/test_event_translator.py`**
- Pure dict-in → `AgentEvent` assertions
- Covers: text chunks, TOOL_CALL detection, TOOL_RESULT (once event structure is verified during implementation), force stop, ignored events

**`tests/unit/strands/test_adapter.py`**
- Mock `strands.Agent.stream_async()` with scripted event sequences
- Tests: normal stream, HITL approval, HITL denial, cancellation mid-stream, model switch

### Integration Tests

**`tests/integration/strands/test_smoke.py`**
- Skipped if `OPENAI_API_KEY` not set
- Sends one message, asserts ≥1 `MESSAGE_CHUNK` and a terminal `MESSAGE_END`

### E2E

Existing pilot/PTY E2E tests are backend-agnostic. Running `uv run agent-tui --agent=strands` exercises the full stack end-to-end.

---

## Out of Scope

- MCP tool integration via Strands (future — Strands has native MCP support)
- Strands skills plugin system (future)
- Non-OpenAI providers (future — architecture supports it via `STRANDS_PROVIDER` env var)
- Sandbox/isolated execution (future — Strands has no built-in sandbox equivalent)
- Memory via AGENTS.md files (future enhancement matching DeepAgents)
