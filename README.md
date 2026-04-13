# agent-tui

Standalone TUI scaffold for building AI coding agent CLIs.

Extracted from the deepagents CLI — full UI with all agent interactions stubbed behind an `AgentProtocol`.

## Quick Start

```bash
uv run agent-tui
```

## Architecture

The TUI communicates with any agent backend through `AgentProtocol` — a Python protocol class that emits typed `AgentEvent` objects. A `StubAgent` is included for development and testing.

See `agent_tui/protocol.py` for the contract.
