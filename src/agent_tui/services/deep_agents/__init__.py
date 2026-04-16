"""DeepAgents integration for agent-tui.

Public API:
- DeepAgentsAdapter: Bridge between DeepAgents and the TUI
- EventTranslator: Translates DeepAgents events to domain events
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_tui.services.deep_agents.adapter import DeepAgentsAdapter
    from agent_tui.services.deep_agents.event_translator import EventTranslator


def __getattr__(name: str):
    """Lazy loading to avoid importing heavy dependencies at module load time."""
    if name == "DeepAgentsAdapter":
        from agent_tui.services.deep_agents.adapter import DeepAgentsAdapter

        return DeepAgentsAdapter
    if name == "EventTranslator":
        from agent_tui.services.deep_agents.event_translator import EventTranslator

        return EventTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeepAgentsAdapter",
    "EventTranslator",
]
