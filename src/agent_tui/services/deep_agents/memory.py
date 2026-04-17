"""DeepAgents memory helpers — AGENTS.md source discovery and content reading."""

from __future__ import annotations

from pathlib import Path


def get_memory_sources() -> list[str]:
    """Return a list of AGENTS.md paths that exist on disk, as POSIX path strings.

    Checks these paths in order:
    1. ~/.deepagents/AGENTS.md (user-level)
    2. ./.deepagents/AGENTS.md (project-level, relative to cwd)

    Only includes paths that exist as files.

    Returns:
        List of POSIX path strings for existing AGENTS.md files.
        Empty list if none exist.
    """
    sources = []
    candidates = [
        "~/.deepagents/AGENTS.md",
        "./.deepagents/AGENTS.md",
    ]

    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            sources.append(Path(candidate).as_posix())

    return sources


def read_memory_content() -> dict[str, str]:
    """Return dict mapping display path (str) → file content (str).

    Reads each path returned by get_memory_sources().
    If a file cannot be read, it is skipped silently.

    Returns:
        Dict mapping original path string (e.g. "~/.deepagents/AGENTS.md")
        to file content. Empty dict if nothing is readable.
    """
    content = {}

    for source in get_memory_sources():
        try:
            path = Path(source).expanduser().resolve()
            content[source] = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            # OSError: permission denied or missing; ValueError: invalid encoding
            pass

    return content


def get_memory_summary() -> str:
    """Return a human-readable summary of what memory is loaded.

    Returns:
        String describing loaded memory sources and their line counts.
        If no sources exist, returns a message with instructions to create one.
    """
    content = read_memory_content()

    if not content:
        return "No AGENTS.md files found. Create ~/.deepagents/AGENTS.md to add persistent memory."

    lines = [f"Memory loaded from {len(content)} source(s):"]
    for source, text in content.items():
        line_count = len(text.splitlines())
        noun = "line" if line_count == 1 else "lines"
        lines.append(f"  {source} ({line_count} {noun})")

    return "\n".join(lines)
