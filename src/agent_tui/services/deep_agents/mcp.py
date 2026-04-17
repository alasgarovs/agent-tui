"""MCP integration for DeepAgents.

Handles MCP server discovery, tool registration, and communication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class MCPToolManager:
    """Manages MCP servers and their tools."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize MCP tool manager.

        Args:
            config_path: Path to .mcp.json configuration file.
                        Defaults to .mcp.json in current working directory.
        """
        self.config_path = config_path or Path.cwd() / ".mcp.json"
        self.servers: dict[str, Any] = {}  # Store server connections
        self.tools: dict[str, BaseTool] = {}  # Store discovered tools
        self._sessions: dict[str, Any] = {}  # Store MCP sessions
        self._stdio_clients: dict[str, Any] = {}  # Store stdio client contexts

    def load_config(self) -> dict[str, Any] | None:
        """Load .mcp.json configuration.

        Returns:
            Parsed configuration dict, or None if file doesn't exist
            or contains invalid JSON.
        """
        if not self.config_path.exists():
            logger.info("No .mcp.json found at %s", self.config_path)
            return None

        try:
            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse .mcp.json: %s", e)
            return None
        except Exception as e:
            logger.error("Error reading .mcp.json: %s", e)
            return None

    async def start_servers(self) -> None:
        """Start all configured MCP servers.

        Discovers tools from each server and converts them to
        LangChain-compatible format.
        """
        config = self.load_config()
        if not config:
            return

        servers_config = config.get("servers", {})
        for name, server_config in servers_config.items():
            try:
                await self._start_server(name, server_config)
            except Exception as e:
                logger.error("Failed to start MCP server '%s': %s", name, e)

    async def _start_server(self, name: str, config: dict) -> None:
        """Start a single MCP server and discover its tools.

        Args:
            name: Server identifier from configuration.
            config: Server configuration dict with command, args, env.

        Raises:
            ImportError: If mcp package is not installed.
            Exception: If server fails to start or tool discovery fails.
        """
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            logger.error("MCP server '%s' has no command configured", name)
            return

        # Import MCP SDK here (lazy load to handle optional dependency)
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.error("mcp package not installed. Run: uv add mcp>=1.0.0 or pip install 'mcp>=1.0.0'")
            return

        # Create server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        try:
            # Connect via stdio
            stdio_ctx = stdio_client(server_params)
            read_stream, write_stream = await stdio_ctx.__aenter__()

            # Initialize MCP session
            session = await ClientSession(read_stream, write_stream).__aenter__()
            await session.initialize()

            # Store for later cleanup
            self.servers[name] = {
                "session": session,
                "stdio_ctx": stdio_ctx,
                "params": server_params,
            }
            self._sessions[name] = session
            self._stdio_clients[name] = stdio_ctx

            # Discover tools
            tools_result = await session.list_tools()

            # Convert and store tools
            for mcp_tool in tools_result.tools:
                try:
                    langchain_tool = self._convert_mcp_tool(mcp_tool, name)
                    self.tools[langchain_tool.name] = langchain_tool
                    logger.info(
                        "Registered MCP tool '%s' from server '%s'",
                        langchain_tool.name,
                        name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to convert MCP tool from server '%s': %s",
                        name,
                        e,
                    )

        except Exception as e:
            logger.error("Error starting MCP server '%s': %s", name, e)
            raise

    def _convert_mcp_tool(
        self,
        mcp_tool: Any,
        server_name: str,
    ) -> BaseTool:
        """Convert MCP tool format to LangChain tool.

        Args:
            mcp_tool: MCP tool object with name, description, inputSchema.
            server_name: Name of the originating MCP server.

        Returns:
            LangChain StructuredTool ready for agent registration.
        """
        tool_name = getattr(mcp_tool, "name", "unnamed_tool")
        description = getattr(mcp_tool, "description", "")
        input_schema = getattr(mcp_tool, "inputSchema", {})

        # Create a Pydantic model for the tool's input schema
        if input_schema and input_schema.get("properties"):
            fields = {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for prop_name, prop_schema in properties.items():
                field_type = self._json_schema_to_python_type(prop_schema)
                field_default = ... if prop_name in required else None
                field_description = prop_schema.get("description", "")

                if field_default is ...:
                    fields[prop_name] = (
                        field_type,
                        Field(..., description=field_description),
                    )
                else:
                    fields[prop_name] = (
                        field_type | None,
                        Field(default=None, description=field_description),
                    )

            if fields:
                InputModel = create_model(f"{tool_name}_input", **fields)
            else:
                InputModel = create_model(f"{tool_name}_input", __base__=BaseModel)
        else:
            InputModel = create_model(f"{tool_name}_input", __base__=BaseModel)

        # Create the async wrapper function
        async def tool_wrapper(**kwargs: Any) -> str:
            """Wrapper to call MCP tool via the session."""
            session = self._sessions.get(server_name)
            if not session:
                return f"Error: MCP server '{server_name}' not connected"

            try:
                result = await session.call_tool(tool_name, arguments=kwargs)

                # Extract content from result
                if hasattr(result, "content"):
                    content = result.content
                    if isinstance(content, list):
                        # Handle text content items
                        texts = []
                        for item in content:
                            if hasattr(item, "text"):
                                texts.append(item.text)
                            elif isinstance(item, str):
                                texts.append(item)
                            elif isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                        return "\n".join(texts) if texts else str(content)
                    return str(content)
                return str(result)
            except Exception as e:
                logger.error("Error calling MCP tool '%s': %s", tool_name, e)
                return f"Error: {e}"

        # Create the StructuredTool
        return StructuredTool(
            name=tool_name,
            description=description,
            func=tool_wrapper,
            args_schema=InputModel,
            return_direct=False,
        )

    def _json_schema_to_python_type(self, prop_schema: dict[str, Any]) -> type:
        """Convert JSON schema type to Python type.

        Args:
            prop_schema: JSON schema property definition.

        Returns:
            Python type for the property.
        """
        schema_type = prop_schema.get("type", "string")

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        return type_mapping.get(schema_type, str)

    def get_tools(self) -> list[BaseTool]:
        """Return list of all MCP tools for registration.

        Returns:
            List of LangChain tool objects ready for DeepAgents registration.
        """
        return list(self.tools.values())

    async def cleanup(self) -> None:
        """Stop all MCP servers and clean up resources."""
        for name, server_info in list(self.servers.items()):
            try:
                # Close session if available
                session = server_info.get("session")
                if session and hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)

                # Close stdio context if available
                stdio_ctx = server_info.get("stdio_ctx")
                if stdio_ctx and hasattr(stdio_ctx, "__aexit__"):
                    await stdio_ctx.__aexit__(None, None, None)

                logger.info("Stopped MCP server '%s'", name)
            except Exception as e:
                logger.warning("Error stopping MCP server '%s': %s", name, e)

        # Clear stored references
        self.servers.clear()
        self._sessions.clear()
        self._stdio_clients.clear()


async def discover_mcp_tools(
    project_dir: Path | None = None,
) -> list[BaseTool]:
    """Discover and return all MCP tools from .mcp.json.

    This is the main integration point for the adapter to call.
    It discovers tools from all configured MCP servers and returns
    them ready for registration with create_deep_agent(tools=...).

    Args:
        project_dir: Directory to look for .mcp.json.
                     Defaults to current working directory.

    Returns:
        List of LangChain tool objects ready for DeepAgents registration.

    Example:
        tools = await discover_mcp_tools(project_dir=Path("/my/project"))
        agent = create_deep_agent(
            model=model,
            tools=tools,  # MCP tools are now available to the agent
            ...
        )
    """
    config_path = (project_dir or Path.cwd()) / ".mcp.json"
    manager = MCPToolManager(config_path=config_path)

    # Check if config exists first
    if not config_path.exists():
        logger.info("No .mcp.json found, skipping MCP tool discovery")
        return []

    await manager.start_servers()
    tools = manager.get_tools()

    # Note: Servers will be cleaned up when the manager is garbage collected
    # or explicitly via manager.cleanup(). The caller should store the manager
    # reference if cleanup is needed later.
    # For now, we store it on the module for potential cleanup
    _active_managers.append(manager)

    return tools


# Global list to track active managers for cleanup
_active_managers: list[MCPToolManager] = []


async def cleanup_all_mcp_servers() -> None:
    """Clean up all active MCP server managers.

    Call this on application shutdown to properly close all MCP connections.
    """
    for manager in _active_managers:
        try:
            await manager.cleanup()
        except Exception as e:
            logger.warning("Error during MCP cleanup: %s", e)
    _active_managers.clear()
