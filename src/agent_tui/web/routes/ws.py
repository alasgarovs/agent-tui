"""WebSocket routes for real-time agent communication."""

from __future__ import annotations

import json
import logging
import os
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent_tui.services.agent_factory import create_agent
from agent_tui.services.web_adapter import WebAdapter
from agent_tui.web.state import ConnectionState, connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections for agent communication."""
    await websocket.accept()
    
    client_id = str(uuid.uuid4())
    
    # Create agent instance based on configuration
    agent_type = os.environ.get('AGENT_TUI_WEB_AGENT', 'stub')
    try:
        agent = create_agent(agent_type)
        logger.info(f"Created {agent_type} agent for client {client_id}")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to initialize agent: {e}"
        })
        await websocket.close()
        return
    
    # Create connection state
    state = ConnectionState(
        websocket=websocket,
        agent=agent
    )
    
    # Register connection
    await connection_manager.connect(client_id, state)
    
    # Create adapter
    adapter = WebAdapter(agent, websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            
            match msg_type:
                case "chat":
                    # Start streaming response
                    await adapter.run_task(
                        message.get("message", ""),
                        thread_id=message.get("thread_id")
                    )
                
                case "approve_tool":
                    await adapter.approve_tool(
                        message.get("tool_id", ""),
                        message.get("approved", False)
                    )
                
                case "answer":
                    await adapter.answer_question(message.get("answer", ""))
                
                case "cancel":
                    await adapter.cancel()
                
                case _:
                    logger.warning("Unknown message type: %s", msg_type)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
    
    except WebSocketDisconnect:
        logger.info("Client %s disconnected", client_id)
    except Exception:
        logger.exception("WebSocket error")
    finally:
        await connection_manager.disconnect(client_id)
