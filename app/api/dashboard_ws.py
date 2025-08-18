"""
Dashboard WebSocket API for real-time online status updates.
Optimized to avoid impact on translation performance.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
import logging
from datetime import datetime
from typing import Dict, Set

from ..core.security import decode_access_token
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Store dashboard WebSocket connections - separate from chat connections
dashboard_connections: Dict[int, Set[WebSocket]] = {}

async def get_user_from_token(token: str, db: AsyncSession) -> User:
    """Get user from JWT token"""
    try:
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        result = await db.execute(select(User).where(User.email == payload["sub"]))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket, token: str = Query(...)):
    """WebSocket endpoint for real-time dashboard updates"""
    user = None
    logger.info(f"Dashboard WebSocket connection attempt with token: {token[:20]}...")
    
    try:
        # Authenticate user first
        from ..core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            user = await get_user_from_token(token, db)
        
        logger.info(f"Dashboard WebSocket authentication successful for user {user.id}")
        await websocket.accept()
        logger.info(f"Dashboard WebSocket connected for user {user.id}")
        
        # Add to dashboard connections
        if user.id not in dashboard_connections:
            dashboard_connections[user.id] = set()
        dashboard_connections[user.id].add(websocket)
        
        try:
            # Send initial online users list
            from ..services.presence_manager import presence_manager
            online_users = await presence_manager.get_online_users()
            initial_message = {
                "type": "online_users_update",
                "online_user_ids": list(online_users),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"Sending initial online users: {list(online_users)}")
            await websocket.send_text(json.dumps(initial_message))
            
            # Keep connection alive and handle any incoming messages
            while True:
                try:
                    # We don't expect many messages from dashboard, just keep alive
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    if message_data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from dashboard")
                except Exception as e:
                    logger.error(f"Error processing dashboard message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in dashboard WebSocket: {e}")
            
    except WebSocketDisconnect:
        if user:
            logger.info(f"Dashboard WebSocket disconnected for user {user.id}")
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        logger.exception("Full dashboard WebSocket error:")
        try:
            await websocket.close(code=4001)
        except Exception:
            pass
    finally:
        # Clean up connection
        if user and user.id in dashboard_connections:
            dashboard_connections[user.id].discard(websocket)
            if not dashboard_connections[user.id]:
                del dashboard_connections[user.id]

async def broadcast_online_status_to_dashboards(user_id: int, status: str):
    """Broadcast online status changes to all dashboard connections"""
    if not dashboard_connections:
        return
    
    try:
        from ..services.presence_manager import presence_manager
        online_users = await presence_manager.get_online_users()
        message = {
            "type": "online_users_update",
            "online_user_ids": list(online_users),
            "user_status_changed": {
                "user_id": user_id,
                "status": status
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_str = json.dumps(message)
        dead_connections = []
        
        # Send to all dashboard connections
        for user_dashboard_id, websockets in list(dashboard_connections.items()):
            for websocket in list(websockets):
                try:
                    await websocket.send_text(message_str)
                except Exception as e:
                    logger.error(f"Failed to send to dashboard WebSocket: {e}")
                    dead_connections.append((user_dashboard_id, websocket))
        
        # Clean up dead connections
        for user_dashboard_id, dead_ws in dead_connections:
            if user_dashboard_id in dashboard_connections:
                dashboard_connections[user_dashboard_id].discard(dead_ws)
                if not dashboard_connections[user_dashboard_id]:
                    del dashboard_connections[user_dashboard_id]
                    
    except Exception as e:
        logger.error(f"Error broadcasting to dashboards: {e}")
