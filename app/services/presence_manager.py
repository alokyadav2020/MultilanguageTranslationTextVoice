"""
Real-time presence and activity manager for the chat application.
Handles user online status, typing indicators, and broadcasts updates.
"""
import asyncio
import json
import logging
from typing import Set, Dict, Optional, List
from fastapi import WebSocket
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

@dataclass
class ChatConnection:
    websocket: WebSocket
    user_id: int
    room_id: str
    user1_id: int
    user2_id: int
    last_activity: float

@dataclass 
class TypingStatus:
    user_id: int
    room_id: str
    last_typing: float

@dataclass
class UserActivity:
    user_id: int
    last_seen: float
    activity_type: str  # 'chat' or 'dashboard'

class PresenceManager:
    def __init__(self):
        # Active chat WebSocket connections
        self.chat_connections: Dict[str, ChatConnection] = {}
        
        # User typing status
        self.typing_status: Dict[str, TypingStatus] = {}
        
        # General user activity tracking (dashboard visits, API calls, etc.)
        self.user_activities: Dict[int, UserActivity] = {}
        
        # Cleanup old data periodically
        self._cleanup_task = None
        
    async def start_background_tasks(self):
        """Start background cleanup tasks."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def stop_background_tasks(self):
        """Stop background cleanup tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def add_chat_connection(self, connection_id: str, websocket: WebSocket, 
                                user_id: int, room_id: str, user1_id: int, user2_id: int):
        """Add a new chat WebSocket connection."""
        logger.info(f"Adding chat connection {connection_id} for user {user_id} in room {room_id}")
        
        connection = ChatConnection(
            websocket=websocket,
            user_id=user_id,
            room_id=room_id,
            user1_id=user1_id,
            user2_id=user2_id,
            last_activity=time.time()
        )
        
        self.chat_connections[connection_id] = connection
        
        # Broadcast updated online users to this chat room
        await self._send_to_chat_room(room_id, {
            "type": "online_users_update",
            "online_user_ids": list(self.get_online_users())
        })
        
        logger.info(f"Chat connection {connection_id} added. Total connections: {len(self.chat_connections)}")
    
    async def remove_chat_connection(self, connection_id: str):
        """Remove a chat WebSocket connection."""
        logger.info(f"Removing chat connection {connection_id}")
        
        if connection_id in self.chat_connections:
            connection = self.chat_connections[connection_id]
            room_id = connection.room_id
            del self.chat_connections[connection_id]
            
            # Broadcast updated online users to affected rooms
            await self._send_to_chat_room(room_id, {
                "type": "online_users_update", 
                "online_user_ids": list(self.get_online_users())
            })
            
            logger.info(f"Chat connection {connection_id} removed. Total connections: {len(self.chat_connections)}")
        else:
            logger.warning(f"Attempted to remove non-existent chat connection {connection_id}")
    
    def get_online_users(self) -> Set[int]:
        """Get set of currently online user IDs."""
        online_users = set()
        current_time = time.time()
        
        # Users with active chat connections
        for connection in self.chat_connections.values():
            # Consider users online if they had activity in the last 2 minutes
            if current_time - connection.last_activity < 120:
                online_users.add(connection.user_id)
        
        # Users with recent general activity (dashboard, API calls)
        for user_id, activity in self.user_activities.items():
            # Consider users online if they had activity in the last 5 minutes
            if current_time - activity.last_seen < 300:
                online_users.add(user_id)
        
        return online_users
    
    async def update_user_general_activity(self, user_id: int, activity_type: str = "dashboard"):
        """Update general user activity (dashboard visits, API calls, etc.)."""
        current_time = time.time()
        self.user_activities[user_id] = UserActivity(
            user_id=user_id,
            last_seen=current_time,
            activity_type=activity_type
        )
        logger.debug(f"Updated {activity_type} activity for user {user_id}")
    
    async def remove_user_from_presence(self, user_id: int):
        """Remove user completely from presence tracking (for logout)."""
        # Remove from user activities
        if user_id in self.user_activities:
            del self.user_activities[user_id]
            logger.info(f"Removed user {user_id} from presence tracking")
        
        # Remove any typing status for this user
        typing_keys_to_remove = [
            key for key, status in self.typing_status.items()
            if status.user_id == user_id
        ]
        for key in typing_keys_to_remove:
            del self.typing_status[key]
        
        # Broadcast updated online status to all chat connections
        await self._broadcast_online_status_to_all()
        
        # Note: Chat connections will be closed automatically when WebSocket disconnects
        logger.info(f"User {user_id} removed from all presence tracking")
    
    async def _broadcast_online_status_to_all(self):
        """Broadcast online users update to all active chat connections."""
        online_user_ids = list(self.get_online_users())
        message = {
            "type": "online_users_update",
            "online_user_ids": online_user_ids
        }
        message_str = json.dumps(message)
        disconnected_connections = []
        
        for connection_id, connection in self.chat_connections.items():
            try:
                await connection.websocket.send_text(message_str)
                logger.debug(f"Sent online status update to connection {connection_id}")
            except Exception as e:
                logger.warning(f"Failed to send online status to connection {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.remove_chat_connection(connection_id)
    
    async def update_user_activity(self, connection_id: str):
        """Update last activity time for a user connection."""
        if connection_id in self.chat_connections:
            self.chat_connections[connection_id].last_activity = time.time()
    
    async def set_user_typing(self, user_id: int, room_id: str, is_typing: bool):
        """Set user typing status and broadcast to room."""
        typing_key = f"{user_id}_{room_id}"
        current_time = time.time()
        
        if is_typing:
            self.typing_status[typing_key] = TypingStatus(
                user_id=user_id,
                room_id=room_id,
                last_typing=current_time
            )
            logger.debug(f"User {user_id} started typing in room {room_id}")
        else:
            if typing_key in self.typing_status:
                del self.typing_status[typing_key]
            logger.debug(f"User {user_id} stopped typing in room {room_id}")
        
        # Broadcast typing status to room
        typing_users = [
            status.user_id for status in self.typing_status.values()
            if status.room_id == room_id and current_time - status.last_typing < 5
        ]
        
        await self._send_to_chat_room(room_id, {
            "type": "typing_update",
            "typing_user_ids": typing_users
        })
    
    async def _send_to_chat_room(self, room_id: str, message: dict):
        """Send message to all connections in a chat room."""
        message_str = json.dumps(message)
        disconnected_connections = []
        
        for connection_id, connection in self.chat_connections.items():
            if connection.room_id == room_id:
                try:
                    await connection.websocket.send_text(message_str)
                    logger.debug(f"Sent message to connection {connection_id} in room {room_id}")
                except Exception as e:
                    logger.warning(f"Failed to send message to connection {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.remove_chat_connection(connection_id)
    
    async def _periodic_cleanup(self):
        """Periodically clean up stale data."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_stale_data(self):
        """Remove stale typing status and inactive connections."""
        current_time = time.time()
        
        # Clean up old typing status (older than 5 seconds)
        stale_typing = [
            key for key, status in self.typing_status.items()
            if current_time - status.last_typing > 5
        ]
        
        for key in stale_typing:
            del self.typing_status[key]
        
        if stale_typing:
            logger.debug(f"Cleaned up {len(stale_typing)} stale typing statuses")
        
        # Clean up very old connections (older than 5 minutes)
        stale_connections = [
            connection_id for connection_id, connection in self.chat_connections.items()
            if current_time - connection.last_activity > 300
        ]
        
        for connection_id in stale_connections:
            logger.info(f"Removing stale connection {connection_id}")
            await self.remove_chat_connection(connection_id)
        
        # Clean up old user activities (older than 10 minutes)
        stale_activities = [
            user_id for user_id, activity in self.user_activities.items()
            if current_time - activity.last_seen > 600
        ]
        
        for user_id in stale_activities:
            del self.user_activities[user_id]
        
        if stale_activities:
            logger.debug(f"Cleaned up {len(stale_activities)} stale user activities")

# Global presence manager instance
presence_manager = PresenceManager()
