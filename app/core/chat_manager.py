from typing import Dict, Set
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self):
        # Mapping: room_id -> set of WebSocket connections
        self.rooms: Dict[int, Set[WebSocket]] = {}
        # Mapping: user_id -> WebSocket connection for personal messages
        self.user_connections: Dict[int, WebSocket] = {}

    async def connect(self, room_id: int, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(room_id, set()).add(ws)

    async def connect_user(self, user_id: int, ws: WebSocket):
        """Connect a user for personal messaging"""
        await ws.accept()
        self.user_connections[user_id] = ws
        logger.info(f"User {user_id} connected for personal messaging")

    def disconnect(self, room_id: int, ws: WebSocket):
        room = self.rooms.get(room_id)
        if room and ws in room:
            room.remove(ws)
            if not room:
                self.rooms.pop(room_id, None)

    def disconnect_user(self, user_id: int):
        """Disconnect a user from personal messaging"""
        if user_id in self.user_connections:
            del self.user_connections[user_id]
            logger.info(f"User {user_id} disconnected from personal messaging")

    async def broadcast(self, room_id: int, message: dict):
        dead: list[WebSocket] = []
        for ws in self.rooms.get(room_id, set()):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(room_id, ws)

    async def send_personal_message(self, message: str, user_id: int):
        """Send a personal message to a specific user"""
        if user_id in self.user_connections:
            try:
                ws = self.user_connections[user_id]
                await ws.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send personal message to user {user_id}: {e}")
                self.disconnect_user(user_id)

    async def broadcast_voice_message(self, voice_data: dict, participant_ids: list):
        """Broadcast voice message to participants"""
        message_json = json.dumps(voice_data)
        for user_id in participant_ids:
            await self.send_personal_message(message_json, user_id)

chat_manager = ChatManager()
