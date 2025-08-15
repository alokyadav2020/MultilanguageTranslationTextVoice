from typing import Dict, Set
from fastapi import WebSocket

class ChatManager:
    def __init__(self):
        # Mapping: room_id -> set of WebSocket connections
        self.rooms: Dict[int, Set[WebSocket]] = {}

    async def connect(self, room_id: int, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(room_id, set()).add(ws)

    def disconnect(self, room_id: int, ws: WebSocket):
        room = self.rooms.get(room_id)
        if room and ws in room:
            room.remove(ws)
            if not room:
                self.rooms.pop(room_id, None)

    async def broadcast(self, room_id: int, message: dict):
        dead: list[WebSocket] = []
        for ws in self.rooms.get(room_id, set()):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(room_id, ws)

chat_manager = ChatManager()
