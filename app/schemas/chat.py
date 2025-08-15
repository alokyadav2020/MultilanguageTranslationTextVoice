
# filepath: app/schemas/chat.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
from .user import UserRead

class ChatroomBase(BaseModel):
    chatroom_name: Optional[str] = None
    is_group_chat: bool = False

class ChatroomCreate(ChatroomBase):
    pass

class ChatroomRead(ChatroomBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    original_text: str
    message_type: str = "text"

class MessageCreate(MessageBase):
    chatroom_id: int

class MessageRead(MessageBase):
    id: int
    chatroom_id: int
    sender_id: int
    original_audio_path: Optional[str] = None
    translations_cache: Optional[Dict[str, Any]] = None
    timestamp: datetime
    class Config:
        from_attributes = True
