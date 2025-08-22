
# filepath: app/schemas/chat.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List
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
    original_language: str = "en"
    message_type: str = "text"

class MessageCreate(MessageBase):
    chatroom_id: int

class MessageRead(MessageBase):
    id: int
    chatroom_id: int
    sender_id: int
    original_audio_path: Optional[str] = None
    # NEW VOICE FIELDS
    audio_urls: Optional[Dict[str, str]] = None  # {language: audio_url}
    audio_duration: Optional[float] = None
    audio_file_size: Optional[int] = None
    translations_cache: Optional[Dict[str, Any]] = None
    timestamp: datetime
    class Config:
        from_attributes = True

# NEW SCHEMAS FOR VOICE MESSAGES
class VoiceMessageCreate(BaseModel):
    chatroom_id: int
    language: str = "en"
    # File will be handled separately in FastAPI

class VoiceMessageResponse(BaseModel):
    success: bool
    message_id: int
    transcribed_text: str
    translations: Dict[str, str]
    audio_urls: Dict[str, str]
    audio_duration: Optional[float] = None

class ChatHistoryMessage(BaseModel):
    sender: Dict[str, Any]  # {id, name, email}
    original_text: str
    original_language: str
    message_type: str
    translations_cache: Optional[Dict[str, str]] = None
    timestamp: str
    # Voice-specific fields
    audio_urls: Optional[Dict[str, str]] = None
    audio_duration: Optional[float] = None
