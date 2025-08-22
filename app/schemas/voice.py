# filepath: app/schemas/voice.py
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, Dict, List, Any

SUPPORTED_LANGUAGES = ['en', 'fr', 'ar']

class AudioFileBase(BaseModel):
    language: str = Field(..., description="Language code (en, fr, ar)")
    file_path: str
    file_url: str
    file_size: int
    duration: Optional[float] = None
    mime_type: str = "audio/mp3"
    
    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Language must be one of {SUPPORTED_LANGUAGES}')
        return v

class AudioFileCreate(AudioFileBase):
    message_id: int

class AudioFileRead(AudioFileBase):
    id: int
    message_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class VoiceMessageUpload(BaseModel):
    language: str = Field(default="en", description="Language code (en, fr, ar)")
    recipient_id: int
    
    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Language must be one of {SUPPORTED_LANGUAGES}')
        return v

class VoiceMessageProcess(BaseModel):
    transcribed_text: str
    original_language: str
    translations: Dict[str, str]
    audio_urls: Dict[str, str]
    duration: Optional[float] = None
    file_size: Optional[int] = None

class VoiceMessageResponse(BaseModel):
    success: bool
    message_id: int
    transcribed_text: str
    translations: Dict[str, str]
    audio_urls: Dict[str, str]
    audio_duration: Optional[float] = None
    error: Optional[str] = None

class VoiceMessageWebSocketData(BaseModel):
    type: str = "voice_message"
    message_type: str = "voice"
    sender: Dict[str, Any]
    original_text: str
    original_language: str
    translations_cache: Optional[Dict[str, str]] = None
    audio_urls: Optional[Dict[str, str]] = None
    audio_duration: Optional[float] = None
    timestamp: str

class VoiceChatStats(BaseModel):
    total_voice_messages: int
    total_voice_duration: float  # in seconds
    languages_used: List[str]
    average_message_duration: float
    largest_file_size: int  # in bytes
