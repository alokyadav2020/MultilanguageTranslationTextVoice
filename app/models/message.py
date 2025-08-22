
# filepath: app/models/message.py
from sqlalchemy import (
    Column, Integer, Text, String, DateTime, func, ForeignKey, Enum, JSON, Float
)
from sqlalchemy.orm import relationship
from ..core.database import Base
from enum import Enum as PyEnum

class MessageType(str, PyEnum):
    text = "text"
    voice = "voice"

class ChatroomMember(Base):
    __tablename__ = "chatroom_members"

    id = Column(Integer, primary_key=True, index=True)
    chatroom_id = Column(Integer, ForeignKey("chatrooms.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    chatroom = relationship("Chatroom", back_populates="members")
    user = relationship("User", back_populates="memberships")

    __table_args__ = (
        # Unique (chatroom_id, user_id)
        # For SQLite, SQLAlchemy will emulate uniqueness.
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)  # message_id
    chatroom_id = Column(Integer, ForeignKey("chatrooms.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_text = Column(Text, nullable=False)
    original_language = Column(String(10), nullable=False, default="en", index=True)  # NEW
    message_type = Column(Enum(MessageType), nullable=False, default=MessageType.text, index=True)
    
    # Voice-specific fields
    original_audio_path = Column(String(255), nullable=True)  # Path to original uploaded audio
    audio_urls = Column(JSON, nullable=True)  # {"en": "/static/voice/file1.mp3", "fr": "/static/voice/file2.mp3"}
    audio_duration = Column(Float, nullable=True)  # Duration in seconds
    audio_file_size = Column(Integer, nullable=True)  # File size in bytes
    
    # Translation and metadata
    translations_cache = Column(JSON, nullable=True)  # {"ar": "...", "fr": "..."}
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    chatroom = relationship("Chatroom", back_populates="messages")
    sender = relationship("User", back_populates="messages")
    audio_files = relationship("AudioFile", back_populates="message", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Message(id={self.id}, type={self.message_type}, sender_id={self.sender_id})>"
    
    @property
    def is_voice_message(self) -> bool:
        """Check if this is a voice message"""
        return self.message_type == MessageType.voice
    
    def get_audio_url_for_language(self, language: str) -> str:
        """Get audio URL for specific language"""
        if not self.audio_urls:
            return None
        return self.audio_urls.get(language) or self.audio_urls.get(self.original_language)
    
    def get_translated_text(self, language: str) -> str:
        """Get translated text for specific language"""
        if language == self.original_language:
            return self.original_text
        
        if self.translations_cache and language in self.translations_cache:
            return self.translations_cache[language]
        
        return self.original_text  # Fallback to original
