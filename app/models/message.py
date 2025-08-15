
# filepath: app/models/message.py
from sqlalchemy import (
    Column, Integer, Text, String, DateTime, func, ForeignKey, Enum, JSON
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
    chatroom_id = Column(Integer, ForeignKey("chatrooms.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    original_text = Column(Text, nullable=False)
    original_language = Column(String(10), nullable=False, default="en")  # NEW
    message_type = Column(Enum(MessageType), nullable=False, default=MessageType.text)
    original_audio_path = Column(String(255), nullable=True)
    translations_cache = Column(JSON, nullable=True)  # {"ar": "...", "fr": "..."}
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    chatroom = relationship("Chatroom", back_populates="messages")
    sender = relationship("User", back_populates="messages")
