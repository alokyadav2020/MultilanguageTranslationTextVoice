from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import relationship
from ..core.database import Base

class Chatroom(Base):
    __tablename__ = "chatrooms"

    id = Column(Integer, primary_key=True, index=True)  # chatroom_id
    chatroom_name = Column(String(100), nullable=True)
    is_group_chat = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    members = relationship("ChatroomMember", back_populates="chatroom", cascade="all,delete")
    messages = relationship("Message", back_populates="chatroom", cascade="all,delete")

    def __repr__(self):
        return f"<Chatroom(id={self.id}, name={self.chatroom_name}, is_group={self.is_group_chat})>"

    def get_voice_message_count(self, db_session) -> int:
        """Get count of voice messages in this chatroom"""
        from .message import Message, MessageType
        return db_session.query(Message).filter(
            Message.chatroom_id == self.id,
            Message.message_type == MessageType.voice
        ).count()
    
    def get_total_message_count(self, db_session) -> int:
        """Get total message count in this chatroom"""
        from .message import Message
        return db_session.query(Message).filter(
            Message.chatroom_id == self.id
        ).count()