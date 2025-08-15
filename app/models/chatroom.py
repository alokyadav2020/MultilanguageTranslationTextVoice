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