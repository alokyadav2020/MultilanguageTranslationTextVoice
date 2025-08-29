# from sqlalchemy import String, Integer, DateTime
# from sqlalchemy.orm import Mapped, mapped_column
# from datetime import datetime
# from ..core.database import Base

# class User(Base):
#     __tablename__ = "users"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
#     full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
#     hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
#     is_active: Mapped[bool] = mapped_column(default=True)
#     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, text, func
from sqlalchemy.orm import relationship
from ..core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)          # user_id
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default=text('1'))  # restore to match existing table
    preferred_language = Column(String(10), nullable=False, default="en")  # NEW
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    messages = relationship("Message", back_populates="sender", cascade="all,delete")
    memberships = relationship("ChatroomMember", back_populates="user", cascade="all,delete")
    
    # Group relationships
    created_groups = relationship("Group", back_populates="creator")
    groups = relationship("Group", secondary="group_members", back_populates="members")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

    def get_voice_message_count(self, db_session) -> int:
        """Get count of voice messages sent by user"""
        from .message import Message, MessageType
        return db_session.query(Message).filter(
            Message.sender_id == self.id,
            Message.message_type == MessageType.voice
        ).count()
    
    def get_total_voice_duration(self, db_session) -> float:
        """Get total duration of voice messages sent by user"""
        from .message import Message, MessageType
        from sqlalchemy import func
        
        result = db_session.query(func.sum(Message.audio_duration)).filter(
            Message.sender_id == self.id,
            Message.message_type == MessageType.voice
        ).scalar()
        
        return result or 0.0
    
    def get_groups(self, db):
        """Get all groups this user is a member of"""
        from .group import Group, group_members
        return db.query(Group).join(group_members).filter(
            group_members.c.user_id == self.id,
            Group.is_active
        ).all()
    
    def get_group_role(self, group_id, db):
        """Get user's role in a specific group"""
        from .group import group_members
        result = db.query(group_members).filter(
            group_members.c.group_id == group_id,
            group_members.c.user_id == self.id
        ).first()
        return result.role if result else None
