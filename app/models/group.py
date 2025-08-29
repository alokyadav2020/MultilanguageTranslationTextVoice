# filepath: app/models/group.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Table, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from ..core.database import Base

# Association table for group members
group_members = Table(
    'group_members',
    Base.metadata,
    Column('group_id', Integer, ForeignKey('groups.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role', String(20), default='member'),  # 'admin', 'moderator', 'member'
    Column('joined_at', DateTime, default=datetime.utcnow),
    Column('preferred_language', String(10), default='en'),  # Language preference for this group
    Column('notifications_enabled', Boolean, default=True),
    Column('voice_language', String(10), default='en')  # Voice language preference
)

class GroupType(enum.Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    ANNOUNCEMENT = "announcement"  # Only admins can post

class Group(Base):
    __tablename__ = 'groups'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    group_type = Column(Enum(GroupType), default=GroupType.PRIVATE)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    max_members = Column(Integer, default=100)
    default_language = Column(String(10), default='en')
    profile_picture = Column(String(255), nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="created_groups")
    members = relationship("User", secondary=group_members, back_populates="groups")
    messages = relationship("GroupMessage", back_populates="group", cascade="all, delete-orphan")
    
    def get_member_count(self, db):
        """Get total number of active members"""
        return db.query(group_members).filter(group_members.c.group_id == self.id).count()
    
    def get_member_role(self, user_id, db):
        """Get user's role in this group"""
        result = db.query(group_members).filter(
            group_members.c.group_id == self.id,
            group_members.c.user_id == user_id
        ).first()
        return result.role if result else None
    
    def get_member_language(self, user_id, db):
        """Get user's preferred language for this group"""
        result = db.query(group_members).filter(
            group_members.c.group_id == self.id,
            group_members.c.user_id == user_id
        ).first()
        return result.preferred_language if result else self.default_language
    
    def get_member_voice_language(self, user_id, db):
        """Get user's preferred voice language for this group"""
        result = db.query(group_members).filter(
            group_members.c.group_id == self.id,
            group_members.c.user_id == user_id
        ).first()
        return result.voice_language if result else self.default_language

class GroupMessage(Base):
    __tablename__ = 'group_messages'
    
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey('groups.id'), nullable=False)
    sender_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    content = Column(Text, nullable=False)
    original_language = Column(String(10), nullable=False)
    message_type = Column(String(20), default='text')  # 'text', 'voice', 'image', 'file', 'announcement'
    reply_to_id = Column(Integer, ForeignKey('group_messages.id'), nullable=True)  # For threading
    voice_file_path = Column(String(255), nullable=True)  # For voice messages
    voice_duration = Column(Integer, nullable=True)  # Voice duration in seconds
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_edited = Column(Boolean, default=False)
    edited_at = Column(DateTime, nullable=True)
    
    # Relationships
    group = relationship("Group", back_populates="messages")
    sender = relationship("User")
    reply_to = relationship("GroupMessage", remote_side=[id])
    translations = relationship("GroupMessageTranslation", back_populates="message", cascade="all, delete-orphan")
    reactions = relationship("MessageReaction", back_populates="group_message", cascade="all, delete-orphan")

class GroupMessageTranslation(Base):
    __tablename__ = 'group_message_translations'
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey('group_messages.id'), nullable=False)
    language = Column(String(10), nullable=False)
    translated_content = Column(Text, nullable=False)
    translation_type = Column(String(10), default='text')  # 'text', 'voice'
    voice_file_path = Column(String(255), nullable=True)  # For translated voice
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    message = relationship("GroupMessage", back_populates="translations")

class MessageReaction(Base):
    __tablename__ = 'message_reactions'
    
    id = Column(Integer, primary_key=True, index=True)
    group_message_id = Column(Integer, ForeignKey('group_messages.id'), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    reaction = Column(String(10), nullable=False)  # emoji like '👍', '❤️', etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    group_message = relationship("GroupMessage", back_populates="reactions")
    user = relationship("User")
