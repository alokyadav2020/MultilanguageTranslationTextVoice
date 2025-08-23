# filepath: app/models/voice_call.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from ..core.database import Base

class CallStatus(str, enum.Enum):
    INITIATED = "initiated"
    RINGING = "ringing"
    ANSWERED = "answered"
    ENDED = "ended"
    DECLINED = "declined"
    MISSED = "missed"
    BUSY = "busy"

class CallType(str, enum.Enum):
    VOICE = "voice"
    VIDEO = "video"  # For future implementation

class VoiceCall(Base):
    """Voice call sessions between users"""
    __tablename__ = "voice_calls"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String(100), unique=True, index=True)  # Unique call identifier
    
    # Participants
    caller_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    callee_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Call details
    call_type = Column(Enum(CallType), default=CallType.VOICE)
    status = Column(Enum(CallStatus), default=CallStatus.INITIATED, index=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # WebRTC signaling data
    caller_offer = Column(Text, nullable=True)  # SDP offer from caller
    callee_answer = Column(Text, nullable=True)  # SDP answer from callee
    ice_candidates = Column(Text, nullable=True)  # JSON array of ICE candidates
    
    # Call quality metrics
    audio_quality_score = Column(Float, nullable=True)  # 1.0 to 5.0
    connection_quality = Column(String(20), nullable=True)  # good/fair/poor
    end_reason = Column(String(50), nullable=True)  # normal/timeout/error/declined
    
    # Additional metadata
    is_encrypted = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    caller = relationship("User", foreign_keys=[caller_id], backref="outgoing_calls")
    callee = relationship("User", foreign_keys=[callee_id], backref="incoming_calls")
    chat_messages = relationship("VoiceCallMessage", back_populates="call", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<VoiceCall(id={self.id}, caller={self.caller_id}, callee={self.callee_id}, status={self.status})>"
    
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration (mm:ss)"""
        if not self.duration_seconds:
            return "0:00"
        
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    @property
    def is_active(self) -> bool:
        """Check if call is currently active"""
        return self.status in [CallStatus.INITIATED, CallStatus.RINGING, CallStatus.ANSWERED]

class VoiceCallMessage(Base):
    """Text messages exchanged during voice calls"""
    __tablename__ = "voice_call_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("voice_calls.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message content
    message_text = Column(Text, nullable=False)
    original_language = Column(String(10), default="en")
    translated_content = Column(Text, nullable=True)  # JSON: {"fr": "...", "ar": "..."}
    
    # Message type
    message_type = Column(String(20), default="text")  # text/system/call_event
    
    # Timing
    sent_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    call = relationship("VoiceCall", back_populates="chat_messages")
    sender = relationship("User", backref="call_messages")
    
    def __repr__(self):
        return f"<VoiceCallMessage(id={self.id}, call={self.call_id}, sender={self.sender_id})>"

class VoiceCallParticipant(Base):
    """Track participants in voice calls (for future group calls)"""
    __tablename__ = "voice_call_participants"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("voice_calls.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Participant details
    preferred_language = Column(String(10), default="en")
    is_moderator = Column(Boolean, default=False)
    
    # Connection status
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    left_at = Column(DateTime(timezone=True), nullable=True)
    connection_quality = Column(String(20), nullable=True)
    
    # Audio settings
    is_muted = Column(Boolean, default=False)
    audio_enabled = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", backref="call_participations")
    
    def __repr__(self):
        return f"<VoiceCallParticipant(call={self.call_id}, user={self.user_id})>"
