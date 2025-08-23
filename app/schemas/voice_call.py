# filepath: app/schemas/voice_call.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

# Request Schemas

class CallInitiateRequest(BaseModel):
    """Request to initiate a voice call"""
    callee_id: int = Field(..., gt=0, description="ID of the user to call")
    call_type: str = Field(default="voice", description="Type of call (voice/video)")
    message: Optional[str] = Field(None, max_length=500, description="Optional initial message")

class CallAnswerRequest(BaseModel):
    """Request to answer an incoming call"""
    sdp_answer: Optional[str] = Field(None, description="WebRTC SDP answer")
    user_language: str = Field(default="en", description="User's preferred language")

class CallEndRequest(BaseModel):
    """Request to end a call"""
    end_reason: Optional[str] = Field(default="normal", description="Reason for ending call")
    quality_score: Optional[int] = Field(None, ge=1, le=5, description="Call quality rating 1-5")
    connection_quality: Optional[str] = Field(None, description="Connection quality assessment")

class CallMessageRequest(BaseModel):
    """Request to send a message during a call"""
    message_text: str = Field(..., min_length=1, max_length=1000, description="Message content")
    language: str = Field(default="en", description="Message language")
    auto_translate: bool = Field(default=True, description="Enable auto-translation")
    message_type: str = Field(default="text", description="Type of message")

# Response Schemas

class CallInitiateResponse(BaseModel):
    """Response when initiating a call"""
    call_id: str = Field(..., description="Unique call identifier")
    status: str = Field(..., description="Call status")
    caller_id: int = Field(..., description="ID of the caller")
    callee_id: int = Field(..., description="ID of the callee")
    message: str = Field(..., description="Response message")
    webrtc_config: Optional[Dict[str, Any]] = Field(None, description="WebRTC configuration")

class CallActionResponse(BaseModel):
    """Generic response for call actions"""
    success: bool = Field(..., description="Whether the action was successful")
    call_id: str = Field(..., description="Call identifier")
    status: str = Field(..., description="Current call status")
    message: str = Field(..., description="Response message")
    duration: Optional[float] = Field(None, description="Call duration in seconds")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")

class MessageResponse(BaseModel):
    """Response for message operations"""
    message_id: int = Field(..., description="Message identifier")
    success: bool = Field(..., description="Whether the message was sent successfully")
    message: str = Field(..., description="Response message")
    translated: Optional[Dict[str, str]] = Field(None, description="Translated versions")

class CallHistoryResponse(BaseModel):
    """Response for call history"""
    call_id: str = Field(..., description="Call identifier")
    other_user_id: int = Field(..., description="Other participant's ID")
    other_user_name: str = Field(..., description="Other participant's name")
    call_type: str = Field(..., description="Type of call")
    status: str = Field(..., description="Call status")
    duration: Optional[float] = Field(None, description="Call duration in seconds")
    started_at: datetime = Field(..., description="Call start time")
    ended_at: Optional[datetime] = Field(None, description="Call end time")
    quality_score: Optional[int] = Field(None, description="Call quality rating")

class CallMessageResponse(BaseModel):
    """Response for call messages"""
    message_id: int = Field(..., description="Message identifier")
    sender_id: int = Field(..., description="Sender's ID")
    sender_name: str = Field(..., description="Sender's name")
    message_text: str = Field(..., description="Message content")
    original_language: str = Field(..., description="Original message language")
    translated_content: Optional[Dict[str, str]] = Field(None, description="Translated versions")
    sent_at: datetime = Field(..., description="Message timestamp")
    message_type: str = Field(..., description="Type of message")

class ActiveCallResponse(BaseModel):
    """Response for active calls"""
    call_id: str = Field(..., description="Call identifier")
    other_user_id: int = Field(..., description="Other participant's ID")
    other_user_name: str = Field(..., description="Other participant's name")
    status: str = Field(..., description="Call status")
    started_at: datetime = Field(..., description="Call start time")
    is_outgoing: bool = Field(..., description="Whether this is an outgoing call")
    duration: Optional[float] = Field(None, description="Current call duration")

# WebSocket Message Schemas

class WebSocketMessage(BaseModel):
    """Base WebSocket message schema"""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")

class CallNotificationMessage(WebSocketMessage):
    """Call notification via WebSocket"""
    type: str = Field(default="call_notification", description="Message type")
    call_id: str = Field(..., description="Call identifier")
    caller_id: int = Field(..., description="Caller's ID")
    caller_name: str = Field(..., description="Caller's name")
    call_type: str = Field(default="voice", description="Type of call")

class CallStatusMessage(WebSocketMessage):
    """Call status update via WebSocket"""
    type: str = Field(default="call_status", description="Message type")
    call_id: str = Field(..., description="Call identifier")
    status: str = Field(..., description="New call status")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional status data")

class CallSignalingMessage(WebSocketMessage):
    """WebRTC signaling message"""
    type: str = Field(default="webrtc_signaling", description="Message type")
    call_id: str = Field(..., description="Call identifier")
    signaling_type: str = Field(..., description="Type of signaling (offer/answer/ice)")
    sdp: Optional[str] = Field(None, description="SDP data")
    candidate: Optional[Dict[str, Any]] = Field(None, description="ICE candidate data")

class CallMessageWebSocket(WebSocketMessage):
    """Call message via WebSocket"""
    type: str = Field(default="call_message", description="Message type")
    call_id: str = Field(..., description="Call identifier")
    message_id: int = Field(..., description="Message identifier")
    sender_id: int = Field(..., description="Sender's ID")
    sender_name: str = Field(..., description="Sender's name")
    message_text: str = Field(..., description="Message content")
    original_language: str = Field(..., description="Original language")
    translated_content: Optional[Dict[str, str]] = Field(None, description="Translations")

# Validation schemas

class CallParticipantInfo(BaseModel):
    """Call participant information"""
    user_id: int = Field(..., description="User ID")
    user_name: str = Field(..., description="User display name")
    language_preference: str = Field(default="en", description="Preferred language")
    is_online: bool = Field(..., description="Online status")
    last_seen: Optional[datetime] = Field(None, description="Last seen timestamp")

class CallQualityMetrics(BaseModel):
    """Call quality metrics"""
    audio_quality: Optional[int] = Field(None, ge=1, le=5, description="Audio quality 1-5")
    connection_stability: Optional[str] = Field(None, description="Connection stability")
    latency_ms: Optional[int] = Field(None, ge=0, description="Latency in milliseconds")
    packet_loss: Optional[float] = Field(None, ge=0, le=100, description="Packet loss percentage")
    jitter_ms: Optional[int] = Field(None, ge=0, description="Jitter in milliseconds")

class WebRTCConfiguration(BaseModel):
    """WebRTC configuration for calls"""
    ice_servers: List[Dict[str, Any]] = Field(default_factory=list, description="ICE servers")
    audio_constraints: Dict[str, Any] = Field(default_factory=dict, description="Audio constraints")
    video_constraints: Optional[Dict[str, Any]] = Field(None, description="Video constraints")
    rtc_configuration: Dict[str, Any] = Field(default_factory=dict, description="RTC configuration")

# Validators

class CallInitiateRequest(CallInitiateRequest):
    @validator('call_type')
    def validate_call_type(cls, v):
        if v not in ['voice', 'video']:
            raise ValueError('call_type must be either "voice" or "video"')
        return v

class CallMessageRequest(CallMessageRequest):
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ['en', 'fr', 'ar', 'es', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in supported_languages:
            raise ValueError(f'language must be one of: {", ".join(supported_languages)}')
        return v
    
    @validator('message_type')
    def validate_message_type(cls, v):
        if v not in ['text', 'emoji', 'system']:
            raise ValueError('message_type must be "text", "emoji", or "system"')
        return v

class CallAnswerRequest(CallAnswerRequest):
    @validator('user_language')
    def validate_user_language(cls, v):
        supported_languages = ['en', 'fr', 'ar', 'es', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in supported_languages:
            raise ValueError(f'user_language must be one of: {", ".join(supported_languages)}')
        return v
