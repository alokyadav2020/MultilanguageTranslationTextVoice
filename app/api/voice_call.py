# filepath: app/api/voice_call.py
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import json
import asyncio
import uuid
import logging
from datetime import datetime, timedelta

from ..core.database import get_db
from ..api.deps import get_current_user, get_user_from_websocket_token
from ..models.voice_call import VoiceCall, VoiceCallMessage, VoiceCallParticipant, CallStatus, CallType
from ..models.user import User
from ..schemas.voice_call import (
    CallInitiateRequest, CallInitiateResponse, CallAnswerRequest, CallActionResponse,
    CallEndRequest, CallMessageRequest, MessageResponse, CallHistoryResponse,
    CallMessageResponse, ActiveCallResponse
)
from ..services.call_manager import call_manager
from ..services.translation import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice-call", tags=["voice-calls"])

@router.post("/initiate", response_model=CallInitiateResponse)
async def initiate_call(
    call_request: CallInitiateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Initiate a voice call to another user"""
    
    # Validate callee exists
    callee = db.query(User).filter(User.id == call_request.callee_id).first()
    if not callee:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent self-calling
    if call_request.callee_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot call yourself")
    
    # Clean up any stale calls first (calls that are older than 5 minutes and not answered)
    stale_cutoff = datetime.utcnow() - timedelta(minutes=5)
    stale_calls = db.query(VoiceCall).filter(
        (VoiceCall.status.in_([CallStatus.INITIATED, CallStatus.RINGING])) &
        (VoiceCall.created_at < stale_cutoff)
    ).all()
    
    for stale_call in stale_calls:
        stale_call.status = CallStatus.ENDED
        stale_call.ended_at = datetime.utcnow()
        await call_manager.cleanup_call(stale_call.call_id)
    
    if stale_calls:
        db.commit()
        logger.info(f"📞 Cleaned up {len(stale_calls)} stale calls")
    
    # Check if either user is already in a call
    ongoing_caller = db.query(VoiceCall).filter(
        ((VoiceCall.caller_id == current_user.id) | (VoiceCall.callee_id == current_user.id)) &
        (VoiceCall.status.in_([CallStatus.INITIATED, CallStatus.RINGING, CallStatus.ANSWERED]))
    ).first()
    
    ongoing_callee = db.query(VoiceCall).filter(
        ((VoiceCall.caller_id == call_request.callee_id) | (VoiceCall.callee_id == call_request.callee_id)) &
        (VoiceCall.status.in_([CallStatus.INITIATED, CallStatus.RINGING, CallStatus.ANSWERED]))
    ).first()
    
    if ongoing_caller:
        raise HTTPException(status_code=409, detail="You are already in a call")
    
    if ongoing_callee:
        raise HTTPException(status_code=409, detail="User is busy in another call")
    
    # Note: We don't check if user is online since they might be online via chat
    # and don't have a voice call WebSocket connection yet
    
    # Generate unique call ID
    call_id = f"call_{uuid.uuid4().hex[:12]}_{current_user.id}_{call_request.callee_id}"
    
    # Create call record
    voice_call = VoiceCall(
        call_id=call_id,
        caller_id=current_user.id,
        callee_id=call_request.callee_id,
        call_type=CallType.VOICE,
        status=CallStatus.INITIATED
    )
    
    db.add(voice_call)
    db.commit()
    db.refresh(voice_call)
    
    # Add to call manager
    await call_manager.register_call(voice_call.id, call_id, current_user.id, call_request.callee_id)
    
    # Send call_initiated message to caller (to open their call window)
    background_tasks.add_task(
        call_manager.send_call_initiated,
        call_id=call_id,
        caller_id=current_user.id
    )
    
    # Send call notification to callee asynchronously
    background_tasks.add_task(
        call_manager.send_call_notification,
        call_id=call_id,
        caller_id=current_user.id,
        callee_id=call_request.callee_id,
        caller_name=current_user.email  # Use display_name if available
    )
    
    return CallInitiateResponse(
        call_id=call_id,
        status="initiated",
        caller_id=current_user.id,
        callee_id=call_request.callee_id,
        message=f"Call initiated to {callee.email}"
    )

@router.post("/answer/{call_id}", response_model=CallActionResponse)
async def answer_call(
    call_id: str,
    answer_data: CallAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Answer an incoming call"""
    
    call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call.callee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to answer this call")
    
    if call.status not in [CallStatus.INITIATED, CallStatus.RINGING]:
        raise HTTPException(status_code=400, detail="Call cannot be answered")
    
    # Update call status
    call.status = CallStatus.ANSWERED
    call.answered_at = datetime.utcnow()
    if answer_data.sdp_answer:
        call.callee_answer = answer_data.sdp_answer
    
    db.commit()
    
    # Notify caller that call was answered
    await call_manager.send_call_answered(
        call_id=call_id,
        caller_id=call.caller_id,
        sdp_answer=answer_data.sdp_answer
    )
    
    return CallActionResponse(
        success=True,
        call_id=call_id,
        status="answered",
        message="Call answered successfully"
    )

@router.post("/decline/{call_id}", response_model=CallActionResponse)
async def decline_call(
    call_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Decline an incoming call"""
    
    call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call.callee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to decline this call")
    
    # Update call status
    call.status = CallStatus.DECLINED
    call.ended_at = datetime.utcnow()
    call.end_reason = "declined"
    
    db.commit()
    
    # Notify caller and clean up
    await call_manager.send_call_declined(call_id, call.caller_id)
    await call_manager.cleanup_call(call_id)
    
    return CallActionResponse(
        success=True,
        call_id=call_id,
        status="declined",
        message="Call declined"
    )

@router.post("/end/{call_id}", response_model=CallActionResponse)
async def end_call(
    call_id: str,
    end_data: CallEndRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End an ongoing call"""
    
    call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Verify user is participant
    if call.caller_id != current_user.id and call.callee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to end this call")
    
    # Calculate duration if call was answered
    duration = None
    if call.answered_at:
        duration = (datetime.utcnow() - call.answered_at).total_seconds()
    
    # Update call record
    call.status = CallStatus.ENDED
    call.ended_at = datetime.utcnow()
    call.duration_seconds = duration
    call.end_reason = end_data.end_reason or "normal"
    
    if end_data.quality_score:
        call.audio_quality_score = end_data.quality_score
    if end_data.connection_quality:
        call.connection_quality = end_data.connection_quality
    
    db.commit()
    
    # Notify other participant and clean up
    other_user_id = call.callee_id if call.caller_id == current_user.id else call.caller_id
    await call_manager.send_call_ended(call_id, other_user_id)
    await call_manager.cleanup_call(call_id)
    
    return CallActionResponse(
        success=True,
        call_id=call_id,
        status="ended",
        duration=duration,
        message="Call ended successfully"
    )

@router.post("/force-cleanup", response_model=CallActionResponse)
async def force_cleanup_user_calls(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Force cleanup all active calls for the current user (for debugging)"""
    
    # Find all active calls for the user
    active_calls = db.query(VoiceCall).filter(
        ((VoiceCall.caller_id == current_user.id) | (VoiceCall.callee_id == current_user.id)) &
        (VoiceCall.status.in_([CallStatus.INITIATED, CallStatus.RINGING, CallStatus.ANSWERED]))
    ).all()
    
    cleanup_count = 0
    for call in active_calls:
        call.status = CallStatus.ENDED
        call.ended_at = datetime.utcnow()
        call.end_reason = "force_cleanup"
        await call_manager.cleanup_call(call.call_id)
        cleanup_count += 1
    
    if cleanup_count > 0:
        db.commit()
        logger.info(f"📞 Force cleaned up {cleanup_count} calls for user {current_user.id}")
    
    return CallActionResponse(
        success=True,
        call_id="",
        status="cleaned",
        message=f"Cleaned up {cleanup_count} active calls"
    )

@router.post("/send-message/{call_id}", response_model=MessageResponse)
async def send_call_message(
    call_id: str,
    message_data: CallMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a text message during a voice call"""
    
    call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Verify user is participant
    if call.caller_id != current_user.id and call.callee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to send messages in this call")
    
    # Translate message if needed
    translated_content = None
    if message_data.auto_translate:
        try:
            # Get supported languages for translation
            target_languages = ['en', 'fr', 'ar']  # Your supported languages
            
            translations = {}
            for lang in target_languages:
                if lang != message_data.language:
                    translated = await translation_service.translate_text_async(
                        message_data.message_text,
                        source_lang=message_data.language,
                        target_lang=lang
                    )
                    translations[lang] = translated
            
            translated_content = json.dumps(translations) if translations else None
            
        except Exception as e:
            print(f"Translation failed: {e}")
    
    # Create message record
    call_message = VoiceCallMessage(
        call_id=call.id,
        sender_id=current_user.id,
        message_text=message_data.message_text,
        original_language=message_data.language,
        translated_content=translated_content,
        message_type="text"
    )
    
    db.add(call_message)
    db.commit()
    db.refresh(call_message)
    
    # Send message to other participant via WebSocket
    await call_manager.send_call_message(call_id, {
        "message_id": call_message.id,
        "sender_id": current_user.id,
        "sender_name": current_user.email,
        "message_text": message_data.message_text,
        "original_language": message_data.language,
        "translated_content": json.loads(translated_content) if translated_content else None,
        "sent_at": call_message.sent_at.isoformat(),
        "message_type": "text"
    })
    
    return MessageResponse(
        message_id=call_message.id,
        success=True,
        message="Message sent successfully"
    )

@router.get("/history", response_model=List[CallHistoryResponse])
async def get_call_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get user's call history"""
    
    calls = db.query(VoiceCall).filter(
        (VoiceCall.caller_id == current_user.id) | (VoiceCall.callee_id == current_user.id)
    ).order_by(VoiceCall.started_at.desc()).offset(offset).limit(limit).all()
    
    history = []
    for call in calls:
        # Determine if it's incoming or outgoing
        is_outgoing = call.caller_id == current_user.id
        other_user_id = call.callee_id if is_outgoing else call.caller_id
        
        # Get other user info
        other_user = db.query(User).filter(User.id == other_user_id).first()
        
        history.append(CallHistoryResponse(
            call_id=call.call_id,
            other_user_id=other_user_id,
            other_user_name=other_user.email if other_user else "Unknown",
            call_type="outgoing" if is_outgoing else "incoming",
            status=call.status,
            duration=call.duration_seconds,
            started_at=call.started_at,
            ended_at=call.ended_at,
            quality_score=call.audio_quality_score
        ))
    
    return history

@router.get("/messages/{call_id}", response_model=List[CallMessageResponse])
async def get_call_messages(
    call_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50
):
    """Get messages from a specific call"""
    
    call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Verify user is participant
    if call.caller_id != current_user.id and call.callee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view these messages")
    
    messages = db.query(VoiceCallMessage).filter(
        VoiceCallMessage.call_id == call.id
    ).order_by(VoiceCallMessage.sent_at.asc()).limit(limit).all()
    
    response_messages = []
    for msg in messages:
        sender = db.query(User).filter(User.id == msg.sender_id).first()
        
        translated_content = None
        if msg.translated_content:
            try:
                translated_content = json.loads(msg.translated_content)
            except Exception:
                pass
        
        response_messages.append(CallMessageResponse(
            message_id=msg.id,
            sender_id=msg.sender_id,
            sender_name=sender.email if sender else "Unknown",
            message_text=msg.message_text,
            original_language=msg.original_language,
            translated_content=translated_content,
            sent_at=msg.sent_at,
            message_type=msg.message_type
        ))
    
    return response_messages

# WebSocket endpoint for real-time call signaling and messaging
@router.websocket("/ws/{call_id}")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_id: str,
    token: str,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time call communication"""
    
    try:
        # Authenticate user
        user = await get_user_from_websocket_token(token, db)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Verify call exists and user is participant
        call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
        if not call or (call.caller_id != user.id and call.callee_id != user.id):
            await websocket.close(code=1008, reason="Call not found or unauthorized")
            return
        
        await websocket.accept()
        print(f"📞 WebSocket connected for call {call_id}, user {user.id}")
        
        # Add to call manager
        await call_manager.add_call_connection(call_id, user.id, websocket)
        
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                message_type = message.get('type')
                
                if message_type == 'join_call':
                    # User joined the call
                    print(f"📞 User {user.id} joined call {call_id}")
                    await websocket.send_text(json.dumps({
                        'type': 'join_acknowledged',
                        'call_id': call_id,
                        'user_id': user.id
                    }))
                
                elif message_type == 'webrtc_signaling':
                    # Forward WebRTC signaling data
                    await call_manager.forward_signaling_message(call_id, user.id, message)
                
                elif message_type == 'ice_candidate':
                    # Forward ICE candidates
                    await call_manager.forward_ice_candidate(call_id, user.id, message)
                
                elif message_type == 'call_status':
                    # Update call status (muted, etc.)
                    await call_manager.broadcast_call_status(call_id, user.id, message)
                
                elif message_type == 'heartbeat':
                    # Keep connection alive
                    await websocket.send_text(json.dumps({'type': 'heartbeat_ack'}))
                
                elif message_type == 'translation_settings':
                    # Forward translation settings to other participant
                    print(f"🌐 Translation settings from user {user.id}: {message}")
                    await call_manager.forward_translation_message(call_id, user.id, message)
                
                elif message_type == 'translation_state':
                    # Forward translation state changes to other participant
                    print(f"🌐 Translation state from user {user.id}: {message}")
                    await call_manager.forward_translation_message(call_id, user.id, message)
                
                elif message_type == 'translated_audio':
                    # Forward translated audio to other participant
                    print(f"🌐 Translated audio from user {user.id}")
                    await call_manager.forward_translation_message(call_id, user.id, message)
                
                else:
                    print(f"❓ Unknown message type: {message_type}")
                    # Still try to forward unknown message types
                    await call_manager.forward_signaling_message(call_id, user.id, message)
                
        except WebSocketDisconnect:
            print(f"📞 WebSocket disconnected for call {call_id}, user {user.id}")
        except Exception as e:
            print(f"❌ WebSocket error in call {call_id}: {e}")
    
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
        try:
            await websocket.close(code=1008, reason="Connection error")
        except Exception:
            pass
    
    finally:
        # Clean up connection
        try:
            await call_manager.remove_call_connection(call_id, user.id)
        except Exception:
            pass

@router.get("/active-calls", response_model=List[ActiveCallResponse])
async def get_active_calls(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's currently active calls"""
    
    active_calls = db.query(VoiceCall).filter(
        ((VoiceCall.caller_id == current_user.id) | (VoiceCall.callee_id == current_user.id)) &
        (VoiceCall.status.in_([CallStatus.INITIATED, CallStatus.RINGING, CallStatus.ANSWERED]))
    ).all()
    
    response = []
    for call in active_calls:
        other_user_id = call.callee_id if call.caller_id == current_user.id else call.caller_id
        other_user = db.query(User).filter(User.id == other_user_id).first()
        
        response.append(ActiveCallResponse(
            call_id=call.call_id,
            other_user_id=other_user_id,
            other_user_name=other_user.email if other_user else "Unknown",
            status=call.status,
            started_at=call.started_at,
            is_outgoing=call.caller_id == current_user.id
        ))
    
    return response

# WebSocket endpoint for general voice call notifications (dashboard, incoming calls)
@router.websocket("/ws")
async def websocket_voice_call_notifications(
    websocket: WebSocket,
    token: str = None,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for voice call notifications"""
    
    try:
        # Authenticate user
        user = await get_user_from_websocket_token(token, db)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        await websocket.accept()
        print(f"📞 Voice call notification WebSocket connected for user {user.id}")
        
        # Add to call manager for notifications
        await call_manager.add_notification_connection(user.id, websocket)
        
        try:
            while True:
                # Keep connection alive and handle any incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                message_type = message.get('type')
                
                if message_type == 'initiate_call':
                    target_user_id = message.get('target_user_id')
                    if target_user_id:
                        # This is handled via REST API, just acknowledge
                        await websocket.send_text(json.dumps({
                            'type': 'call_initiation_received',
                            'target_user_id': target_user_id
                        }))
                
                elif message_type == 'heartbeat':
                    await websocket.send_text(json.dumps({'type': 'heartbeat_ack'}))
                
                elif message_type == 'answer_call':
                    call_id = message.get('call_id')
                    # Handle via REST API
                    await websocket.send_text(json.dumps({
                        'type': 'call_answer_received',
                        'call_id': call_id
                    }))
                
                elif message_type == 'decline_call':
                    call_id = message.get('call_id')
                    # Handle via REST API
                    await websocket.send_text(json.dumps({
                        'type': 'call_decline_received',
                        'call_id': call_id
                    }))
        
        except WebSocketDisconnect:
            print(f"📞 Voice call notification WebSocket disconnected for user {user.id}")
        except Exception as e:
            print(f"❌ Voice call notification WebSocket error for user {user.id}: {e}")
    
    finally:
        # Clean up connection
        try:
            await call_manager.remove_notification_connection(user.id)
        except Exception:
            pass
