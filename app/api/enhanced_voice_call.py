# Enhanced Voice Call API with SeamlessM4T Real-time Translation
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from typing import Optional
import json
import logging
import time
from datetime import datetime

from ..core.database import get_db
from ..core.security import decode_access_token
from ..api.deps import get_current_user
from ..models.user import User
from ..models.voice_call import VoiceCall, CallStatus, CallType
from ..services.voice_call_manager import voice_call_manager
from ..services.seamless_translation_service import seamless_translation_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice-call", tags=["voice-calls"])

async def authenticate_websocket_user(token: str, db: Session) -> Optional[User]:
    """Authenticate user from WebSocket token"""
    try:
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            return None
        
        user = db.query(User).filter(User.email == payload["sub"]).first()
        return user
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None

@router.post("/initiate")
async def initiate_call(
    target_user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Initiate a new voice call with real-time translation"""
    try:
        # Validate target user
        target_user = db.query(User).filter(User.id == target_user_id).first()
        if not target_user:
            raise HTTPException(status_code=404, detail="Target user not found")
        
        # Prevent self-calling
        if target_user_id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot call yourself")
        
        # Check if user is already in a call
        existing_call = voice_call_manager.get_user_call(current_user.id)
        if existing_call:
            raise HTTPException(status_code=409, detail="You are already in a call")
        
        # Create call session
        call_id = await voice_call_manager.create_call(current_user.id, target_user_id)
        
        # Create database record
        voice_call = VoiceCall(
            call_id=call_id,
            caller_id=current_user.id,
            callee_id=target_user_id,
            call_type=CallType.VOICE,
            status=CallStatus.INITIATED
        )
        
        db.add(voice_call)
        db.commit()
        db.refresh(voice_call)
        
        logger.info(f"Initiated call {call_id} from {current_user.id} to {target_user_id}")
        
        return {
            "success": True,
            "call_id": call_id,
            "caller_id": current_user.id,
            "callee_id": target_user_id,
            "status": "initiated",
            "translation_enabled": True,
            "supported_languages": seamless_translation_service.get_supported_languages()
        }
        
    except Exception as e:
        logger.error(f"Call initiation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/answer/{call_id}")
async def answer_call(
    call_id: str,
    language: str = "en",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Answer an incoming call"""
    try:
        # Validate call exists
        voice_call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
        if not voice_call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Validate user is the callee
        if voice_call.callee_id != current_user.id:
            raise HTTPException(status_code=403, detail="You are not authorized to answer this call")
        
        # Update call status
        voice_call.status = CallStatus.ANSWERED
        voice_call.answered_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Call {call_id} answered by user {current_user.id} with language {language}")
        
        return {
            "success": True,
            "call_id": call_id,
            "status": "answered",
            "user_language": language,
            "translation_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Call answer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/end/{call_id}")
async def end_call(
    call_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End a voice call"""
    try:
        # Validate call exists
        voice_call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
        if not voice_call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Validate user is participant
        if voice_call.caller_id != current_user.id and voice_call.callee_id != current_user.id:
            raise HTTPException(status_code=403, detail="You are not authorized to end this call")
        
        # End call in manager
        await voice_call_manager.end_call(call_id, current_user.id)
        
        # Update database
        voice_call.status = CallStatus.ENDED
        voice_call.ended_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Call {call_id} ended by user {current_user.id}")
        
        return {
            "success": True,
            "call_id": call_id,
            "status": "ended"
        }
        
    except Exception as e:
        logger.error(f"Call end error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{call_id}")
async def voice_call_websocket(
    websocket: WebSocket,
    call_id: str,
    token: str = Query(...),
    language: str = Query(default="en"),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for voice call with real-time translation"""
    await websocket.accept()
    
    try:
        # Authenticate user
        user = await authenticate_websocket_user(token, db)
        if not user:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Authentication failed"
            }))
            await websocket.close()
            return
        
        # Validate supported language
        if language not in seamless_translation_service.supported_languages:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Language '{language}' not supported. Supported: {list(seamless_translation_service.supported_languages.keys())}"
            }))
            await websocket.close()
            return
        
        # Join call
        joined = await voice_call_manager.join_call(call_id, user.id, websocket, language)
        if not joined:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Failed to join call"
            }))
            await websocket.close()
            return
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "call_id": call_id,
            "user_id": user.id,
            "language": language,
            "translation_available": seamless_translation_service.is_available(),
            "supported_languages": seamless_translation_service.get_supported_languages()
        }))
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(call_id, user.id, message, websocket)
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received in voice call WebSocket")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid message format"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user.id if 'user' in locals() else 'unknown'} in call {call_id}")
        if 'user' in locals():
            await voice_call_manager.leave_call(call_id, user.id)
            
    except Exception as e:
        logger.error(f"WebSocket error in call {call_id}: {e}")
        if 'user' in locals():
            await voice_call_manager.leave_call(call_id, user.id)

async def handle_websocket_message(call_id: str, user_id: int, message: dict, websocket: WebSocket):
    """Handle different types of WebSocket messages"""
    message_type = message.get("type")
    
    try:
        if message_type == "webrtc_offer":
            await voice_call_manager.handle_webrtc_offer(call_id, user_id, message)
            
        elif message_type == "webrtc_answer":
            await voice_call_manager.handle_webrtc_answer(call_id, user_id, message)
            
        elif message_type == "ice_candidate":
            await voice_call_manager.handle_ice_candidate(call_id, user_id, message)
            
        elif message_type == "voice_chunk":
            # Real-time voice translation
            await voice_call_manager.handle_voice_translation(call_id, user_id, message)
            
        elif message_type == "translation_settings":
            # Update translation preferences
            settings = message.get("settings", {})
            await voice_call_manager.update_translation_settings(call_id, user_id, settings)
            
        elif message_type == "ping":
            # Heartbeat
            await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
            
        else:
            logger.warning(f"Unknown message type: {message_type}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }))
            
    except Exception as e:
        logger.error(f"Error handling message {message_type}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to process {message_type}"
        }))

@router.get("/status/{call_id}")
async def get_call_status(
    call_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current call status"""
    try:
        # Get call from database
        voice_call = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
        if not voice_call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Validate user is participant
        if voice_call.caller_id != current_user.id and voice_call.callee_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get call session from manager
        call_session = voice_call_manager.get_call_session(call_id)
        
        return {
            "call_id": call_id,
            "status": voice_call.status.value,
            "caller_id": voice_call.caller_id,
            "callee_id": voice_call.callee_id,
            "created_at": voice_call.created_at.isoformat() if voice_call.created_at else None,
            "answered_at": voice_call.answered_at.isoformat() if voice_call.answered_at else None,
            "ended_at": voice_call.ended_at.isoformat() if voice_call.ended_at else None,
            "active_participants": call_session.get_participant_count() if call_session else 0,
            "translation_enabled": True,
            "supported_languages": seamless_translation_service.get_supported_languages()
        }
        
    except Exception as e:
        logger.error(f"Get call status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/translation/test")
async def test_translation_service():
    """Test SeamlessM4T translation service"""
    try:
        stats = seamless_translation_service.get_translation_stats()
        
        return {
            "service_available": seamless_translation_service.is_available(),
            "supported_languages": seamless_translation_service.get_supported_languages(),
            "stats": stats,
            "test_message": "SeamlessM4T real-time voice translation ready!" if seamless_translation_service.is_available() else "SeamlessM4T not available - please install seamless_communication"
        }
        
    except Exception as e:
        logger.error(f"Translation service test error: {e}")
        return {
            "service_available": False,
            "error": str(e),
            "supported_languages": {},
            "test_message": "Translation service test failed"
        }

@router.get("/stats")
async def get_call_stats(current_user: User = Depends(get_current_user)):
    """Get voice call system statistics"""
    try:
        manager_stats = voice_call_manager.get_stats()
        translation_stats = seamless_translation_service.get_translation_stats()
        
        return {
            "call_manager": manager_stats,
            "translation_service": translation_stats,
            "user_active_call": voice_call_manager.get_user_call(current_user.id)
        }
        
    except Exception as e:
        logger.error(f"Get call stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Remove the placeholder function since we imported the correct one
