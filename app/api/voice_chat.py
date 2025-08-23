# filepath: app/api/voice_chat.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import uuid
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..core.database import get_db
from ..api.deps import get_current_user
from ..models.user import User
from ..models.message import Message, MessageType
from ..models.chatroom import Chatroom
from ..models.audio_file import AudioFile
from ..schemas.voice import VoiceMessageResponse
from ..services.voice_service import VoiceService
from ..core.chat_manager import chat_manager

router = APIRouter(prefix="/api/voice", tags=["voice-chat"])

# Create uploads directory
UPLOAD_DIR = Path("app/static/uploads/voice")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize voice service
voice_service = VoiceService()

@router.post("/upload-message", response_model=VoiceMessageResponse)
async def upload_voice_message(
    audio: UploadFile = File(...),
    language: str = Form(...),
    recipient_id: Optional[int] = Form(None),
    chatroom_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process voice message for English, French, and Arabic only
    - Enhanced speech recognition with Whisper
    - Asynchronous translation to other supported languages
    - Generates voice files for translations
    - Broadcasts via WebSocket
    
    Supports both direct messages (recipient_id) and chatroom messages (chatroom_id)
    """
    try:
        # Validate language selection
        supported_languages = ['en', 'fr', 'ar']
        if language not in supported_languages:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language '{language}'. Supported languages: {supported_languages}"
            )
        
        # Determine chatroom - either from chatroom_id or create/find direct message chatroom
        target_chatroom_id = None
        
        if chatroom_id:
            # Using existing chatroom
            chatroom = db.query(Chatroom).filter(Chatroom.id == chatroom_id).first()
            if not chatroom:
                raise HTTPException(status_code=404, detail="Chatroom not found")
            target_chatroom_id = chatroom_id
            
        elif recipient_id:
            # Direct message - create or find chatroom
            recipient = db.query(User).filter(User.id == recipient_id).first()
            if not recipient:
                raise HTTPException(status_code=404, detail="Recipient not found")
            
            # Find or create direct message chatroom
            target_chatroom_id = await get_or_create_direct_chatroom(current_user.id, recipient_id, db)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either chatroom_id or recipient_id must be provided"
            )
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        print(f"🎙️ Processing voice message: {len(audio_data)} bytes, language: {language}")
        
        try:
            # Process voice message using enhanced async voice service
            result = await voice_service.create_voice_message(
                audio_data=audio_data,
                user_id=current_user.id,
                chatroom_id=target_chatroom_id,
                target_language=language,
                db=db
            )
            
            # Broadcast message via WebSocket
            await broadcast_voice_message(
                message=result["message"],
                sender=current_user
            )
            
            # Check if this is a voice-only message (transcription failed)
            if result.get("voice_only", False):
                return VoiceMessageResponse(
                    success=True,
                    message_id=result["message"].id,
                    transcribed_text=result.get("transcribed_text", ""),
                    translations=result.get("translations", {}),
                    audio_urls=result.get("audio_urls", {}),
                    audio_duration=result.get("audio_duration", 0),
                    error="Audio saved successfully, but transcription was unavailable. Recipients can still listen to your voice message."
                )
            else:
                # Normal successful transcription and translation
                return VoiceMessageResponse(
                    success=True,
                    message_id=result["message"].id,
                    transcribed_text=result["transcribed_text"],
                    translations=result["translations"],
                    audio_urls=result["audio_urls"],
                    audio_duration=result.get("audio_duration", 0)
                )
            
        except Exception as e:
            error_message = str(e)
            
            # Return error response for any actual exceptions
            return VoiceMessageResponse(
                success=False,
                message_id=0,
                transcribed_text="",
                translations={},
                audio_urls={},
                error=f"Processing failed: {error_message}"
            )
                
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Handle any unexpected errors at the top level
        return VoiceMessageResponse(
            success=False,
            message_id=0,
            transcribed_text="",
            translations={},
            audio_urls={},
            error=f"Unexpected error: {str(e)}"
        )

@router.get("/audio/{file_name}")
async def serve_audio_file(file_name: str):
    """Serve audio files"""
    file_path = UPLOAD_DIR / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=file_name
    )

@router.get("/stats/{user_id}")
async def get_voice_stats(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get voice chat statistics for a user"""
    
    # Check if requesting own stats or if admin
    if current_user.id != user_id:
        # Add admin check here if needed
        raise HTTPException(status_code=403, detail="Access denied")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    voice_count = user.get_voice_message_count(db)
    total_duration = user.get_total_voice_duration(db)
    
    return {
        "user_id": user_id,
        "total_voice_messages": voice_count,
        "total_voice_duration": total_duration,
        "average_message_duration": total_duration / voice_count if voice_count > 0 else 0
    }

async def save_temp_audio(audio: UploadFile) -> str:
    """Save uploaded audio to temporary file"""
    # Generate unique temp filename
    file_extension = "webm"  # Default for browser recordings
    if audio.filename:
        file_extension = audio.filename.split('.')[-1] if '.' in audio.filename else "webm"
    
    # Use tempfile to get the correct temporary directory for the OS
    temp_dir = tempfile.gettempdir()
    temp_filename = f"voice_{uuid.uuid4().hex}.{file_extension}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    # Save file content
    content = await audio.read()
    with open(temp_path, "wb") as temp_file:
        temp_file.write(content)
    
    return temp_path

async def broadcast_voice_message(message: Message, sender: User):
    """Broadcast voice message via WebSocket"""
    
    # Get chatroom
    chatroom = message.chatroom
    
    # Prepare WebSocket message
    message_data = {
        "type": "message",
        "message_type": "voice",
        "id": message.id,
        "sender": {
            "id": sender.id,
            "name": sender.full_name or sender.email
        },
        "original_text": message.original_text,
        "original_language": message.original_language,
        "translations_cache": message.translations_cache or {},
        "audio_urls": message.audio_urls or {},
        "audio_duration": message.audio_duration,
        "timestamp": message.timestamp.isoformat(),
        "chatroom_id": chatroom.id
    }
    
    # Broadcast to the entire chatroom (room-based messaging)
    try:
        await chat_manager.broadcast(chatroom.id, message_data)
        print(f"✅ Voice message broadcasted to chatroom {chatroom.id}")
    except Exception as e:
        print(f"❌ Failed to broadcast voice message to chatroom {chatroom.id}: {e}")

@router.delete("/audio/{message_id}")
async def delete_voice_message(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete voice message and associated audio files"""
    
    # Get message
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check if user owns the message
    if message.sender_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this message")
    
    # Delete audio files from filesystem
    if message.audio_urls:
        for url in message.audio_urls.values():
            if url and url.startswith("/static/uploads/voice/"):
                file_name = url.split("/")[-1]
                file_path = UPLOAD_DIR / file_name
                if file_path.exists():
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Failed to delete audio file {file_path}: {e}")
    
    # Delete AudioFile records
    db.query(AudioFile).filter(AudioFile.message_id == message_id).delete()
    
    # Delete message
    db.delete(message)
    db.commit()
    
    return {"success": True, "message": "Voice message deleted"}

async def get_or_create_direct_chatroom(user1_id: int, user2_id: int, db: Session) -> int:
    """Get or create a direct message chatroom between two users"""
    
    # Ensure consistent ordering to avoid duplicate chatrooms
    if user1_id > user2_id:
        user1_id, user2_id = user2_id, user1_id
    
    # Look for existing direct message chatroom (not group chat)
    existing_chatroom = db.query(Chatroom).join(
        Chatroom.members
    ).filter(
        ~Chatroom.is_group_chat,
        Chatroom.members.any(user_id=user1_id),
        Chatroom.members.any(user_id=user2_id)
    ).first()
    
    if existing_chatroom:
        return existing_chatroom.id
    
    # Create new direct message chatroom
    chatroom_name = f"DM_{user1_id}_{user2_id}"
    new_chatroom = Chatroom(
        chatroom_name=chatroom_name,
        is_group_chat=False
    )
    
    db.add(new_chatroom)
    db.flush()  # Get the ID
    
    # Add both users as members
    from ..models.message import ChatroomMember
    
    member1 = ChatroomMember(user_id=user1_id, chatroom_id=new_chatroom.id)
    member2 = ChatroomMember(user_id=user2_id, chatroom_id=new_chatroom.id)
    
    db.add(member1)
    db.add(member2)
    db.commit()
    
    return new_chatroom.id

@router.post("/translate-realtime")
async def translate_realtime_voice(
    audio: UploadFile = File(...),
    language: str = Form(...),
    target_language: str = Form(...),
    call_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Real-time voice translation for voice calls
    - Speech-to-text using Whisper
    - Translation using local models
    - Text-to-speech for translated audio
    """
    try:
        # Validate languages
        supported_languages = ['en', 'fr', 'ar']
        if language not in supported_languages or target_language not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported: {supported_languages}"
            )
        
        if language == target_language:
            raise HTTPException(status_code=400, detail="Source and target languages cannot be the same")
        
        # Read audio data
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        print(f"🌐 Real-time translation: {language} → {target_language} ({len(audio_data)} bytes)")
        
        # Process with voice service
        result = await voice_service.translate_realtime(
            audio_data=audio_data,
            source_language=language,
            target_language=target_language,
            user_id=current_user.id,
            call_id=call_id
        )
        
        return {
            "success": True,
            "original_text": result.get("original_text"),
            "translated_text": result.get("translated_text"),
            "translated_audio_url": result.get("translated_audio_url"),
            "source_language": language,
            "target_language": target_language,
            "processing_time": result.get("processing_time", 0)
        }
        
    except Exception as e:
        print(f"❌ Real-time translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
