"""
Chat Summary API Routes
Provides endpoints for generating and downloading chat summaries
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io
import logging
import re
from datetime import datetime

# Import local modules
from ..core.database import get_db
from ..core.security import decode_access_token
from ..models.user import User
from ..models.chatroom import Chatroom
from ..models.message import Message
from ..services.chat_summary_service import chat_summary_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat-summary", tags=["chat-summary"])


def get_current_user_from_session(request: Request, db: Session = Depends(get_db)) -> User:
    """Get current user from session token"""
    token = request.session.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        payload = decode_access_token(token)
    except Exception as e:
        logger.debug(f"Token decode failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


@router.get("/direct/{other_user_id}")
async def generate_direct_chat_summary(
    other_user_id: int,
    request: Request,
    language: str = Query("en", description="Language for summary generation"),
    db: Session = Depends(get_db)
):
    """
    Generate AI summary for direct chat conversation
    
    Args:
        other_user_id: ID of the other user in the conversation
        language: Language code for summary (en, ar, fr, etc.)
        
    Returns:
        JSON with summary data and statistics
    """
    try:
        # Get current user
        current_user = get_current_user_from_session(request, db)
        
        # Validate other user exists
        other_user = db.query(User).filter(User.id == other_user_id).first()
        if not other_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent self-chat summary
        if other_user_id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot summarize chat with yourself")
        
        # Get direct chat messages
        messages = await _get_direct_chat_messages(current_user.id, other_user_id, db, language)
        
        if not messages:
            return {
                "success": False,
                "error": "No chat history found",
                "message_count": 0,
                "summary": "No messages to summarize"
            }
        
        # Prepare participants list
        participants = [
            current_user.full_name or current_user.email,
            other_user.full_name or other_user.email
        ]
        
        # Generate summary using AI service
        summary_result = await chat_summary_service.generate_chat_summary(
            messages=messages,
            user_language=language,
            chat_type="direct",
            participants=participants,
            user_id=current_user.id
        )
        
        if summary_result["success"]:
            logger.info(f"Generated summary for user {current_user.id} with {other_user.email}")
            return {
                "success": True,
                "summary": summary_result["summary"],
                "statistics": summary_result["statistics"],
                "generated_at": summary_result["generated_at"],
                "language": language,
                "participants": participants,
                "message_count": len(messages),
                "other_user": {
                    "id": other_user.id,
                    "name": other_user.full_name or other_user.email,
                    "email": other_user.email
                }
            }
        else:
            raise HTTPException(status_code=500, detail=summary_result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct chat summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.get("/group/{group_id}")
async def generate_group_chat_summary(
    group_id: int,
    request: Request,
    language: str = Query("en", description="Language for summary generation"),
    db: Session = Depends(get_db)
):
    """
    Generate AI summary for group chat conversation
    
    Args:
        group_id: ID of the group chat
        language: Language code for summary (en, ar, fr, etc.)
        
    Returns:
        JSON with summary data and statistics
    """
    try:
        # Get current user
        current_user = get_current_user_from_session(request, db)
        
        # Get and validate group
        group = db.query(Chatroom).filter(
            Chatroom.id == group_id,
            Chatroom.is_group_chat.is_(True)
        ).first()
        
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Check if user is member of the group
        if current_user not in group.members:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        
        # Get group chat messages
        messages = await _get_group_chat_messages(group_id, db, language)
        
        if not messages:
            return {
                "success": False,
                "error": "No chat history found",
                "message_count": 0,
                "summary": "No messages to summarize"
            }
        
        # Get participant names
        participants = []
        participant_ids = set()
        for msg in messages:
            if msg.get('sender_id') not in participant_ids:
                participant_ids.add(msg.get('sender_id'))
                participants.append(msg.get('sender_name', 'Unknown'))
        
        # Generate summary using AI service
        summary_result = await chat_summary_service.generate_chat_summary(
            messages=messages,
            user_language=language,
            chat_type="group",
            participants=participants,
            user_id=current_user.id
        )
        
        if summary_result["success"]:
            logger.info(f"Generated group summary for user {current_user.id} in group {group_id}")
            return {
                "success": True,
                "summary": summary_result["summary"],
                "statistics": summary_result["statistics"],
                "generated_at": summary_result["generated_at"],
                "language": language,
                "participants": participants,
                "message_count": len(messages),
                "group": {
                    "id": group.id,
                    "name": group.chatroom_name,
                    "member_count": len(group.members) if hasattr(group, 'members') else 0
                }
            }
        else:
            raise HTTPException(status_code=500, detail=summary_result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Group chat summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.get("/direct/{other_user_id}/download")
async def download_direct_chat_summary(
    other_user_id: int,
    request: Request,
    language: str = Query("en", description="Language for summary generation"),
    format: str = Query("markdown", description="Download format: markdown or txt"),
    db: Session = Depends(get_db)
):
    """
    Download direct chat summary as a file
    
    Args:
        other_user_id: ID of the other user in the conversation
        language: Language code for summary
        format: File format (markdown or txt)
        
    Returns:
        File download response
    """
    try:
        # Get current user (for authentication)
        get_current_user_from_session(request, db)
        
        # Get other user for filename
        other_user = db.query(User).filter(User.id == other_user_id).first()
        if not other_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate summary (reuse the endpoint logic)
        summary_response = await generate_direct_chat_summary(
            other_user_id, request, language, db
        )
        
        if not summary_response.get("success"):
            raise HTTPException(status_code=404, detail="No chat history to summarize")
        
        # Create downloadable content
        summary_data = summary_response
        content = chat_summary_service.create_downloadable_summary(summary_data, format)
        
        # Generate filename
        other_user_name = (other_user.full_name or other_user.email).replace(' ', '_')
        other_user_name = re.sub(r'[<>:"/\\|?*]', '', other_user_name)
        current_date = datetime.now().strftime('%Y-%m-%d')
        file_extension = "md" if format == "markdown" else "txt"
        filename = f"chat_summary_{other_user_name}_{current_date}_{language}.{file_extension}"
        
        # Determine media type
        media_type = 'text/markdown' if format == "markdown" else 'text/plain'
        
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type=media_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': f'{media_type}; charset=utf-8'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct chat summary download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary file")


@router.get("/group/{group_id}/download")
async def download_group_chat_summary(
    group_id: int,
    request: Request,
    language: str = Query("en", description="Language for summary generation"),
    format: str = Query("markdown", description="Download format: markdown or txt"),
    db: Session = Depends(get_db)
):
    """
    Download group chat summary as a file
    
    Args:
        group_id: ID of the group chat
        language: Language code for summary
        format: File format (markdown or txt)
        
    Returns:
        File download response
    """
    try:
        # Get current user (for authentication)
        get_current_user_from_session(request, db)
        
        # Get group for filename
        group = db.query(Chatroom).filter(
            Chatroom.id == group_id,
            Chatroom.is_group_chat.is_(True)
        ).first()
        
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Generate summary (reuse the endpoint logic)
        summary_response = await generate_group_chat_summary(
            group_id, request, language, db
        )
        
        if not summary_response.get("success"):
            raise HTTPException(status_code=404, detail="No group chat history to summarize")
        
        # Create downloadable content
        summary_data = summary_response
        content = chat_summary_service.create_downloadable_summary(summary_data, format)
        
        # Generate filename
        group_name = group.chatroom_name.replace(' ', '_')
        group_name = re.sub(r'[<>:"/\\|?*]', '', group_name)
        current_date = datetime.now().strftime('%Y-%m-%d')
        file_extension = "md" if format == "markdown" else "txt"
        filename = f"group_summary_{group_name}_{current_date}_{language}.{file_extension}"
        
        # Determine media type
        media_type = 'text/markdown' if format == "markdown" else 'text/plain'
        
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type=media_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': f'{media_type}; charset=utf-8'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Group chat summary download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary file")


@router.get("/status")
async def get_summary_service_status():
    """
    Get status of the chat summary service
    
    Returns:
        Service status and model information
    """
    try:
        status = chat_summary_service.get_model_status()
        return {
            "success": True,
            "service_status": "operational",
            "model_status": status,
            "supported_formats": ["markdown", "txt"],
            "supported_languages": status.get("supported_languages", [])
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {
            "success": False,
            "service_status": "error",
            "error": str(e),
            "supported_formats": ["markdown", "txt"],
            "supported_languages": ["en", "ar", "fr"]
        }


# Helper functions

async def _get_direct_chat_messages(user1_id: int, user2_id: int, db: Session, language: str = "en"):
    """Get direct chat messages between two users"""
    try:
        # Find direct chatroom
        direct_key = f"direct:{min(user1_id, user2_id)}:{max(user1_id, user2_id)}"
        chatroom = (
            db.query(Chatroom)
            .filter(
                Chatroom.chatroom_name == direct_key,
                Chatroom.is_group_chat.is_(False)
            )
            .first()
        )
        
        if not chatroom:
            return []
        
        # Get messages
        messages = (
            db.query(Message)
            .filter(Message.chatroom_id == chatroom.id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        
        # Format messages for AI processing
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "timestamp": message.timestamp.isoformat() if message.timestamp else "",
                "sender_id": message.sender_id,
                "sender_name": message.sender.full_name or message.sender.email,
                "message_type": message.message_type.value if message.message_type else "text",
                "original_text": message.original_text or "",
                "translated_text": message.get_translated_text(language) or "",  # Use requested language
                "original_language": message.original_language or "",
                "audio_duration": message.audio_duration or 0
            })
        
        return formatted_messages
        
    except Exception as e:
        logger.error(f"Error getting direct chat messages: {e}")
        return []


async def _get_group_chat_messages(group_id: int, db: Session, language: str = "en"):
    """Get group chat messages"""
    try:
        # Get messages
        messages = (
            db.query(Message)
            .filter(Message.chatroom_id == group_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        
        # Format messages for AI processing
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "timestamp": message.timestamp.isoformat() if message.timestamp else "",
                "sender_id": message.sender_id,
                "sender_name": message.sender.full_name or message.sender.email,
                "message_type": message.message_type.value if message.message_type else "text",
                "original_text": message.original_text or "",
                "translated_text": message.get_translated_text(language) or "",  # Use requested language
                "original_language": message.original_language or "",
                "audio_duration": message.audio_duration or 0
            })
        
        return formatted_messages
        
    except Exception as e:
        logger.error(f"Error getting group chat messages: {e}")
        return []
