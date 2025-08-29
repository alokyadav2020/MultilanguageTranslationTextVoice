# filepath: app/api/groups.py
from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional
import os
import uuid
import logging

from ..core.database import get_db
from ..api.deps import get_current_user
from ..models.user import User
from ..models.group import Group, GroupMessage, GroupMessageTranslation, group_members, GroupType, MessageReaction
from ..services.translation import translation_service
from ..services.group_manager import group_manager
from ..services.voice_service import VoiceMessageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/groups", tags=["groups"])

# Initialize voice service
voice_service = VoiceMessageService()

@router.post("/create")
async def create_group(
    name: str = Form(...),
    description: str = Form(""),
    group_type: str = Form("private"),
    default_language: str = Form("en"),
    max_members: int = Form(100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new group"""
    try:
        # Validate group type
        group_type_enum = GroupType(group_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid group type")
    
    # Create group
    group = Group(
        name=name,
        description=description,
        group_type=group_type_enum,
        created_by=current_user.id,
        default_language=default_language,
        max_members=max_members
    )
    
    db.add(group)
    db.commit()
    db.refresh(group)
    
    # Add creator as admin
    db.execute(
        group_members.insert().values(
            group_id=group.id,
            user_id=current_user.id,
            role='admin',
            preferred_language=default_language,
            voice_language=default_language
        )
    )
    db.commit()
    
    logger.info(f"👥 Group '{name}' created by user {current_user.id}")
    
    return {
        "success": True,
        "group_id": group.id,
        "message": f"Group '{name}' created successfully"
    }

@router.get("/list")
async def list_user_groups(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of groups user is a member of"""
    groups = current_user.get_groups(db)
    
    group_list = []
    for group in groups:
        member_count = group.get_member_count(db)
        user_role = group.get_member_role(current_user.id, db)
        
        group_list.append({
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "type": group.group_type.value,
            "member_count": member_count,
            "user_role": user_role,
            "default_language": group.default_language,
            "created_at": group.created_at.isoformat(),
            "profile_picture": group.profile_picture
        })
    
    return {"groups": group_list}

@router.get("/{group_id}")
async def get_group_details(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed group information"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Get all members with their roles and language preferences
    members_query = db.query(
        User.id, User.full_name, User.email,
        group_members.c.role, 
        group_members.c.preferred_language,
        group_members.c.voice_language,
        group_members.c.joined_at
    ).join(group_members).filter(group_members.c.group_id == group_id).all()
    
    members = []
    for member in members_query:
        members.append({
            "id": member.id,
            "name": member.full_name or member.email,
            "email": member.email,
            "role": member.role,
            "preferred_language": member.preferred_language,
            "voice_language": member.voice_language,
            "joined_at": member.joined_at.isoformat()
        })
    
    return {
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "type": group.group_type.value,
        "default_language": group.default_language,
        "max_members": group.max_members,
        "created_at": group.created_at.isoformat(),
        "profile_picture": group.profile_picture,
        "members": members,
        "user_role": membership.role,
        "user_preferred_language": membership.preferred_language,
        "user_voice_language": membership.voice_language
    }

@router.post("/{group_id}/add-member")
async def add_group_member(
    group_id: int,
    user_email: str = Form(...),
    role: str = Form("member"),
    preferred_language: str = Form("en"),
    voice_language: str = Form("en"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a member to the group"""
    # Check if current user is admin or moderator
    user_role = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not user_role or user_role.role not in ['admin', 'moderator']:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Check if group exists
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if user exists
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user is already a member
    existing = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == user.id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="User is already a member")
    
    # Check group member limit
    if group.get_member_count(db) >= group.max_members:
        raise HTTPException(status_code=400, detail="Group is full")
    
    # Add member
    db.execute(
        group_members.insert().values(
            group_id=group_id,
            user_id=user.id,
            role=role,
            preferred_language=preferred_language,
            voice_language=voice_language
        )
    )
    db.commit()
    
    # Notify group via WebSocket
    await group_manager.notify_group(group_id, {
        "type": "member_added",
        "user": {
            "id": user.id,
            "name": user.full_name or user.email,
            "role": role
        },
        "added_by": current_user.full_name or current_user.email
    })
    
    return {"success": True, "message": f"User {user.full_name or user.email} added to group"}

@router.post("/{group_id}/remove-member")
async def remove_group_member(
    group_id: int,
    user_id: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a member from the group"""
    # Check if current user is admin
    user_role = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not user_role or user_role.role != 'admin':
        raise HTTPException(status_code=403, detail="Only admins can remove members")
    
    # Don't allow removing self
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot remove yourself")
    
    # Remove member
    result = db.execute(
        group_members.delete().where(
            (group_members.c.group_id == group_id) & 
            (group_members.c.user_id == user_id)
        )
    )
    
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")
    
    db.commit()
    
    # Get user details for notification
    user = db.query(User).filter(User.id == user_id).first()
    
    # Notify group
    await group_manager.notify_group(group_id, {
        "type": "member_removed",
        "user": {
            "id": user_id,
            "name": user.full_name if user else "Unknown"
        },
        "removed_by": current_user.full_name or current_user.email
    })
    
    return {"success": True, "message": "Member removed successfully"}

@router.post("/{group_id}/leave")
async def leave_group(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Leave a group"""
    # Check if user is member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    group = db.query(Group).filter(Group.id == group_id).first()
    
    # If user is the creator and there are other members, transfer ownership
    if group.created_by == current_user.id:
        other_admins = db.query(group_members).filter(
            group_members.c.group_id == group_id,
            group_members.c.user_id != current_user.id,
            group_members.c.role == 'admin'
        ).first()
        
        if not other_admins:
            # Make the first member an admin
            first_member = db.query(group_members).filter(
                group_members.c.group_id == group_id,
                group_members.c.user_id != current_user.id
            ).first()
            
            if first_member:
                db.execute(
                    group_members.update().where(
                        (group_members.c.group_id == group_id) & 
                        (group_members.c.user_id == first_member.user_id)
                    ).values(role='admin')
                )
                group.created_by = first_member.user_id
            else:
                # No other members, delete the group
                db.delete(group)
                db.commit()
                return {"success": True, "message": "Group deleted as you were the last member"}
    
    # Remove user from group
    db.execute(
        group_members.delete().where(
            (group_members.c.group_id == group_id) & 
            (group_members.c.user_id == current_user.id)
        )
    )
    db.commit()
    
    # Notify group
    await group_manager.notify_group(group_id, {
        "type": "member_left",
        "user": {
            "id": current_user.id,
            "name": current_user.full_name or current_user.email
        }
    })
    
    return {"success": True, "message": "Left group successfully"}

@router.post("/{group_id}/update-preferences")
async def update_language_preferences(
    group_id: int,
    preferred_language: str = Form(...),
    voice_language: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's language preferences for this group"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Update preferences
    db.execute(
        group_members.update().where(
            (group_members.c.group_id == group_id) & 
            (group_members.c.user_id == current_user.id)
        ).values(
            preferred_language=preferred_language,
            voice_language=voice_language
        )
    )
    db.commit()
    
    return {
        "success": True, 
        "message": "Language preferences updated",
        "preferred_language": preferred_language,
        "voice_language": voice_language
    }

@router.post("/{group_id}/send-message")
async def send_group_message(
    group_id: int,
    content: str = Form(...),
    language: str = Form(...),
    message_type: str = Form("text"),
    reply_to_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a text message to the group"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Get group
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Create message
    message = GroupMessage(
        group_id=group_id,
        sender_id=current_user.id,
        content=content,
        original_language=language,
        message_type=message_type,
        reply_to_id=reply_to_id
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    # Get all group members with their language preferences
    members_query = db.query(
        User.id,
        User.full_name,
        User.email,
        group_members.c.user_id,
        group_members.c.preferred_language,
        group_members.c.voice_language
    ).join(
        group_members, User.id == group_members.c.user_id
    ).filter(
        group_members.c.group_id == group_id
    ).all()
    
    # Concurrent translation for all unique languages
    import asyncio
    unique_languages = set()
    for member in members_query:
        member_lang = member.preferred_language or 'en'
        if member_lang != language:
            unique_languages.add(member_lang)
    
    # Translate message for each unique language concurrently
    translations = {language: content}  # Include original
    
    async def translate_for_language(target_lang: str) -> tuple:
        try:
            logger.info(f"Translating '{content[:50]}...' from {language} to {target_lang}")
            translated = await translation_service.translate_text_async(
                content, language, target_lang
            )
            logger.info(f"Translation result for {target_lang}: '{translated[:50]}...'")
            return target_lang, translated
        except Exception as e:
            logger.error(f"Translation failed for {target_lang}: {e}")
            return target_lang, content  # Fallback to original
    
    if unique_languages:
        logger.info(f"Starting translations for {len(unique_languages)} languages: {list(unique_languages)}")
        translation_tasks = [translate_for_language(lang) for lang in unique_languages]
        translation_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        for result in translation_results:
            if isinstance(result, tuple):
                lang, translated_text = result
                translations[lang] = translated_text
                logger.info(f"Saving translation for {lang}: '{translated_text[:30]}...'")
                
                # Save translation to database
                translation_record = GroupMessageTranslation(
                    message_id=message.id,
                    language=lang,
                    translated_content=translated_text,
                    translation_type='text'
                )
                db.add(translation_record)
    else:
        logger.info("No unique languages found for translation")
    
    db.commit()
    
    logger.info(f"Broadcasting message to group {group_id} with {len(translations)} translations")
    logger.info(f"Available translations: {list(translations.keys())}")
    
    # Enhanced broadcast with member-specific content
    base_message_data = {
        "type": "group_message",  # Use standard type for frontend compatibility
        "group_id": group_id,
        "message_id": message.id,
        "id": message.id,  # Frontend compatibility
        "sender": {
            "id": current_user.id,
            "name": current_user.full_name or current_user.email
        },
        "content": content,
        "original_text": content,  # Frontend expects this
        "original_language": language,
        "message_type": message_type,
        "reply_to_id": reply_to_id,
        "timestamp": message.timestamp.isoformat(),
        "translations": translations,  # Include all translations
        "translations_cache": translations,  # Frontend compatibility
        "all_translations": translations  # Backward compatibility
    }
    
    # Broadcast to all members with their preferred content
    await group_manager.broadcast_to_group_enhanced(group_id, base_message_data, members_query, translations)
    
    return {
        "success": True,
        "message_id": message.id,
        "translations": translations
    }

@router.post("/{group_id}/test-translation")
async def test_group_translation(
    group_id: int,
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("fr"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test translation functionality for debugging"""
    try:
        # Test direct translation
        translated = await translation_service.translate_text_async(text, source_lang, target_lang)
        
        return {
            "success": True,
            "original": text,
            "translated": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_service_available": True
        }
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "original": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

@router.post("/{group_id}/send-voice")
async def send_voice_message(
    group_id: int,
    audio: UploadFile = File(...),
    language: str = Form(...),
    reply_to_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a voice message to the group"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Validate audio file
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    # Create voice directory if it doesn't exist
    voice_dir = "app/static/voice_messages/groups"
    os.makedirs(voice_dir, exist_ok=True)
    
    # Save audio file
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(audio.filename)[1] or '.webm'
    audio_filename = f"{file_id}{file_extension}"
    audio_path = os.path.join(voice_dir, audio_filename)
    
    with open(audio_path, "wb") as buffer:
        content = await audio.read()
        buffer.write(content)
    
    # Transcribe voice message  
    try:
        global voice_service
        transcription = await voice_service.transcribe_audio(audio_path, language)
        content = transcription
        logger.info(f"Voice transcribed successfully: {content[:50]}...")
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        content = "[Voice message]"
    
    # Calculate audio duration
    audio_duration = 0
    try:
        # Try using librosa first
        import librosa
        y, sr = librosa.load(audio_path)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio duration calculated with librosa: {audio_duration} seconds")
    except ImportError:
        logger.warning("librosa not available, using file size estimation")
        try:
            # Fallback: estimate duration based on file size (rough approximation)
            file_size = os.path.getsize(audio_path)
            # Rough estimation: WebM audio is typically ~16kbps = 2KB/s
            estimated_duration = file_size / 2048  # bytes per second
            audio_duration = max(1, min(estimated_duration, 300))  # Cap between 1-300 seconds
            logger.info(f"Audio duration estimated: {audio_duration} seconds")
        except Exception as e:
            logger.warning(f"Could not estimate audio duration: {e}")
            audio_duration = 10  # Default fallback
    except Exception as e:
        logger.warning(f"Could not calculate audio duration: {e}")
        audio_duration = 10  # Default fallback
    
    # Create message
    message = GroupMessage(
        group_id=group_id,
        sender_id=current_user.id,
        content=content,
        original_language=language,
        message_type='voice',
        voice_file_path=audio_path,
        voice_duration=audio_duration,
        reply_to_id=reply_to_id
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    # Get all group members and their language preferences
    members = db.query(group_members).filter(group_members.c.group_id == group_id).all()
    
    # Translate and generate voice for each member's preferences
    translations = {}
    voice_translations = {}
    
    for member in members:
        # Text translation
        if member.preferred_language != language and content != "[Voice message]":
            try:
                translated_text = await translation_service.translate_text_async(
                    content, language, member.preferred_language
                )
                translations[member.preferred_language] = translated_text
                
                # Save text translation
                translation_record = GroupMessageTranslation(
                    message_id=message.id,
                    language=member.preferred_language,
                    translated_content=translated_text,
                    translation_type='text'
                )
                db.add(translation_record)
                
                # Voice translation if member's voice language is different
                if member.voice_language != language:
                    try:
                        from ..services.voice_service import voice_service
                        # Use the groups voice directory
                        groups_voice_dir = "app/static/voice_messages/groups"
                        voice_file = await voice_service.text_to_speech(
                            translated_text, member.voice_language, groups_voice_dir
                        )
                        if voice_file:
                            # Convert to URL format expected by frontend
                            voice_filename = os.path.basename(voice_file)
                            voice_url = f"/static/voice_messages/groups/{voice_filename}"
                            voice_translations[member.voice_language] = voice_url
                            
                            # Save voice translation
                            voice_translation_record = GroupMessageTranslation(
                                message_id=message.id,
                                language=member.voice_language,
                                translated_content=translated_text,
                                translation_type='voice',
                                voice_file_path=voice_file
                            )
                            db.add(voice_translation_record)
                    except Exception as e:
                        logger.error(f"Voice translation failed for {member.voice_language}: {e}")
                        
            except Exception as e:
                logger.error(f"Translation failed for {member.preferred_language}: {e}")
    
    db.commit()
    
    # Get all group members with their language preferences for enhanced broadcasting
    members_query = db.query(
        User.id,
        User.full_name,
        User.email,
        group_members.c.user_id,
        group_members.c.preferred_language,
        group_members.c.voice_language
    ).join(
        group_members, User.id == group_members.c.user_id
    ).filter(
        group_members.c.group_id == group_id
    ).all()
    
    # Prepare audio_urls structure for frontend
    audio_urls = {language: f"/static/voice_messages/groups/{audio_filename}"}
    audio_urls.update(voice_translations)
    
    # Broadcast message to all group members with personalized audio
    message_data = {
        "type": "group_voice_message",
        "group_id": group_id,
        "message_id": message.id,
        "id": message.id,  # Frontend compatibility
        "sender": {
            "id": current_user.id,
            "name": current_user.full_name or current_user.email
        },
        "content": content,
        "original_text": content,
        "transcription": content,
        "original_language": language,
        "message_type": "voice",
        "audio_urls": audio_urls,  # Frontend expects this structure
        "voice_translations": voice_translations,  # Fallback compatibility
        "voice_file": f"/static/voice_messages/groups/{audio_filename}",  # Will be personalized per user
        "translations": translations,
        "translations_cache": translations,  # Frontend compatibility
        "audio_duration": message.voice_duration or 0,
        "voice_duration": message.voice_duration or 0,
        "duration": message.voice_duration or 0,
        "reply_to_id": reply_to_id,
        "timestamp": message.timestamp.isoformat()
    }
    
    # Use enhanced voice broadcasting that sends personalized audio to each user
    await group_manager.broadcast_voice_message_to_group(group_id, message_data, members_query, translations, audio_urls)
    
    return {
        "success": True,
        "message_id": message.id,
        "voice_file": f"/static/voice_messages/groups/{audio_filename}",
        "transcription": content,
        "translations": translations
    }

@router.post("/{group_id}/send-voice-enhanced")
async def send_group_voice_message_enhanced(
    group_id: int,
    audio: UploadFile = File(...),
    language: str = Form(...),
    reply_to_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced group voice message with concurrent translation to all member languages"""
    import asyncio
    import aiofiles
    from pathlib import Path
    
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Validate audio file
    if not audio.content_type or not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    # Create voice directory if it doesn't exist
    voice_dir = Path("app/static/voice_messages/groups")
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # Save audio file
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(audio.filename)[1] or '.webm'
    audio_filename = f"{file_id}{file_extension}"
    audio_path = voice_dir / audio_filename
    
    content = await audio.read()
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(content)
    
    try:
        # Transcribe voice message asynchronously
        from ..services.voice_service import voice_service
        transcription = await voice_service.transcribe_audio(str(audio_path), language)
        if not transcription or transcription == "[Voice message]":
            transcription = "[Voice message - transcription failed]"
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        transcription = "[Voice message - transcription failed]"
    
    # Create message record
    message = GroupMessage(
        group_id=group_id,
        sender_id=current_user.id,
        content=transcription,
        original_language=language,
        message_type='voice',
        voice_file_path=str(audio_path),
        reply_to_id=reply_to_id
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    # Get all group members with their language preferences
    members_query = db.query(
        User.id,
        User.full_name,
        User.email,
        User.preferred_language,
        group_members.c.preferred_language.label('group_language'),
        group_members.c.voice_language.label('group_voice_language')
    ).join(
        group_members, User.id == group_members.c.user_id
    ).filter(
        group_members.c.group_id == group_id
    ).all()
    
    # Prepare concurrent translation
    translations = {}
    voice_translations = {}
    
    # Get unique languages for translation
    unique_text_languages = set()
    unique_voice_languages = set()
    
    for member in members_query:
        text_lang = member.group_language or member.preferred_language or 'en'
        voice_lang = member.group_voice_language or member.preferred_language or 'en'
        unique_text_languages.add(text_lang)
        unique_voice_languages.add(voice_lang)
    
    # Create concurrent translation tasks
    async def translate_text_async(target_lang: str) -> tuple:
        if target_lang == language:
            return target_lang, transcription
        try:
            translated = await translation_service.translate_text_async(
                transcription, language, target_lang
            )
            return target_lang, translated
        except Exception as e:
            logger.error(f"Text translation failed for {target_lang}: {e}")
            return target_lang, transcription
    
    async def generate_voice_async(target_lang: str, text: str) -> tuple:
        if target_lang == language:
            return target_lang, f"/static/voice_messages/groups/{audio_filename}"
        try:
            from ..services.voice_service import voice_service
            # Use target directory for voice generation
            voice_filename = f"{file_id}_{target_lang}.mp3"
            voice_file = await voice_service.text_to_speech(
                text, target_lang, target_dir=str(voice_dir)
            )
            if voice_file:
                return target_lang, f"/static/voice_messages/groups/{voice_filename}"
            return target_lang, None
        except Exception as e:
            logger.error(f"Voice translation failed for {target_lang}: {e}")
            return target_lang, None
    
    # Execute text translations concurrently
    text_tasks = [translate_text_async(lang) for lang in unique_text_languages]
    if text_tasks:
        text_results = await asyncio.gather(*text_tasks, return_exceptions=True)
        for result in text_results:
            if isinstance(result, tuple):
                lang, translated_text = result
                translations[lang] = translated_text
    
    # Save text translations to database
    for lang, translated_content in translations.items():
        if lang != language:
            translation_record = GroupMessageTranslation(
                message_id=message.id,
                language=lang,
                translated_content=translated_content,
                translation_type='text'
            )
            db.add(translation_record)
    
    # Execute voice translations concurrently
    voice_tasks = []
    for lang in unique_voice_languages:
        text_for_voice = translations.get(lang, transcription)
        voice_tasks.append(generate_voice_async(lang, text_for_voice))
    
    if voice_tasks:
        voice_results = await asyncio.gather(*voice_tasks, return_exceptions=True)
        for result in voice_results:
            if isinstance(result, tuple):
                lang, voice_file = result
                if voice_file:
                    voice_translations[lang] = voice_file
                    
                    # Save voice translation to database
                    voice_translation_record = GroupMessageTranslation(
                        message_id=message.id,
                        language=lang,
                        translated_content=translations.get(lang, transcription),
                        translation_type='voice',
                        voice_file_path=voice_file
                    )
                    db.add(voice_translation_record)
    
    db.commit()
    
    # Broadcast enhanced message to all group members
    message_data = {
        "type": "group_voice_message_enhanced",
        "group_id": group_id,
        "message_id": message.id,
        "sender": {
            "id": current_user.id,
            "name": current_user.full_name or current_user.email
        },
        "content": transcription,
        "original_language": language,
        "voice_file": f"/static/voice_messages/groups/{audio_filename}",
        "translations": translations,
        "voice_translations": voice_translations,
        "reply_to_id": reply_to_id,
        "timestamp": message.timestamp.isoformat(),
        "processing_time": f"Processed {len(translations)} text and {len(voice_translations)} voice translations"
    }
    
    await group_manager.broadcast_to_group(group_id, message_data)
    
    return {
        "success": True,
        "message_id": message.id,
        "voice_file": f"/static/voice_messages/groups/{audio_filename}",
        "transcription": transcription,
        "text_translations": translations,
        "voice_translations": voice_translations,
        "members_served": len(members_query)
    }

@router.get("/{group_id}/messages")
async def get_group_messages(
    group_id: int,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get group messages with translations"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Get user's preferred language for this group
    user_language = membership.preferred_language
    
    # Get messages
    messages = db.query(GroupMessage)\
        .filter(GroupMessage.group_id == group_id)\
        .order_by(GroupMessage.timestamp.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    message_list = []
    for msg in messages:
        # Get all translations for this message
        text_translations = {}
        audio_urls = {}
        
        # Get all text translations
        text_translation_records = db.query(GroupMessageTranslation)\
            .filter(
                GroupMessageTranslation.message_id == msg.id,
                GroupMessageTranslation.translation_type == 'text'
            )\
            .all()
        
        for translation_record in text_translation_records:
            text_translations[translation_record.language] = translation_record.translated_content
        
        # Get all voice translations (audio files)
        voice_translation_records = db.query(GroupMessageTranslation)\
            .filter(
                GroupMessageTranslation.message_id == msg.id,
                GroupMessageTranslation.translation_type == 'voice'
            )\
            .all()
        
        for voice_record in voice_translation_records:
            if voice_record.voice_file_path:
                audio_urls[voice_record.language] = f"/static/voice_messages/groups/{os.path.basename(voice_record.voice_file_path)}"
        
        # Add original audio file if it's a voice message
        if msg.message_type == 'voice' and msg.voice_file_path:
            audio_urls[msg.original_language] = f"/static/voice_messages/groups/{os.path.basename(msg.voice_file_path)}"
        
        # Get reactions
        reactions = db.query(MessageReaction)\
            .filter(MessageReaction.group_message_id == msg.id)\
            .all()
        
        reaction_summary = {}
        for reaction in reactions:
            if reaction.reaction not in reaction_summary:
                reaction_summary[reaction.reaction] = []
            reaction_summary[reaction.reaction].append({
                "user_id": reaction.user_id,
                "user_name": reaction.user.full_name or reaction.user.email
            })
        
        # Determine the display content based on user's preference
        display_content = msg.content
        if user_language in text_translations:
            display_content = text_translations[user_language]
        
        message_data = {
            "id": msg.id,
            "message_id": msg.id,  # Add both for compatibility
            "sender": {
                "id": msg.sender.id,
                "name": msg.sender.full_name or msg.sender.email
            },
            "content": display_content,
            "original_text": msg.content,  # Always include original
            "translated_content": text_translations.get(user_language),  # Specific user translation
            "translations": text_translations,  # All translations
            "translations_cache": text_translations,  # Frontend compatibility
            "original_language": msg.original_language,
            "message_type": msg.message_type,
            "reply_to_id": msg.reply_to_id,
            "timestamp": msg.timestamp.isoformat(),
            "reactions": reaction_summary,
            "is_edited": msg.is_edited
        }
        
        # Add voice-specific fields with correct structure
        if msg.message_type == 'voice':
            message_data.update({
                "audio_urls": audio_urls,  # Frontend expects this structure
                "voice_translations": audio_urls,  # Fallback compatibility
                "voice_file": audio_urls.get(msg.original_language),  # Original voice file
                "voice_duration": msg.voice_duration or 0,
                "audio_duration": msg.voice_duration or 0,  # Frontend compatibility
                "duration": msg.voice_duration or 0,  # Additional compatibility
                "transcription": msg.content  # Voice transcription
            })
        
        message_list.append(message_data)
    
    return {"messages": list(reversed(message_list))}  # Reverse to show oldest first

@router.post("/{group_id}/react")
async def react_to_message(
    group_id: int,
    message_id: int = Form(...),
    reaction: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add or remove reaction to a message"""
    # Check if user is a member
    membership = db.query(group_members).filter(
        group_members.c.group_id == group_id,
        group_members.c.user_id == current_user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Check if reaction already exists
    existing_reaction = db.query(MessageReaction).filter(
        MessageReaction.group_message_id == message_id,
        MessageReaction.user_id == current_user.id,
        MessageReaction.reaction == reaction
    ).first()
    
    if existing_reaction:
        # Remove reaction
        db.delete(existing_reaction)
        action = "removed"
    else:
        # Add reaction
        new_reaction = MessageReaction(
            group_message_id=message_id,
            user_id=current_user.id,
            reaction=reaction
        )
        db.add(new_reaction)
        action = "added"
    
    db.commit()
    
    # Notify group
    await group_manager.broadcast_to_group(group_id, {
        "type": "reaction_update",
        "group_id": group_id,
        "message_id": message_id,
        "user_id": current_user.id,
        "user_name": current_user.full_name or current_user.email,
        "reaction": reaction,
        "action": action
    })
    
    return {"success": True, "action": action}
