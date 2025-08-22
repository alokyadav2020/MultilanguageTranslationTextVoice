from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, Request, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import json
import asyncio
from ..core.database import get_db
from ..core.security import decode_access_token
from ..models.user import User
from ..models.message import Message, ChatroomMember
from ..models.chatroom import Chatroom
from ..core.chat_manager import chat_manager
from ..services.translation import translation_service

router = APIRouter(tags=["chat"])


def get_user_from_token(token: str, db: Session) -> User | None:
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
    return db.query(User).filter(User.email == payload["sub"]).first()


def room_id_for_users(user_a_id: int, user_b_id: int) -> int:
    return int(f"1{min(user_a_id, user_b_id):06d}{max(user_a_id, user_b_id):06d}")


@router.get("/api/chat/history/{other_user_id}")
def chat_history(
    other_user_id: int,
    request: Request,
    token: str | None = Query(None),
    db: Session = Depends(get_db),
):
    # Allow token via query or Authorization header
    if not token:
        auth = request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    me = get_user_from_token(token, db)
    if not me:
        raise HTTPException(status_code=401, detail="Invalid token")
    other = db.query(User).filter(User.id == other_user_id).first()
    if not other:
        raise HTTPException(status_code=404, detail="User not found")
    direct_key = f"direct:{min(me.id, other.id)}:{max(me.id, other.id)}"
    chatroom = (
        db.query(Chatroom)
        .filter(Chatroom.chatroom_name == direct_key, Chatroom.is_group_chat.is_(False))
        .first()
    )
    if not chatroom:
        return []  # no history yet
    messages = (
        db.query(Message)
        .filter(Message.chatroom_id == chatroom.id)
        .order_by(Message.timestamp.asc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": m.id,
            "sender_id": m.sender_id,
            "sender": {
                "id": m.sender.id,
                "name": m.sender.full_name or m.sender.email
            },
            "original_text": m.original_text,
            "original_language": m.original_language,
            "message_type": m.message_type.value if m.message_type else "text",
            "translations_cache": m.translations_cache or {},
            "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            # Voice-specific fields
            "audio_urls": m.audio_urls if hasattr(m, 'audio_urls') else None,
            "audio_duration": m.audio_duration if hasattr(m, 'audio_duration') else None,
        }
        for m in messages
    ]


@router.get("/api/chat/online-users/global")
async def get_global_online_users(
    request: Request,
    token: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get list of all currently online users."""
    # Allow token via query or Authorization header
    if not token:
        auth = request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
    if not token:
        print("DEBUG: No token provided")
        raise HTTPException(status_code=401, detail="Missing token")
    me = get_user_from_token(token, db)
    if not me:
        print("DEBUG: Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Get online users from presence manager
    from ..services.presence_manager import presence_manager
    
    # Update this user's activity since they're making an API call
    await presence_manager.update_user_general_activity(me.id, "api_call")
    
    online_user_ids = list(presence_manager.get_online_users())
    
    print(f"DEBUG: Online users API called by user {me.id}, found {len(online_user_ids)} online users: {online_user_ids}")
    
    return {"online_users": online_user_ids}


@router.websocket("/ws/chat/{other_user_id}")
async def chat_ws(
    websocket: WebSocket,
    other_user_id: int,
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    me = get_user_from_token(token, db)
    if not me:
        await websocket.close(code=4401)
        return
    other = db.query(User).filter(User.id == other_user_id).first()
    if not other:
        await websocket.close(code=4404)
        return

    # Ensure a persistent chatroom exists for this direct pair
    direct_key = f"direct:{min(me.id, other.id)}:{max(me.id, other.id)}"
    chatroom = (
        db.query(Chatroom)
        .filter(Chatroom.chatroom_name == direct_key, Chatroom.is_group_chat.is_(False))
        .first()
    )
    created = False
    if not chatroom:
        chatroom = Chatroom(chatroom_name=direct_key, is_group_chat=False)
        db.add(chatroom)
        db.commit()
        db.refresh(chatroom)
        created = True
    # Ensure both memberships
    existing_member_ids = {m.user_id for m in chatroom.members}
    for uid in (me.id, other.id):
        if uid not in existing_member_ids:
            db.add(ChatroomMember(chatroom_id=chatroom.id, user_id=uid))
    if created or len(existing_member_ids) < 2:
        db.commit()

    room_id = chatroom.id  # use actual chatroom id for room mapping
    await chat_manager.connect(room_id, websocket)
    join_payload = {
        "system": True,
        "event": "join",
        "room_id": room_id,
        "user": {"id": me.id, "email": me.email},
        "timestamp": datetime.utcnow().isoformat(),
    }
    await chat_manager.broadcast(room_id, join_payload)
    
    # Register connection with presence manager
    from ..services.presence_manager import presence_manager
    connection_id = f"{me.id}_{room_id}_{id(websocket)}"
    await presence_manager.add_chat_connection(
        connection_id, websocket, me.id, str(room_id), me.id, other.id
    )
    
    try:
        while True:
            # Receive message data (could be JSON with message type info)
            raw_data = await websocket.receive_text()
            
            # Update user activity in presence manager
            await presence_manager.update_user_activity(connection_id)
            
            # Try to parse as JSON for message type and language info
            try:
                data = json.loads(raw_data)
                message_type = data.get("type", "message")
                
                # Handle typing indicators
                if message_type == "typing":
                    is_typing = data.get("typing", False)
                    await presence_manager.set_user_typing(me.id, str(room_id), is_typing)
                    continue  # Don't process typing as a regular message
                
                # Handle regular messages
                text = data.get("text", raw_data)
                source_lang = data.get("language", "en")  # Default to English
                auto_translate = data.get("auto_translate", True)
            except (json.JSONDecodeError, AttributeError):
                # Fallback for plain text messages
                text = raw_data
                source_lang = "en"
                auto_translate = True
            
            # Generate translations if auto_translate is enabled
            translations_cache = {}
            if auto_translate:
                try:
                    # Get available target languages
                    target_languages = translation_service.get_available_translations(source_lang)
                    
                    # Generate translations for each target language asynchronously
                    if target_languages:
                        translated_texts = await translation_service.translate_batch_async(
                            [text] * len(target_languages), 
                            source_lang, 
                            target_languages[0]  # This will be improved with a proper batch API
                        )
                        
                        # Create async tasks for all target languages
                        translation_tasks = []
                        for target_lang in target_languages:
                            task = translation_service.translate_text_async(text, source_lang, target_lang)
                            translation_tasks.append((target_lang, task))
                        
                        # Execute all translations in parallel
                        if translation_tasks:
                            results = await asyncio.gather(*[task for _, task in translation_tasks], return_exceptions=True)
                            
                            for i, (target_lang, _) in enumerate(translation_tasks):
                                result = results[i]
                                if not isinstance(result, Exception) and result:
                                    translations_cache[target_lang] = result
                except Exception as e:
                    print(f"Translation error: {e}")
                    # Continue without translations if service fails
            
            # Create message with translation data
            msg = Message(
                chatroom_id=room_id,
                sender_id=me.id,
                original_text=text,
                original_language=source_lang,
                translations_cache=translations_cache if translations_cache else None,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            
            # Broadcast message with translation data
            payload = {
                "message_id": msg.id,
                "room_id": room_id,
                "sender": {"id": me.id, "email": me.email, "name": me.full_name or me.email},
                "original_text": msg.original_text,
                "original_language": msg.original_language,
                "translations_cache": msg.translations_cache or {},
                "message_type": msg.message_type.value if hasattr(msg.message_type, "value") else str(msg.message_type),
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.utcnow().isoformat(),
            }
            await chat_manager.broadcast(room_id, payload)
    except WebSocketDisconnect:
        # Remove connection from presence manager
        await presence_manager.remove_chat_connection(connection_id)
        
        leave_payload = {
            "system": True,
            "event": "leave",
            "room_id": room_id,
            "user": {"id": me.id, "email": me.email},
            "timestamp": datetime.utcnow().isoformat(),
        }
        await chat_manager.broadcast(room_id, leave_payload)
        chat_manager.disconnect(room_id, websocket)
