from fastapi import FastAPI, Request, Depends, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import logging
import os
import uuid
import csv
import io
import re
from datetime import datetime
from pathlib import Path
from .core.config import get_settings
from .core.database import engine, Base, get_db
from .core.migrate import run_startup_migrations
from .core.group_migration import run_group_migration
from .core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from .models.user import User
from .api import auth, users, chat_ws, translation, voice_chat, voice_call, groups, group_ws
from .api import enhanced_voice_call  # Import enhanced voice call API
from .api import chat_summary  # Import chat summary API
from . import models
from pathlib import Path

logger = logging.getLogger(__name__)

settings = get_settings()

# Create / update tables (create_all does not alter, so we run a small migration helper next)
Base.metadata.create_all(bind=engine)
run_startup_migrations(engine)

# Run group migration
run_group_migration(engine)

app = FastAPI(title=settings.project_name)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    session_cookie="app_session",
    same_site="lax",
    https_only=False,
    path="/",
    max_age=60*60*24*7,
)

BASE_DIR = Path(__file__).resolve().parent.parent  # Points to project root
STATIC_DIR = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to handle secure context for voice features
@app.middleware("http")
async def secure_context_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Add headers to indicate if secure context is available
    if request.url.scheme == "https" or request.url.hostname in ["localhost", "127.0.0.1"]:
        response.headers["X-Secure-Context"] = "true"
    else:
        response.headers["X-Secure-Context"] = "false"
        response.headers["X-HTTPS-Upgrade"] = f"https://{request.url.hostname}:{request.url.port or 443}{request.url.path}"
    
    return response

# ===================================
# CHAT HISTORY DOWNLOAD ENDPOINT (Before router includes)
# ===================================

@app.get("/api/chat/download-history/{other_user_id:int}")
async def download_chat_history_csv(
    other_user_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Download chat history as CSV file
    Available at: GET /api/chat/download-history/{other_user_id}
    """
    try:
        # Get current user from session token
        current_user = get_current_ui_user(request, db)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Validate other user exists
        from .models.user import User
        other_user = db.query(User).filter(User.id == other_user_id).first()
        if not other_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get chat history
        from .models.chatroom import Chatroom
        from .models.message import Message
        
        # Find chatroom between the two users
        direct_key = f"direct:{min(current_user.id, other_user_id)}:{max(current_user.id, other_user_id)}"
        chatroom = (
            db.query(Chatroom)
            .filter(Chatroom.chatroom_name == direct_key, Chatroom.is_group_chat.is_(False))
            .first()
        )
        
        if not chatroom:
            # No chat history exists, return empty CSV
            messages = []
        else:
            # Get all messages from the chatroom
            messages = (
                db.query(Message)
                .filter(Message.chatroom_id == chatroom.id)
                .order_by(Message.timestamp.asc())
                .all()
            )
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # Write CSV headers
        writer.writerow([
            'Timestamp',
            'Sender',
            'Message Type',
            'Original Text',
            'Original Language',
            'Translations',
            'Audio Duration (seconds)',
            'Message ID'
        ])
        
        # Write message data
        for message in messages:
            # Get sender name
            sender_name = message.sender.full_name or message.sender.email
            
            # Format timestamp
            timestamp_str = message.timestamp.strftime('%Y-%m-%d %H:%M:%S') if message.timestamp else ''
            
            # Format translations
            translations_str = ''
            if message.translations_cache:
                translations_list = []
                for lang, text in message.translations_cache.items():
                    translations_list.append(f"{lang}: {text}")
                translations_str = ' | '.join(translations_list)
            
            # Audio duration
            audio_duration = message.audio_duration if message.audio_duration else ''
            
            writer.writerow([
                timestamp_str,
                sender_name,
                message.message_type.value if message.message_type else 'text',
                message.original_text,
                message.original_language,
                translations_str,
                audio_duration,
                message.id
            ])
        
        # Prepare file content
        csv_content = output.getvalue()
        output.close()
        
        # Generate filename
        other_user_name = (other_user.full_name or other_user.email).replace(' ', '_')
        # Remove special characters for filename safety
        other_user_name = re.sub(r'[<>:"/\\|?*]', '', other_user_name)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        filename = f"chat_history_{other_user_name}_{current_date}.csv"
        
        # Return CSV file as response
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Chat history download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate chat history file")


@app.get("/api/chat/test-download")
async def test_download_endpoint():
    """Test endpoint to verify route registration"""
    return {"message": "Download endpoint is working", "status": "ok"}


@app.get("/api/test-translation")
async def test_translation_endpoint():
    """Test endpoint to verify translation service is working"""
    try:
        from .services.translation import translation_service
        result = await translation_service.translate_text_async("Hello", "en", "fr")
        return {
            "status": "ok", 
            "test_translation": result,
            "message": "Translation service is working",
            "supported_languages": list(translation_service.supported_languages)
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "message": "Translation service failed"}


app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat_ws.router)
app.include_router(translation.router)
app.include_router(voice_chat.router)
app.include_router(voice_call.router)
app.include_router(enhanced_voice_call.router)  # Enhanced voice call with SeamlessM4T
app.include_router(groups.router)
app.include_router(group_ws.router)
app.include_router(chat_summary.router)  # Chat summary with AI


@app.on_event("startup")
async def startup_event():
    """Initialize services and preload models for better performance."""
    from .services.translation import translation_service
    # from .services.call_manager import start_heartbeat_task
    # import asyncio
    
    logger.info("🚀 Starting translation service optimization...")
    
    # Preload common translation models in background (non-blocking)
    import threading
    
    # def preload_in_background():
    #     """Run preload in background thread to avoid blocking startup"""
    #     try:
    #         translation_service.preload_models()
    #         logger.info("🎯 Background model preloading completed")
    #     except Exception as e:
    #         logger.error(f"❌ Background preloading failed: {e}")
    
    # # Start preloading in background thread using threading instead of asyncio.to_thread
    # thread = threading.Thread(target=preload_in_background, daemon=True)
    # thread.start()
    
    logger.info("✅ Translation service startup complete")
    
    # Start call manager heartbeat task
    logger.info("📞 Starting call manager heartbeat...")
    # Temporarily disabled for debugging
    # asyncio.create_task(start_heartbeat_task())
    logger.info("✅ Call manager heartbeat started (disabled for debugging)")


@app.get("/_debug/session")
def debug_session(request: Request):
    return {"session": dict(request.session)}


@app.get("/health")
async def health():
    return {"status": "ok"}


def get_current_ui_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.session.get("token")
    logger.debug("Session keys: %s, token present: %s", list(request.session.keys()), "token" in request.session)
    if not token:
        logger.debug("No token in session")
        return None
    try:
        payload = decode_access_token(token)
    except Exception as e:
        logger.debug("Token decode failed: %s", e)
        # Clear invalid session
        request.session.clear()
        return None
    if not payload or "sub" not in payload:
        logger.debug("Invalid payload or missing sub")
        # Clear invalid session
        request.session.clear()
        return None
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        logger.debug("User not found for sub: %s", payload.get("sub"))
        # Clear session for non-existent user
        request.session.clear()
        return None
    else:
        logger.debug("Found user: %s", user.email)
    return user


@app.get("/chat/{other_user_id}", response_class=HTMLResponse)
async def chat_page(
    other_user_id: int,
    request: Request,
    current_user: User | None = Depends(get_current_ui_user),
    db: Session = Depends(get_db),
):
    # Must be logged in
    if not current_user:
        response = RedirectResponse("/login", 302)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    # Disallow chatting with self (redirect to dashboard)
    if other_user_id == current_user.id:
        response = RedirectResponse("/", 302)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    remote_user = db.query(User).filter(User.id == other_user_id).first()
    if not remote_user:
        response = RedirectResponse("/", 302)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    token = request.session.get("token")
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": current_user,  # Add this for navbar
            "current_user": current_user,
            "remote_user": remote_user,
            "token": token,
            "title": f"Chat + Voice Calls with {remote_user.full_name or remote_user.email}",
        },
    )
    # Add cache control headers for authenticated content
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/groups", response_class=HTMLResponse)
async def groups_page(
    request: Request, 
    user: User | None = Depends(get_current_ui_user)
):
    """Groups page for multilingual group conversations"""
    if not user:
        return RedirectResponse("/login", 302)
    
    token = request.session.get("token")
    response = templates.TemplateResponse(
        "groups.html",
        {
            "request": request,
            "user": user,
            "title": "Groups - Multilingual Chat",
            "token": token
        }
    )
    
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(get_current_ui_user), db: Session = Depends(get_db)):
    if user:
        # Track user dashboard activity
        from .services.presence_manager import presence_manager
        await presence_manager.update_user_general_activity(user.id, "dashboard")
        
        # Get the token from session for dashboard API calls
        token = request.session.get("token")
        other_users = db.query(User).filter(User.id != user.id).all()
        response = templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "title": "Dashboard",
                "other_users": other_users,
                "token": token,  # Add token for API calls
            },
        )
        # Add cache control headers for authenticated content
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    
    # Create redirect response with cache control
    response = RedirectResponse(url="/login", status_code=302)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User | None = Depends(get_current_ui_user)):
    if user:
        return RedirectResponse("/", 302)
    return templates.TemplateResponse("auth/register.html", {"request": request, "user": user, "title": "Register"})



@app.post("/register")
async def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    preferred_language: str = Form(...),   # NEW
    full_name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        request.session.setdefault("flash", []).append(("danger", "Email already registered"))
        return RedirectResponse(url="/register", status_code=303)
    user = User(
        email=email,
        full_name=full_name,
        preferred_language=preferred_language,
        hashed_password=get_password_hash(password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    request.session.setdefault("flash", []).append(("success", "Registration successful. Please login."))
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User | None = Depends(get_current_ui_user)):
    if user:
        return RedirectResponse("/", 302)
    return templates.TemplateResponse("auth/login.html", {"request": request, "user": user, "title": "Login"})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == username).first()
    if not user or not verify_password(password, user.hashed_password):
        request.session.setdefault("flash", []).append(("danger", "Invalid credentials"))
        return RedirectResponse(url="/login", status_code=303)
    token = create_access_token({"sub": user.email})
    request.session["token"] = token
    request.session.setdefault("flash", []).append(("success", "Logged in"))
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    # Get current user before clearing session
    current_user = None
    token = request.session.get("token")
    if token:
        try:
            from .core.security import decode_access_token
            from .core.database import get_db
            from .models.user import User
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                # Get database session
                db = next(get_db())
                current_user = db.query(User).filter(User.email == payload["sub"]).first()
                db.close()
        except Exception:
            pass  # Ignore errors, just proceed with logout
    
    # Clear user from presence manager if found
    if current_user:
        try:
            from .services.presence_manager import presence_manager
            # Remove user from all presence tracking
            await presence_manager.remove_user_from_presence(current_user.id)
        except Exception:
            pass  # Ignore errors, just proceed with logout
    
    # Clear all session data
    request.session.clear()
    # Add logout message to new clean session
    request.session.setdefault("flash", []).append(("info", "Logged out successfully"))
    
    # Create response with proper cache control headers
    response = RedirectResponse(url="/login", status_code=303)
    
    # Add headers to prevent caching of authenticated content
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


@app.get("/voice-call", response_class=HTMLResponse)
async def voice_call_page(
    request: Request, 
    call_id: str = None,
    incoming: bool = False,
    participant: str = None,
    user: User | None = Depends(get_current_ui_user)
):
    """Voice call window page"""
    if not user:
        return RedirectResponse("/login", 302)
    
    # Get token for WebSocket authentication
    token = request.session.get("token")
    if not token:
        return RedirectResponse("/login", 302)
    
    response = templates.TemplateResponse(
        "voice_call.html",
        {
            "request": request,
            "user": user,
            "title": f"Voice Call - {participant or 'Unknown'}",
            "call_id": call_id,
            "incoming": incoming,
            "participant": participant,
            "token": token
        }
    )
    
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/enhanced-voice-call", response_class=HTMLResponse)
async def enhanced_voice_call_page(
    request: Request, 
    call_id: str = None,
    incoming: bool = False,
    participant: str = None,
    language: str = "en",
    user: User | None = Depends(get_current_ui_user)
):
    """Enhanced voice call window with real-time translation"""
    if not user:
        return RedirectResponse("/login", 302)
    
    # Get token for WebSocket authentication
    token = request.session.get("token")
    if not token:
        return RedirectResponse("/login", 302)
    
    # Validate language
    supported_languages = ["ar", "en", "fr"]
    if language not in supported_languages:
        language = "en"
    
    response = templates.TemplateResponse(
        "enhanced_voice_call.html",
        {
            "request": request,
            "user": user,
            "title": f"Enhanced Voice Call - {participant or 'Unknown'}",
            "call_id": call_id,
            "incoming": incoming,
            "participant": participant,
            "language": language,
            "token": token,
            "supported_languages": supported_languages
        }
    )
    
    # Prevent caching for security
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


# ===================================
# VOICE CHAT API ENDPOINTS
# ===================================

@app.post("/api/voice/upload-message")
async def upload_voice_message(
    audio: UploadFile = File(...),
    language: str = Form(...),
    recipient_id: int = Form(...),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Upload and process voice message endpoint
    Available at: POST /api/voice/upload-message
    """
    try:
        # Get current user from session token
        current_user = get_current_ui_user(request, db)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Validate recipient exists
        from .models.user import User
        recipient = db.query(User).filter(User.id == recipient_id).first()
        if not recipient:
            raise HTTPException(status_code=404, detail="Recipient not found")
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("app/static/uploads/voice")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.webm"
        temp_path = upload_dir / temp_filename
        
        content = await audio.read()
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        try:
            # Process voice message using voice service
            from .services.voice_service import voice_service
            result = await voice_service.process_voice_message(
                audio_file_path=str(temp_path),
                language=language,
                sender_id=current_user.id,
                recipient_id=recipient_id,
                db=db
            )
            
            # Validate result data and provide defaults for None values
            transcribed_text = result.get("transcribed_text") or "[Voice message - transcription unavailable]"
            translations = result.get("translations") or {}
            audio_urls = result.get("audio_urls") or {}
            duration = result.get("duration") or 0
            message = result.get("message")
            
            # Ensure message object exists
            if not message:
                logger.error("Voice service did not return a message object")
                return {
                    "success": False,
                    "message_id": 0,
                    "transcribed_text": "[Voice message - processing failed]",
                    "translations": {},
                    "audio_urls": {},
                    "audio_duration": 0,
                    "error": "Failed to create message record"
                }
            
            # Broadcast message via WebSocket to the direct chat room
            from .core.chat_manager import chat_manager
            from .models.chatroom import Chatroom
            
            # Find the direct chat room between sender and recipient
            direct_key = f"direct:{min(current_user.id, recipient_id)}:{max(current_user.id, recipient_id)}"
            direct_room = db.query(Chatroom).filter(
                Chatroom.chatroom_name == direct_key, 
                Chatroom.is_group_chat.is_(False)
            ).first()
            
            if direct_room:
                message_data = {
                    "type": "message",
                    "message_type": "voice",
                    "sender": {
                        "id": current_user.id,
                        "name": current_user.full_name or current_user.email
                    },
                    "original_text": transcribed_text,
                    "original_language": language,
                    "translations_cache": translations,
                    "audio_urls": audio_urls,
                    "audio_duration": duration,
                    "timestamp": message.timestamp.isoformat() if hasattr(message, 'timestamp') else ""
                }
                
                # Broadcast to the direct chat room
                await chat_manager.broadcast(direct_room.id, message_data)
            else:
                logger.warning(f"No direct chat room found between users {current_user.id} and {recipient_id}")
            
            return {
                "success": True,
                "message_id": message.id,
                "transcribed_text": transcribed_text,
                "translations": translations,
                "audio_urls": audio_urls,
                "audio_duration": duration
            }
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Voice message upload error: {e}")
        return {
            "success": False,
            "message_id": 0,
            "transcribed_text": "[Voice message - upload failed]",
            "translations": {},
            "audio_urls": {},
            "audio_duration": 0,
            "error": str(e)
        }


@app.get("/api/voice/audio/{file_name}")
async def serve_audio_file(file_name: str):
    """
    Serve audio files
    Available at: GET /api/voice/audio/{file_name}
    """
    upload_dir = Path("app/static/uploads/voice")
    file_path = upload_dir / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=file_name
    )


@app.get("/api/voice/stats/{user_id}")
async def get_voice_stats(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get voice chat statistics for a user
    Available at: GET /api/voice/stats/{user_id}
    """
    # Get current user from session
    current_user = get_current_ui_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check if requesting own stats
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    from .models.user import User
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        voice_count = user.get_voice_message_count(db)
        total_duration = user.get_total_voice_duration(db)
        
        return {
            "user_id": user_id,
            "total_voice_messages": voice_count,
            "total_voice_duration": total_duration,
            "average_message_duration": total_duration / voice_count if voice_count > 0 else 0
        }
    except Exception as e:
        logger.error(f"Voice stats error: {e}")
        return {
            "user_id": user_id,
            "total_voice_messages": 0,
            "total_voice_duration": 0,
            "average_message_duration": 0
        }


@app.get("/api/voice/test")
async def test_voice_endpoint():
    """
    Test endpoint to verify voice API is working
    Available at: GET /api/voice/test
    """
    try:
        from .services.voice_service import voice_service, VOICE_LIBS_AVAILABLE
        
        return {
            "status": "ok",
            "voice_libraries_available": VOICE_LIBS_AVAILABLE,
            "upload_directory_exists": Path("app/static/uploads/voice").exists(),
            "message": "Voice chat API is ready!" if VOICE_LIBS_AVAILABLE else "Voice libraries not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "voice_libraries_available": False,
            "error": str(e)
        }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
if __name__ == "__main__":
    # import uvicorn
    # import os
    
    # # Always run HTTP for ngrok (simpler)
    # # print("🚀 Starting HTTP server for ngrok tunneling...")
    # # print("📡 Use ngrok to create HTTPS tunnel")
    # # uvicorn.run(
    # #     "main:app",
    # #     host="0.0.0.0",
    # #     port=443,  # Standard HTTPS port; use 8443 if 443 is blocked
    # #     ssl_keyfile="key.pem",
    # #     ssl_certfile="cert.pem"
    # # )
    # uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    import uvicorn
    import ssl
    import os
    from pathlib import Path
    
    # SSL configuration
    ssl_cert_path = Path("cert.pem")
    ssl_key_path = Path("key.pem")
    
    # Check if SSL certificates exist
    if ssl_cert_path.exists() and ssl_key_path.exists():
        print("🔒 Starting server with HTTPS...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_cert_reqs=ssl.CERT_NONE,
            ssl_check_hostname=False,
            ssl_ciphers="ALL"
        )
    else:
        print("⚠️  SSL certificates not found. Starting with HTTP (voice features will be limited)...")
        print("📝 To enable voice features, generate SSL certificates:")
        print("   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )