# 🌍 Real-Time Voice Translation with SeamlessM4T

## ✨ Overview

This project implements real-time voice-to-voice translation using Facebook's SeamlessM4T model for Arabic, English, and French languages during WebRTC voice calls.

### 🎯 Key Features
- **🗣️ Voice-to-Voice Translation**: Direct speech translation without intermediate text
- **🌍 3 Languages**: Arabic ↔ English ↔ French
- **⚡ Real-time Processing**: 2-second audio chunks with minimal latency
- **📱 WebRTC Integration**: Peer-to-peer voice calls
- **🎮 GPU Acceleration**: Automatic CUDA detection and optimization
- **👤 User Management**: JWT authentication and user registration
- **🔒 Secure**: Password hashing and token-based authentication
- List users endpoint `/users/`
- CORS enabled
- SQLite database (`app.db`)

## Quick Start

### 1. Install dependencies
```powershell
# (Recommended) create & activate virtual env with uv
uv venv
.\.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

### 2. Run the server
```powershell
uvicorn app.main:app --reload
```

### 3. Open docs
Visit: http://127.0.0.1:8000/docs

### Example Requests
```bash
# Register
curl -X POST http://127.0.0.1:8000/auth/register -H "Content-Type: application/json" -d '{"email":"user@example.com","password":"secret","full_name":"User"}'

# Login
curl -X POST -d "username=user@example.com&password=secret" -H "Content-Type: application/x-www-form-urlencoded" http://127.0.0.1:8000/auth/login

# Authorized request
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/users/me
```

## Environment Variables
Create a `.env` file (optional):
```
SECRET_KEY=supersecretkey
ACCESS_TOKEN_EXPIRE_MINUTES=1440
SQLITE_DB=sqlite:///./app.db
```

Adjust in `app/core/config.py` or via environment.

## Notes
- Replace the default `secret_key` for production.
- For persistent storage path, point `sqlite_db` to a folder outside ephemeral containers.
