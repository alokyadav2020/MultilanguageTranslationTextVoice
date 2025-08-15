# Translation Production App API

FastAPI backend providing user registration, login (JWT), and basic user endpoints using SQLite.

## Features
- User registration (email, password, optional full name)
- Secure password hashing (bcrypt via passlib)
- JWT authentication (access token)
- Login endpoint (OAuth2 password flow compatible)
- Current user endpoint `/users/me`
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
