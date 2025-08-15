from fastapi import FastAPI, Request, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import logging
from .core.config import get_settings
from .core.database import engine, Base, get_db
from .core.migrate import run_startup_migrations
from .core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from .models.user import User
from .api import auth, users, chat_ws, translation
from . import models

logger = logging.getLogger(__name__)

settings = get_settings()

# Create / update tables (create_all does not alter, so we run a small migration helper next)
Base.metadata.create_all(bind=engine)
run_startup_migrations(engine)

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
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat_ws.router)
app.include_router(translation.router)

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
        return None
    if not payload or "sub" not in payload:
        logger.debug("Invalid payload or missing sub")
        return None
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        logger.debug("User not found for sub: %s", payload.get("sub"))
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
        return RedirectResponse("/login", 302)
    # Disallow chatting with self (redirect to dashboard)
    if other_user_id == current_user.id:
        return RedirectResponse("/", 302)
    remote_user = db.query(User).filter(User.id == other_user_id).first()
    if not remote_user:
        return RedirectResponse("/", 302)
    token = request.session.get("token")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": current_user,  # Add this for navbar
            "current_user": current_user,
            "remote_user": remote_user,
            "token": token,
            "title": f"Chat with {remote_user.full_name or remote_user.email}",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(get_current_ui_user), db: Session = Depends(get_db)):
    if user:
        other_users = db.query(User).filter(User.id != user.id).all()
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "title": "Dashboard",
                "other_users": other_users,
            },
        )
    return RedirectResponse(url="/login", status_code=302)


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
    request.session.pop("token", None)
    request.session.setdefault("flash", []).append(("info", "Logged out"))
    return RedirectResponse(url="/login", status_code=303)
