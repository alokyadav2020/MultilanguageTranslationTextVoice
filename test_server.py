from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User

app = FastAPI(title="Test Translation App")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Test server is running"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    # Simple test without authentication
    users = db.query(User).all()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": users[0] if users else None,
            "title": "Test Dashboard",
            "other_users": users[1:] if len(users) > 1 else [],
            "token": "test_token"
        }
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "auth/login.html",
        {"request": request, "title": "Login"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
