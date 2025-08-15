from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..schemas.user import UserRead
from ..models.user import User
from .deps import get_current_user
from ..core.database import get_db

router = APIRouter(prefix="/users", tags=["users"]) 

@router.get("/me", response_model=UserRead)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/", response_model=list[UserRead])
async def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()
