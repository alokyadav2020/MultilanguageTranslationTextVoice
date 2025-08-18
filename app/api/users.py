from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..schemas.user import UserRead
from ..models.user import User
from .deps import get_current_user
from ..core.database import get_db

router = APIRouter(prefix="/users", tags=["users"]) 

class LanguageUpdateRequest(BaseModel):
    preferred_language: str

@router.get("/me", response_model=UserRead)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/", response_model=list[UserRead])
async def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@router.post("/update-language")
async def update_user_language(
    language_data: LanguageUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's preferred language"""
    # Validate language code
    valid_languages = ["en", "fr", "ar"]
    if language_data.preferred_language not in valid_languages:
        raise HTTPException(status_code=400, detail="Invalid language code")
    
    # Update user's preferred language
    current_user.preferred_language = language_data.preferred_language
    db.commit()
    
    return {"message": "Language preference updated successfully"}
