# from pydantic import BaseModel, EmailStr
# from datetime import datetime

# class UserBase(BaseModel):
#     email: EmailStr
#     full_name: str | None = None

# class UserCreate(UserBase):
#     password: str

# class UserLogin(BaseModel):
#     email: EmailStr
#     password: str

# class UserRead(UserBase):
#     id: int
#     is_active: bool
#     created_at: datetime

#     class Config:
#         from_attributes = True


# filepath: app/schemas/user.py
# ...existing code...
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    preferred_language: str = "en"      # NEW

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    preferred_language: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserStats(BaseModel):
    total_messages: int
    voice_messages: int
    total_voice_duration: float  # in seconds
    preferred_language: str
# ...existing code...

