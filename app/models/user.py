# from sqlalchemy import String, Integer, DateTime
# from sqlalchemy.orm import Mapped, mapped_column
# from datetime import datetime
# from ..core.database import Base

# class User(Base):
#     __tablename__ = "users"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
#     full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
#     hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
#     is_active: Mapped[bool] = mapped_column(default=True)
#     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, text, func
from sqlalchemy.orm import relationship
from ..core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)          # user_id
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default=text('1'))  # restore to match existing table
    preferred_language = Column(String(10), nullable=False, default="en")  # NEW
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    messages = relationship("Message", back_populates="sender", cascade="all,delete")
    memberships = relationship("ChatroomMember", back_populates="user", cascade="all,delete")
