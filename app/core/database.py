from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import get_settings
from urllib.parse import urlparse

settings = get_settings()

db_url = settings.effective_database_url
parsed = urlparse(db_url if not db_url.startswith("sqlite:///") else db_url.replace("sqlite:///", "sqlite:////")) if db_url else None

is_sqlite = db_url.startswith("sqlite:")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(db_url, connect_args=connect_args, echo=settings.db_echo, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
