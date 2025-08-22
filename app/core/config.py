import os
from functools import lru_cache

class Settings:
    project_name: str = "Translation App API"
    secret_key: str = os.getenv("SECRET_KEY", "CHANGE_ME_SECRET")
    access_token_expire_minutes: int = 60 * 24  # 1 day
    algorithm: str = "HS256"
    # Default local SQLite database (fallback)
    sqlite_db: str = "sqlite:///./app.db"
    # Optional unified database URL (overrides sqlite_db if provided), e.g.
    # postgresql+psycopg://user:pass@localhost:5432/mydb
    # mysql+pymysql://user:pass@localhost:3306/mydb
    # mssql+pyodbc://user:pass@localhost:1433/mydb?driver=ODBC+Driver+18+for+SQL+Server
    database_url: str = None
    db_echo: bool = False  # enable SQL logging if needed
    cors_origins: list = ["*"]

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.db_echo = os.getenv("DB_ECHO", "false").lower() == "true"

    @property
    def effective_database_url(self) -> str:
        return self.database_url or self.sqlite_db

@lru_cache
def get_settings() -> Settings:
    return Settings()
