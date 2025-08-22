from sqlalchemy.engine import Engine
import logging
from .voice_migration import run_voice_chat_migrations, verify_voice_chat_schema

logger = logging.getLogger(__name__)

def ensure_user_preferred_language(engine: Engine):
    """Add preferred_language column to users table if it does not exist (SQLite only)."""
    if not engine.url.get_backend_name().startswith("sqlite"):
        return
    with engine.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info(users);")
        cols = [r[1] for r in res.fetchall()]
        if "preferred_language" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN preferred_language VARCHAR(10) NOT NULL DEFAULT 'en';"
            )
            logger.info("✅ Added preferred_language column to users table")


def ensure_user_is_active(engine: Engine):
    if not engine.url.get_backend_name().startswith("sqlite"):
        return
    with engine.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info(users);")
        cols = [r[1] for r in res.fetchall()]
        if "is_active" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1;"
            )
            logger.info("✅ Added is_active column to users table")

def run_startup_migrations(engine: Engine):
    """Run all startup migrations including voice chat features"""
    try:
        logger.info("🚀 Starting database migrations...")
        
        # Run existing migrations
        ensure_user_preferred_language(engine)
        ensure_user_is_active(engine)
        
        # Run voice chat migrations
        logger.info("🎤 Starting voice chat migrations...")
        voice_success = run_voice_chat_migrations(engine)
        
        if voice_success:
            # Verify the schema is correct
            verify_voice_chat_schema(engine)
            logger.info("🎉 All migrations completed successfully!")
        else:
            logger.error("❌ Voice chat migrations failed!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
