# filepath: app/core/group_migration.py
"""
Migration script to create group-related tables
"""
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

def run_group_migration(engine):
    """Run migration to create group tables"""
    try:
        with engine.connect() as connection:
            # Create groups table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    group_type VARCHAR(20) DEFAULT 'private',
                    created_by INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    max_members INTEGER DEFAULT 100,
                    default_language VARCHAR(10) DEFAULT 'en',
                    profile_picture VARCHAR(255),
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            """))
            
            # Create group_members table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS group_members (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role VARCHAR(20) DEFAULT 'member',
                    joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    preferred_language VARCHAR(10) DEFAULT 'en',
                    notifications_enabled BOOLEAN DEFAULT 1,
                    voice_language VARCHAR(10) DEFAULT 'en',
                    PRIMARY KEY (group_id, user_id),
                    FOREIGN KEY (group_id) REFERENCES groups (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            
            # Create group_messages table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS group_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    sender_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    original_language VARCHAR(10) NOT NULL,
                    message_type VARCHAR(20) DEFAULT 'text',
                    reply_to_id INTEGER,
                    voice_file_path VARCHAR(255),
                    voice_duration INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_edited BOOLEAN DEFAULT 0,
                    edited_at DATETIME,
                    FOREIGN KEY (group_id) REFERENCES groups (id),
                    FOREIGN KEY (sender_id) REFERENCES users (id),
                    FOREIGN KEY (reply_to_id) REFERENCES group_messages (id)
                )
            """))
            
            # Create group_message_translations table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS group_message_translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    language VARCHAR(10) NOT NULL,
                    translated_content TEXT NOT NULL,
                    translation_type VARCHAR(10) DEFAULT 'text',
                    voice_file_path VARCHAR(255),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES group_messages (id)
                )
            """))
            
            # Create message_reactions table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS message_reactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_message_id INTEGER,
                    user_id INTEGER NOT NULL,
                    reaction VARCHAR(10) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (group_message_id) REFERENCES group_messages (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            
            connection.commit()
            logger.info("✅ Group tables created successfully")
            
    except Exception as e:
        logger.error(f"❌ Error creating group tables: {e}")
        raise
