# filepath: app/core/voice_migration.py
from sqlalchemy import text, inspect
import logging

logger = logging.getLogger(__name__)

def run_voice_chat_migrations(engine):
    """
    Migration script for voice chat feature
    Adds required columns and tables for voice message support
    """
    with engine.connect() as conn:
        try:
            inspector = inspect(engine)
            
            # Check if messages table exists
            if 'messages' not in inspector.get_table_names():
                logger.error("Messages table does not exist. Run basic migrations first.")
                return False
            
            # Get current columns in messages table
            current_columns = [col['name'] for col in inspector.get_columns('messages')]
            logger.info(f"Current message columns: {current_columns}")
            
            # Track changes made
            changes_made = []
            
            # Add voice-related columns to messages table
            voice_columns = {
                'audio_urls': 'JSON',
                'audio_duration': 'REAL',
                'audio_file_size': 'INTEGER'
            }
            
            for column_name, column_type in voice_columns.items():
                if column_name not in current_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE messages ADD COLUMN {column_name} {column_type}"))
                        changes_made.append(f"Added {column_name} column")
                        logger.info(f"✅ Added {column_name} column to messages table")
                    except Exception as e:
                        logger.error(f"❌ Failed to add {column_name}: {e}")
                        raise
                else:
                    logger.info(f"⏭️  Column {column_name} already exists")
            
            # Create audio_files table for detailed audio file tracking
            if 'audio_files' not in inspector.get_table_names():
                create_audio_files_table = text("""
                CREATE TABLE audio_files (
                    id INTEGER PRIMARY KEY,
                    message_id INTEGER NOT NULL,
                    language VARCHAR(10) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    file_url VARCHAR(500) NOT NULL,
                    file_size INTEGER NOT NULL,
                    duration REAL,
                    mime_type VARCHAR(50) DEFAULT 'audio/mp3',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES messages (id) ON DELETE CASCADE
                )
                """)
                
                conn.execute(create_audio_files_table)
                changes_made.append("Created audio_files table")
                logger.info("✅ Created audio_files table")
            else:
                logger.info("⏭️  audio_files table already exists")
            
            # Create indexes for better performance
            indexes_to_create = [
                ("idx_messages_message_type", "CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type)"),
                ("idx_messages_original_language", "CREATE INDEX IF NOT EXISTS idx_messages_original_language ON messages(original_language)"),
                ("idx_audio_files_message_id", "CREATE INDEX IF NOT EXISTS idx_audio_files_message_id ON audio_files(message_id)"),
                ("idx_audio_files_language", "CREATE INDEX IF NOT EXISTS idx_audio_files_language ON audio_files(language)")
            ]
            
            for index_name, index_sql in indexes_to_create:
                try:
                    conn.execute(text(index_sql))
                    changes_made.append(f"Created index {index_name}")
                    logger.info(f"✅ Created index {index_name}")
                except Exception as e:
                    logger.warning(f"⚠️  Index {index_name} creation failed (might already exist): {e}")
            
            # Commit all changes
            conn.commit()
            
            if changes_made:
                logger.info("🎉 Voice chat migration completed successfully!")
                logger.info("Changes made:")
                for change in changes_made:
                    logger.info(f"  - {change}")
            else:
                logger.info("✅ All voice chat features already exist - no migration needed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Voice chat migration failed: {e}")
            conn.rollback()
            raise

def verify_voice_chat_schema(engine):
    """
    Verify that all required voice chat columns and tables exist
    """
    with engine.connect() as conn:
        try:
            inspector = inspect(engine)
            
            # Check messages table columns
            messages_columns = [col['name'] for col in inspector.get_columns('messages')]
            required_columns = ['audio_urls', 'audio_duration', 'audio_file_size', 'message_type', 'original_audio_path']
            
            missing_columns = [col for col in required_columns if col not in messages_columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in messages table: {missing_columns}")
                return False
            
            # Check if audio_files table exists
            if 'audio_files' not in inspector.get_table_names():
                logger.warning("audio_files table does not exist (optional but recommended)")
            
            logger.info("✅ Voice chat schema verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema verification failed: {e}")
            return False

def rollback_voice_chat_migrations(engine):
    """
    Rollback voice chat migrations (USE WITH CAUTION - for development only)
    """
    with engine.connect() as conn:
        try:
            inspector = inspect(engine)
            
            # Drop audio_files table
            if 'audio_files' in inspector.get_table_names():
                conn.execute(text("DROP TABLE audio_files"))
                logger.info("🗑️  Dropped audio_files table")
            
            # Remove columns from messages table (SQLite doesn't support DROP COLUMN easily)
            # This would require recreating the table, so we'll just log a warning
            logger.warning("⚠️  Cannot easily remove columns from messages table in SQLite")
            logger.warning("⚠️  Consider recreating the database if you need a clean rollback")
            
            conn.commit()
            logger.info("🔄 Partial rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            conn.rollback()
            raise
