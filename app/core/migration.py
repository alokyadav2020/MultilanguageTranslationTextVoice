"""
Database migration to add original_language column to messages table.
This ensures compatibility with existing data.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from ..core.database import engine

def ensure_message_original_language():
    """
    Ensure the original_language column exists in the messages table.
    This function is safe to run multiple times.
    """
    with Session(engine) as db:
        try:
            # Check if column exists
            result = db.execute(text("""
                SELECT COUNT(*) FROM pragma_table_info('messages') 
                WHERE name = 'original_language'
            """)).fetchone()
            
            if result[0] == 0:
                # Column doesn't exist, add it
                db.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN original_language VARCHAR(10) DEFAULT 'en'
                """))
                
                # Update existing records to have 'en' as default
                db.execute(text("""
                    UPDATE messages 
                    SET original_language = 'en' 
                    WHERE original_language IS NULL
                """))
                
                db.commit()
                print("✓ Added original_language column to messages table")
            else:
                print("✓ original_language column already exists")
                
            # Check if translations_cache column exists
            result = db.execute(text("""
                SELECT COUNT(*) FROM pragma_table_info('messages') 
                WHERE name = 'translations_cache'
            """)).fetchone()
            
            if result[0] == 0:
                # Column doesn't exist, add it
                db.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN translations_cache JSON
                """))
                
                db.commit()
                print("✓ Added translations_cache column to messages table")
            else:
                print("✓ translations_cache column already exists")
                
        except Exception as e:
            print(f"Migration error: {e}")
            db.rollback()
            raise

def run_migration():
    """Run the migration."""
    print("Running database migration for translation support...")
    ensure_message_original_language()
    print("Migration completed successfully!")

if __name__ == "__main__":
    run_migration()
