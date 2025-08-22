#!/usr/bin/env python3
"""
Standalone script to run voice chat migrations manually
Usage: python run_voice_migrations.py
"""

import sys
import os
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.database import engine
from app.core.voice_migration import run_voice_chat_migrations, verify_voice_chat_schema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run voice chat migrations"""
    try:
        logger.info("🚀 Starting voice chat migrations...")
        
        # Run migrations
        success = run_voice_chat_migrations(engine)
        
        if success:
            # Verify schema
            verify_success = verify_voice_chat_schema(engine)
            
            if verify_success:
                logger.info("✅ Voice chat migrations completed successfully!")
                logger.info("🎉 Your database is ready for voice chat features!")
                
                logger.info("\n📋 Next steps:")
                logger.info("1. Install voice processing libraries: pip install speech-recognition gtts pydub")
                logger.info("2. For Linux: sudo apt-get install ffmpeg espeak espeak-data")
                logger.info("3. Restart your FastAPI application")
                logger.info("4. Test voice chat functionality")
                
            else:
                logger.error("❌ Schema verification failed!")
                return False
        else:
            logger.error("❌ Migration failed!")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
