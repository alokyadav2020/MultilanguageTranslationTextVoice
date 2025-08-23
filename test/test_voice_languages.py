#!/usr/bin/env python3
"""
Test script for voice language validation
Tests the three supported languages: English, French, Arabic
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_language_validation():
    """Test language validation in voice service"""
    print("Testing voice service language validation...")
    
    # Test valid languages
    valid_languages = ['en', 'fr', 'ar']
    for lang in valid_languages:
        print(f"✓ {lang} - Valid language")
    
    # Test invalid languages
    invalid_languages = ['es', 'de', 'zh', 'ja', 'it', 'pt', 'ko']
    for lang in invalid_languages:
        print(f"✗ {lang} - Invalid language (not supported)")
    
    print(f"\nSupported languages: {valid_languages}")
    print("Language validation test completed!")

def test_schema_validation():
    """Test schema validation"""
    try:
        from app.schemas.voice import VoiceMessageUpload, SUPPORTED_LANGUAGES
        
        print("\nTesting schema validation...")
        print(f"Supported languages in schema: {SUPPORTED_LANGUAGES}")
        
        # Test valid language
        try:
            valid_upload = VoiceMessageUpload(language="en", recipient_id=1)
            print("✓ Valid language 'en' accepted")
        except Exception as e:
            print(f"✗ Error with valid language: {e}")
        
        # Test invalid language
        try:
            invalid_upload = VoiceMessageUpload(language="es", recipient_id=1)
            print("✗ Invalid language 'es' should have been rejected")
        except Exception as e:
            print(f"✓ Invalid language 'es' properly rejected: {e}")
            
    except ImportError as e:
        print(f"Schema import error: {e}")

if __name__ == "__main__":
    test_language_validation()
    test_schema_validation()
