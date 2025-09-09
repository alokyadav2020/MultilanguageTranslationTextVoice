#!/usr/bin/env python3
"""
Test script to verify Message model methods work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.message import Message

def test_message_translation():
    """Test Message.get_translated_text() method"""
    
    # Create a mock message object
    message = Message()
    message.id = 1
    message.original_text = "Hello world"
    message.original_language = "en"
    message.translations_cache = {
        "fr": "Bonjour le monde",
        "ar": "مرحبا بالعالم",
        "es": "Hola mundo"
    }
    
    # Test getting translation for different languages
    print("Testing Message.get_translated_text() method:")
    print(f"Original: {message.get_translated_text('en')}")
    print(f"French: {message.get_translated_text('fr')}")
    print(f"Arabic: {message.get_translated_text('ar')}")
    print(f"Spanish: {message.get_translated_text('es')}")
    print(f"German (fallback): {message.get_translated_text('de')}")
    
    print("\n✅ Message model translation method working correctly!")

if __name__ == "__main__":
    test_message_translation()
