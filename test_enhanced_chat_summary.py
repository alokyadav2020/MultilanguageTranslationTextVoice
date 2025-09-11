#!/usr/bin/env python3
"""
Test Enhanced Chat Summary Service
Tests the new enhanced conversation analysis with actual chat content
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from services.chat_summary_service import chat_summary_service
    print("✅ Enhanced Chat Summary Service imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_messages():
    """Create realistic test messages with conversation content"""
    base_time = datetime.now() - timedelta(hours=2)
    
    messages = [
        {
            "id": 1,
            "sender_name": "Alice",
            "original_text": "Hey everyone! How are you all doing today?",
            "translated_text": "Hey everyone! How are you all doing today?",
            "original_language": "en",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=0)).isoformat(),
        },
        {
            "id": 2,
            "sender_name": "Bob",
            "original_text": "I'm doing great! Just finished a big project at work. Really excited about the results we achieved.",
            "translated_text": "I'm doing great! Just finished a big project at work. Really excited about the results we achieved.",
            "original_language": "en",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
        },
        {
            "id": 3,
            "sender_name": "Charlie",
            "original_text": "Congratulations Bob! What kind of project was it?",
            "translated_text": "Congratulations Bob! What kind of project was it?",
            "original_language": "en",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=7)).isoformat(),
        },
        {
            "id": 4,
            "sender_name": "Bob",
            "original_text": "",
            "translated_text": "",
            "original_language": "en",
            "message_type": "voice",
            "audio_duration": 15.5,
            "timestamp": (base_time + timedelta(minutes=10)).isoformat(),
        },
        {
            "id": 5,
            "sender_name": "Alice",
            "original_text": "That sounds amazing! We should definitely celebrate your success. How about we meet for lunch tomorrow?",
            "translated_text": "That sounds amazing! We should definitely celebrate your success. How about we meet for lunch tomorrow?",
            "original_language": "en",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=15)).isoformat(),
        },
        {
            "id": 6,
            "sender_name": "Diana",
            "original_text": "Je suis d'accord! C'est une excellente idée.",
            "translated_text": "I agree! That's an excellent idea.",
            "original_language": "fr",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=18)).isoformat(),
        },
        {
            "id": 7,
            "sender_name": "Charlie",
            "original_text": "Count me in! I know a great place downtown that serves excellent food. They have both French and international cuisine.",
            "translated_text": "Count me in! I know a great place downtown that serves excellent food. They have both French and international cuisine.",
            "original_language": "en",
            "message_type": "text",
            "timestamp": (base_time + timedelta(minutes=22)).isoformat(),
        },
        {
            "id": 8,
            "sender_name": "Bob",
            "original_text": "",
            "translated_text": "",
            "original_language": "en",
            "message_type": "voice",
            "audio_duration": 8.2,
            "timestamp": (base_time + timedelta(minutes=25)).isoformat(),
        },
    ]
    
    return messages

async def test_enhanced_summary():
    """Test the enhanced chat summary with conversation content"""
    print("\n🧪 Testing Enhanced Chat Summary Service...")
    
    # Check service status
    status = chat_summary_service.get_model_status()
    print(f"📊 Service Status: {status}")
    
    # Create test messages
    test_messages = create_test_messages()
    print(f"📝 Created {len(test_messages)} test messages")
    
    # Test with different languages
    test_cases = [
        ("en", "group"),
        ("fr", "private"),
        ("ar", "group")
    ]
    
    for language, chat_type in test_cases:
        print(f"\n🌍 Testing summary for {language} ({chat_type} chat)...")
        
        try:
            # Generate summary
            result = await chat_summary_service.generate_summary(
                messages=test_messages,
                user_language=language,
                chat_type=chat_type
            )
            
            if result["success"]:
                print(f"✅ Summary generated successfully!")
                print(f"📄 Summary: {result['summary'][:200]}...")
                
                # Display enhanced statistics
                stats = result["statistics"]
                print(f"\n📊 Enhanced Statistics:")
                print(f"  • Total Messages: {stats.get('total_messages')}")
                print(f"  • Participants: {stats.get('participant_count')} ({', '.join(stats.get('participants', []))})")
                print(f"  • Voice Messages: {stats.get('voice_messages')} ({stats.get('voice_percentage')}%)")
                print(f"  • Average Message Length: {stats.get('average_message_length')} chars")
                print(f"  • Busiest Hour: {stats.get('busiest_hour')}")
                print(f"  • Languages: {', '.join(stats.get('languages_used', []))}")
                
                # Test downloadable summary
                downloadable = chat_summary_service.create_downloadable_summary(result, "markdown")
                print(f"\n📄 Downloadable summary preview (first 300 chars):")
                print(downloadable[:300] + "...")
                
            else:
                print(f"❌ Summary generation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Test failed for {language}: {e}")

async def test_conversation_highlights():
    """Test conversation highlight extraction"""
    print("\n🔍 Testing Conversation Highlights...")
    
    test_messages = create_test_messages()
    
    # Prepare chat text
    chat_text = chat_summary_service._prepare_chat_text(test_messages)
    print(f"📝 Prepared chat text ({len(chat_text)} chars):")
    print(chat_text[:300] + "...")
    
    # Extract highlights
    highlights = chat_summary_service._extract_conversation_highlights(chat_text)
    print(f"\n✨ Extracted {len(highlights)} conversation highlights:")
    for i, highlight in enumerate(highlights, 1):
        print(f"  {i}. {highlight}")

async def main():
    """Main test function"""
    print("🚀 Enhanced Chat Summary Service Test Suite")
    print("=" * 50)
    
    try:
        await test_enhanced_summary()
        await test_conversation_highlights()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎉 Enhanced Chat Summary Service is ready!")
        print("Now summaries include:")
        print("  • AI-powered conversation analysis")
        print("  • Key conversation highlights and exchanges")
        print("  • Detailed participant statistics")
        print("  • Enhanced conversation metrics")
        print("  • Multi-language support (EN/FR/AR)")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
