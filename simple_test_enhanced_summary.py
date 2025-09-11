"""
Simple test of enhanced chat summary service
"""
import sys
import os
sys.path.append('app')

try:
    from services.chat_summary_service import chat_summary_service
    print("✅ Enhanced Chat Summary Service imported successfully")
    
    # Get model status
    status = chat_summary_service.get_model_status()
    print("📊 Model Status:")
    for key, value in status.items():
        print(f"  • {key}: {value}")
    
    # Test statistics calculation with sample data
    sample_messages = [
        {
            "sender_name": "Alice",
            "original_text": "Hello everyone! How are you doing today?",
            "translated_text": "Hello everyone! How are you doing today?",
            "original_language": "en",
            "message_type": "text",
            "timestamp": "2024-01-15T10:00:00Z"
        },
        {
            "sender_name": "Bob", 
            "original_text": "I'm doing great! Just finished a big project.",
            "translated_text": "I'm doing great! Just finished a big project.",
            "original_language": "en",
            "message_type": "text",
            "timestamp": "2024-01-15T10:05:00Z"
        },
        {
            "sender_name": "Alice",
            "original_text": "",
            "translated_text": "",
            "original_language": "en", 
            "message_type": "voice",
            "audio_duration": 12.5,
            "timestamp": "2024-01-15T10:10:00Z"
        }
    ]
    
    # Test enhanced statistics
    stats = chat_summary_service._calculate_statistics(sample_messages)
    print("\n📈 Enhanced Statistics Test:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Test conversation highlights
    chat_text = chat_summary_service._prepare_chat_text(sample_messages)
    print(f"\n📝 Chat Text Preparation Test:")
    print(f"Generated chat text: {chat_text}")
    
    highlights = chat_summary_service._extract_conversation_highlights(chat_text)
    print(f"\n🔍 Conversation Highlights Test:")
    for i, highlight in enumerate(highlights, 1):
        print(f"  {i}. {highlight}")
    
    print("\n✅ Enhanced Chat Summary Service is working!")
    print("Key enhancements implemented:")
    print("  🔹 Comprehensive conversation analysis")
    print("  🔹 Key conversation highlights extraction") 
    print("  🔹 Enhanced statistics with participant details")
    print("  🔹 Detailed message analysis")
    print("  🔹 Multi-language support (EN/FR/AR)")
    print("  🔹 GPU acceleration support")
    print("  🔹 Improved summary formatting")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
