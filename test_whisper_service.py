#!/usr/bin/env python3
"""
Test script for Whisper Translation Service
Tests the real-time voice translation pipeline
"""

import asyncio
import base64
import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_audio():
    """Create a simple test audio file"""
    import numpy as np
    import soundfile as sf
    
    # Generate 2 seconds of 440Hz sine wave (A note)
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Save as temporary WAV file
    temp_path = "test_audio.wav"
    sf.write(temp_path, audio, sample_rate)
    
    # Read and encode as base64
    with open(temp_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Clean up
    Path(temp_path).unlink(missing_ok=True)
    
    return base64.b64encode(audio_bytes).decode()

async def test_whisper_service():
    """Test the Whisper translation service"""
    print("🧪 Testing Whisper Translation Service")
    print("=" * 50)
    
    try:
        # Import the service
        from app.services.whisper_translation_service import whisper_translation_service
        
        # Check if service is available
        if not whisper_translation_service.is_available:
            print("❌ Whisper Translation Service not available")
            print("💡 Please run: python install_whisper_service.py")
            return False
        
        print("✅ Whisper Translation Service is available")
        
        # Get service status
        status = whisper_translation_service.get_service_status()
        print(f"📊 Service Status: {json.dumps(status, indent=2)}")
        
        # Create test audio
        print("\n🔄 Creating test audio...")
        test_audio_base64 = create_test_audio()
        print(f"📁 Test audio size: {len(test_audio_base64)} characters")
        
        # Test translation
        print("\n🌍 Testing translation pipeline...")
        print("🔄 English → French")
        
        start_time = time.time()
        
        result = await whisper_translation_service.process_voice_chunk_realtime(
            call_id="test_call",
            user_id=1,
            audio_data=test_audio_base64,
            source_language="en",
            target_language="fr"
        )
        
        processing_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Result: {json.dumps(result, indent=2)}")
        
        if result.get("success"):
            print("✅ Translation test successful!")
            
            # Test multiple chunks to verify buffering
            print("\n🔄 Testing chunk buffering...")
            for i in range(3):
                chunk_result = await whisper_translation_service.process_voice_chunk_realtime(
                    call_id="test_call_2",
                    user_id=1,
                    audio_data=test_audio_base64,
                    source_language="en",
                    target_language="ar"
                )
                
                print(f"📦 Chunk {i+1}: {chunk_result.get('status', 'unknown')}")
                
                if chunk_result.get("success"):
                    print(f"🎯 Arabic translation: {chunk_result.get('translated_text', 'N/A')}")
                    break
            
            # Test cleanup
            print("\n🧹 Testing cleanup...")
            whisper_translation_service.cleanup_call_buffers("test_call")
            whisper_translation_service.cleanup_call_buffers("test_call_2")
            print("✅ Cleanup completed")
            
            return True
        else:
            print(f"❌ Translation test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure the Whisper service is properly installed")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

async def test_concurrent_processing():
    """Test concurrent processing capabilities"""
    print("\n🚀 Testing Concurrent Processing")
    print("=" * 50)
    
    try:
        from app.services.whisper_translation_service import whisper_translation_service
        
        if not whisper_translation_service.is_available:
            print("❌ Service not available for concurrent testing")
            return False
        
        # Create test audio
        test_audio_base64 = create_test_audio()
        
        # Create multiple concurrent translation tasks
        tasks = []
        start_time = time.time()
        
        # Test different language pairs simultaneously
        test_cases = [
            ("en", "fr", "call_1"),
            ("en", "ar", "call_2"),
            ("fr", "en", "call_3"),
            ("ar", "en", "call_4")
        ]
        
        for src_lang, tgt_lang, call_id in test_cases:
            task = whisper_translation_service.process_voice_chunk_realtime(
                call_id=call_id,
                user_id=1,
                audio_data=test_audio_base64,
                source_language=src_lang,
                target_language=tgt_lang
            )
            tasks.append((task, src_lang, tgt_lang))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*[task for task, _, _ in tasks], return_exceptions=True)
        
        total_time = time.time() - start_time
        
        print(f"⏱️  Total concurrent processing time: {total_time:.2f} seconds")
        print(f"📊 Average time per translation: {total_time/len(tasks):.2f} seconds")
        
        # Analyze results
        successful = 0
        for i, (result, src_lang, tgt_lang) in enumerate(zip(results, [tc[:2] for tc in test_cases])):
            if isinstance(result, Exception):
                print(f"❌ Task {i+1} ({src_lang}→{tgt_lang}): Exception - {result}")
            elif result.get("success"):
                print(f"✅ Task {i+1} ({src_lang}→{tgt_lang}): Success")
                successful += 1
            else:
                print(f"⚠️  Task {i+1} ({src_lang}→{tgt_lang}): {result.get('status', 'Failed')}")
        
        print(f"\n📈 Success rate: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")
        
        # Cleanup
        for _, _, call_id in test_cases:
            whisper_translation_service.cleanup_call_buffers(call_id)
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Concurrent test error: {e}")
        return False

def test_dependencies():
    """Test if all dependencies are available"""
    print("🔍 Testing Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("whisper", "OpenAI Whisper"),
        ("librosa", "Audio processing"),
        ("soundfile", "Audio file I/O"),
        ("numpy", "Numerical computing"),
        ("googletrans", "Google Translate"),
        ("gtts", "Google Text-to-Speech")
    ]
    
    missing = []
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {description} ({module})")
        except ImportError:
            print(f"❌ {description} ({module}) - Missing")
            missing.append(module)
    
    if missing:
        print(f"\n💡 Missing dependencies: {', '.join(missing)}")
        print("🔧 Run: python install_whisper_service.py")
        return False
    
    print("\n✅ All dependencies are available!")
    return True

async def main():
    """Main test function"""
    print("🎤 Whisper Translation Service Test Suite")
    print("🌍 Real-time Voice Translation Testing")
    print("=" * 60)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n💥 Dependency test failed!")
        return
    
    # Test basic service functionality
    basic_success = await test_whisper_service()
    
    if basic_success:
        print("\n✅ Basic functionality test passed!")
        
        # Test concurrent processing
        concurrent_success = await test_concurrent_processing()
        
        if concurrent_success:
            print("\n🎉 All tests passed! Whisper Translation Service is ready!")
            print("\n📋 Next steps:")
            print("1. 🚀 Start your FastAPI server:")
            print("   python -m uvicorn app.main:app --reload")
            print("2. 🌐 Test voice calls:")
            print("   http://localhost:8000/enhanced-voice-call")
            print("3. 🗣️ Try real-time translation between Arabic, English, and French!")
        else:
            print("\n⚠️  Basic tests passed but concurrent processing had issues")
    else:
        print("\n💥 Basic functionality test failed!")
        print("🔧 Please check the errors above and ensure all dependencies are installed")

if __name__ == "__main__":
    asyncio.run(main())
