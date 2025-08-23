#!/usr/bin/env python3
"""
Test speech recognition with a simple audio test
"""

import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment

def test_speech_recognition():
    """Test if speech recognition is working"""
    print("Testing speech recognition setup...")
    
    # Test if microphone access works
    r = sr.Recognizer()
    
    try:
        # Create a simple test audio (silence)
        print("Creating test audio...")
        test_audio = AudioSegment.silent(duration=2000)  # 2 seconds of silence
        test_audio = test_audio.set_frame_rate(16000).set_channels(1)
        
        # Save to temp file
        temp_path = tempfile.mktemp(suffix='.wav')
        test_audio.export(temp_path, format='wav')
        print(f"Test audio saved to: {temp_path}")
        
        # Try to process with speech recognition
        with sr.AudioFile(temp_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.record(source)
            print("Audio data extracted successfully")
        
        # Test recognition (will fail on silence, but should not crash)
        try:
            text = r.recognize_google(audio_data, language='en-US')
            print(f"Recognition result: {text}")
        except sr.UnknownValueError:
            print("✓ Recognition properly detected silence (no speech)")
        except sr.RequestError as e:
            print(f"✗ Network/API error: {e}")
            return False
        
        # Clean up
        os.unlink(temp_path)
        print("✓ Speech recognition setup is working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Speech recognition test failed: {e}")
        return False

def test_audio_formats():
    """Test audio format conversion"""
    print("\nTesting audio format conversion...")
    
    try:
        # Test creating and converting audio
        audio = AudioSegment.silent(duration=1000)
        print(f"Original: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
        
        # Test normalization
        normalized = audio.normalize()
        print("✓ Audio normalization works")
        
        # Test channel conversion
        mono = audio.set_channels(1)
        print("✓ Channel conversion works")
        
        # Test sample rate conversion
        resampled = audio.set_frame_rate(16000)
        print("✓ Sample rate conversion works")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio format test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Voice Service Diagnostics ===")
    
    setup_ok = test_speech_recognition()
    format_ok = test_audio_formats()
    
    if setup_ok and format_ok:
        print("\n✓ All tests passed! Voice service should work correctly.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
