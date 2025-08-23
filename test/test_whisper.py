#!/usr/bin/env python3
"""
Test Whisper Speech Recognition
"""

import tempfile
import os
from pydub import AudioSegment
from pydub.generators import Sine

def test_whisper_installation():
    """Test if Whisper is properly installed"""
    print("=== Testing Whisper Installation ===")
    
    try:
        import whisper
        print("✅ Whisper library imported successfully")
        
        # Test model loading
        print("🔄 Loading Whisper base model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        
        return model
        
    except ImportError as e:
        print(f"❌ Whisper not installed: {e}")
        print("Install with: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return None

def create_test_audio_with_speech():
    """Create a simple test audio that might work better with Whisper"""
    print("\n=== Creating Test Audio ===")
    
    # Create a longer audio with some variation (simulating speech patterns)
    duration = 3000  # 3 seconds
    audio = AudioSegment.silent(duration=duration)
    
    # Add some tones at different frequencies (simulating speech formants)
    frequencies = [200, 400, 800, 1600]  # Rough speech frequency range
    
    for i, freq in enumerate(frequencies):
        start_time = i * 700
        tone = Sine(freq).to_audio_segment(duration=500)
        tone = tone - 20  # Make it a bit quieter
        
        if start_time + 500 <= duration:
            # Overlay the tone
            audio = audio.overlay(tone, position=start_time)
    
    # Normalize and adjust
    audio = audio.normalize()
    
    return audio

def test_whisper_with_audio(model, audio_file):
    """Test Whisper transcription with an audio file"""
    print(f"\n=== Testing Whisper Transcription ===")
    
    if not model:
        print("❌ No Whisper model available")
        return False
    
    try:
        print(f"Transcribing: {audio_file}")
        
        # Test with different languages
        languages = ['english', 'french', 'arabic']
        
        for lang in languages:
            print(f"\n--- Testing {lang} ---")
            try:
                result = model.transcribe(audio_file, language=lang, fp16=False, verbose=False)
                text = result["text"].strip()
                confidence = result.get("confidence", "N/A")
                
                print(f"Result: '{text}'")
                print(f"Confidence: {confidence}")
                
                if text:
                    print(f"✅ Whisper detected some content in {lang}")
                else:
                    print(f"⚠️  No transcription for {lang}")
                    
            except Exception as e:
                print(f"❌ Error with {lang}: {e}")
        
        # Test with auto-detection
        print(f"\n--- Testing Auto-Detection ---")
        try:
            result = model.transcribe(audio_file, fp16=False, verbose=False)
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            
            print(f"Auto-detected language: {detected_lang}")
            print(f"Result: '{text}'")
            
            if text:
                print("✅ Auto-detection worked")
                return True
            else:
                print("⚠️  Auto-detection returned empty")
                
        except Exception as e:
            print(f"❌ Auto-detection error: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

def main():
    print("🎙️ WHISPER SPEECH RECOGNITION TESTER 🎙️")
    print("=" * 50)
    
    # Test installation
    model = test_whisper_installation()
    
    if not model:
        print("\n❌ Cannot proceed without Whisper model")
        return
    
    # Create test audio
    test_audio = create_test_audio_with_speech()
    
    # Save to temp file
    temp_path = tempfile.mktemp(suffix='.wav')
    test_audio.export(temp_path, format='wav')
    print(f"Test audio saved: {temp_path}")
    print(f"Audio info: {len(test_audio)}ms, {test_audio.dBFS:.1f} dBFS")
    
    # Test Whisper
    success = test_whisper_with_audio(model, temp_path)
    
    # Test with real failed audio file if available
    import glob
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, "voice_*.wav")
    voice_files = glob.glob(pattern)
    
    if voice_files:
        print(f"\n=== Testing with Real Audio Files ===")
        latest_file = max(voice_files, key=os.path.getmtime)
        print(f"Testing with: {latest_file}")
        test_whisper_with_audio(model, latest_file)
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass
    
    if success:
        print(f"\n✅ Whisper is working! Ready to replace Google Speech Recognition.")
    else:
        print(f"\n⚠️  Whisper setup needs attention. Check the results above.")

if __name__ == "__main__":
    main()
