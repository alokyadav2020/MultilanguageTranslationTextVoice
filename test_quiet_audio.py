#!/usr/bin/env python3
"""
Test improved speech recognition with quiet audio
"""

import sys
import os
import tempfile
from pydub import AudioSegment
import speech_recognition as sr

def create_quiet_test_audio():
    """Create a very quiet test audio file"""
    # Create a short audio with some content (not silence)
    audio = AudioSegment.silent(duration=2000)
    
    # Add a simple tone to simulate speech
    from pydub.generators import Sine
    tone = Sine(440).to_audio_segment(duration=500)  # A4 note for 0.5 seconds
    
    # Make it very quiet like the problem audio
    quiet_tone = tone - 60  # Make it 60dB quieter
    
    # Insert the quiet tone into the middle of the audio
    audio = audio[:750] + quiet_tone + audio[1250:]
    
    return audio

def test_quiet_audio_processing():
    """Test the audio processing pipeline with quiet audio"""
    print("=== Testing Quiet Audio Processing ===\n")
    
    # Create test audio
    test_audio = create_quiet_test_audio()
    print(f"Original test audio: {len(test_audio)}ms, {test_audio.dBFS:.1f} dBFS")
    
    # Save to temp file
    temp_path = tempfile.mktemp(suffix='.wav')
    test_audio.export(temp_path, format='wav')
    print(f"Saved test audio to: {temp_path}")
    
    try:
        # Simulate the processing pipeline
        print("\n--- Processing Pipeline ---")
        
        # Load audio
        audio = AudioSegment.from_file(temp_path)
        print(f"Loaded: {len(audio)}ms, {audio.dBFS:.1f} dBFS")
        
        # Normalize
        audio = audio.normalize()
        print(f"After normalize: {audio.dBFS:.1f} dBFS")
        
        # Boost if quiet
        if audio.dBFS < -35:
            boost_db = -20 - audio.dBFS
            audio = audio + boost_db
            print(f"Boosted by {boost_db:.1f}dB, new level: {audio.dBFS:.1f} dBFS")
        
        # Convert settings
        audio = audio.set_channels(1).set_frame_rate(16000)
        print(f"Final: 1 channel, 16000Hz, {audio.dBFS:.1f} dBFS")
        
        # Save processed version
        processed_path = temp_path.replace('.wav', '_processed.wav')
        audio.export(processed_path, format='wav')
        print(f"Processed audio saved to: {processed_path}")
        
        # Test speech recognition
        print("\n--- Speech Recognition Test ---")
        r = sr.Recognizer()
        
        with sr.AudioFile(processed_path) as source:
            r.adjust_for_ambient_noise(source, duration=1.0)
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            audio_data = r.record(source)
        
        try:
            text = r.recognize_google(audio_data, language='en-US')
            print(f"Recognition result: {text}")
        except sr.UnknownValueError:
            print("Recognition: No speech detected (expected for tone test)")
        except sr.RequestError as e:
            print(f"Recognition error: {e}")
        
        print("\n✅ Audio processing pipeline test completed")
        
        # Clean up
        os.unlink(temp_path)
        if os.path.exists(processed_path):
            os.unlink(processed_path)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quiet_audio_processing()
