#!/usr/bin/env python3
"""
Advanced Speech Recognition Tester
Tests different speech recognition methods and settings
"""

import speech_recognition as sr
import os
import sys
from pydub import AudioSegment

def test_google_speech_recognition(wav_file):
    """Test Google Speech Recognition with different settings"""
    print(f"=== Testing Google Speech Recognition ===")
    print(f"File: {wav_file}")
    
    if not os.path.exists(wav_file):
        print(f"❌ File not found: {wav_file}")
        return False
    
    r = sr.Recognizer()
    
    # Test different recognizer settings
    settings = [
        {
            "name": "Default settings",
            "energy_threshold": 4000,
            "dynamic_energy_threshold": True,
            "noise_duration": 1.0
        },
        {
            "name": "Low threshold for quiet speech", 
            "energy_threshold": 300,
            "dynamic_energy_threshold": True,
            "noise_duration": 1.0
        },
        {
            "name": "High threshold for noisy audio",
            "energy_threshold": 8000,
            "dynamic_energy_threshold": False,
            "noise_duration": 0.5
        },
        {
            "name": "Minimal noise adjustment",
            "energy_threshold": 1000,
            "dynamic_energy_threshold": True,
            "noise_duration": 0.1
        }
    ]
    
    languages = ['en-US', 'en-GB', 'en-AU', 'en-IN']
    
    success = False
    
    for setting in settings:
        print(f"\n--- {setting['name']} ---")
        
        try:
            with sr.AudioFile(wav_file) as source:
                # Apply settings
                r.energy_threshold = setting['energy_threshold']
                r.dynamic_energy_threshold = setting['dynamic_energy_threshold']
                
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source, duration=setting['noise_duration'])
                print(f"Energy threshold after adjustment: {r.energy_threshold}")
                
                # Record audio
                audio_data = r.record(source)
                
                # Try different English variants
                for lang in languages:
                    try:
                        print(f"  Trying {lang}...")
                        
                        # Try with show_all=False first
                        result = r.recognize_google(audio_data, language=lang)
                        print(f"  ✅ SUCCESS ({lang}): {result}")
                        success = True
                        break
                        
                    except sr.UnknownValueError:
                        print(f"  ❌ No speech detected ({lang})")
                        continue
                    except sr.RequestError as e:
                        print(f"  ❌ API error ({lang}): {e}")
                        continue
                
                # If successful, try show_all=True for more details
                if success:
                    try:
                        detailed_results = r.recognize_google(audio_data, language='en-US', show_all=True)
                        if detailed_results:
                            print(f"  📊 Detailed results:")
                            for i, alt in enumerate(detailed_results[:3]):
                                conf = alt.get('confidence', 'N/A')
                                print(f"    {i+1}. {alt['transcript']} (confidence: {conf})")
                    except:
                        pass
                
                if success:
                    break
                    
        except Exception as e:
            print(f"❌ Error with {setting['name']}: {e}")
    
    return success

def test_audio_content_analysis(wav_file):
    """Analyze audio content for speech detection"""
    print(f"\n=== Audio Content Analysis ===")
    
    try:
        audio = AudioSegment.from_file(wav_file)
        
        # Check for silence
        silence_thresh = audio.dBFS - 16  # 16dB below average
        non_silent = audio.strip_silence(silence_thresh=silence_thresh, chunk_len=100)
        
        silence_ratio = (len(audio) - len(non_silent)) / len(audio)
        print(f"Silence ratio: {silence_ratio:.2%}")
        
        if silence_ratio > 0.8:
            print("⚠️  Audio is mostly silence")
        elif silence_ratio > 0.5:
            print("⚠️  Audio has significant silence")
        else:
            print("✅ Audio has good speech content")
        
        # Check volume variation (indicates speech patterns)
        chunks = [audio[i:i+1000] for i in range(0, len(audio), 1000)]
        volumes = [chunk.dBFS for chunk in chunks if len(chunk) > 100]
        
        if len(volumes) > 1:
            volume_std = sum((v - sum(volumes)/len(volumes))**2 for v in volumes) ** 0.5 / len(volumes)
            print(f"Volume variation: {volume_std:.1f} dB")
            
            if volume_std < 2:
                print("⚠️  Low volume variation - may not be speech")
            else:
                print("✅ Good volume variation - likely contains speech")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio analysis error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_speech_recognition.py <wav_file>")
        print("\nLooking for recent voice files...")
        
        import tempfile
        import glob
        
        temp_dir = tempfile.gettempdir()
        pattern = os.path.join(temp_dir, "voice_*.wav")
        voice_files = glob.glob(pattern)
        
        if voice_files:
            # Test the most recent file
            latest_file = max(voice_files, key=os.path.getmtime)
            print(f"Testing latest file: {latest_file}")
            wav_file = latest_file
        else:
            print("No voice files found.")
            return
    else:
        wav_file = sys.argv[1]
    
    print(f"Testing speech recognition for: {wav_file}\n")
    
    # Test audio content
    test_audio_content_analysis(wav_file)
    
    # Test Google Speech Recognition
    success = test_google_speech_recognition(wav_file)
    
    if success:
        print(f"\n✅ Speech recognition successful!")
    else:
        print(f"\n❌ All speech recognition attempts failed")
        print("Possible issues:")
        print("• Audio may not contain clear speech")
        print("• Background noise may be too high")
        print("• Speech may be in a different language")
        print("• Audio quality may be insufficient")

if __name__ == "__main__":
    main()
