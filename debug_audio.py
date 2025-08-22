#!/usr/bin/env python3
"""
Audio Quality Checker for Voice Messages
This script helps diagnose issues with voice recordings
"""

import sys
import os
from pydub import AudioSegment
from pathlib import Path

def analyze_audio_file(file_path):
    """Analyze an audio file and provide recommendations"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        audio = AudioSegment.from_file(file_path)
        
        print(f"📁 File: {file_path}")
        print(f"⏱️  Duration: {len(audio)}ms ({len(audio)/1000:.1f} seconds)")
        print(f"🔊 Channels: {audio.channels}")
        print(f"📊 Sample Rate: {audio.frame_rate}Hz")
        print(f"🔉 Volume (dBFS): {audio.dBFS:.1f}")
        print(f"💾 File Size: {os.path.getsize(file_path)} bytes")
        
        # Quality checks
        issues = []
        recommendations = []
        
        # Check duration
        if len(audio) < 500:
            issues.append("Audio too short (< 0.5 seconds)")
            recommendations.append("Record for at least 1-2 seconds")
        elif len(audio) < 1000:
            issues.append("Audio quite short (< 1 second)")
            recommendations.append("Consider recording longer messages")
        
        # Check volume
        if audio.dBFS < -40:
            issues.append("Audio very quiet")
            recommendations.append("Speak louder or closer to microphone")
        elif audio.dBFS < -25:
            issues.append("Audio quiet")
            recommendations.append("Consider speaking a bit louder")
        
        # Check sample rate
        if audio.frame_rate < 16000:
            issues.append("Low sample rate")
            recommendations.append("Use higher quality recording settings")
        
        # Check channels
        if audio.channels > 1:
            recommendations.append("Audio will be converted to mono for processing")
        
        # Summary
        if not issues:
            print("\n✅ Audio quality looks good!")
        else:
            print(f"\n⚠️  Issues found ({len(issues)}):")
            for issue in issues:
                print(f"   • {issue}")
        
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ Error analyzing audio: {e}")
        return False

def check_temp_directory():
    """Check if temp directory is accessible"""
    import tempfile
    temp_dir = tempfile.gettempdir()
    print(f"📂 Temp directory: {temp_dir}")
    
    # Check if we can write to temp
    try:
        test_file = os.path.join(temp_dir, "voice_test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.unlink(test_file)
        print("✅ Temp directory is writable")
        return True
    except Exception as e:
        print(f"❌ Cannot write to temp directory: {e}")
        return False

if __name__ == "__main__":
    print("=== Voice Message Audio Quality Checker ===\n")
    
    # Check temp directory
    check_temp_directory()
    print()
    
    if len(sys.argv) > 1:
        # Analyze specific file
        file_path = sys.argv[1]
        analyze_audio_file(file_path)
    else:
        # Look for recent voice files in temp
        import tempfile
        import glob
        
        temp_dir = tempfile.gettempdir()
        pattern = os.path.join(temp_dir, "voice_*.webm")
        voice_files = glob.glob(pattern)
        
        pattern2 = os.path.join(temp_dir, "voice_*.wav")
        voice_files.extend(glob.glob(pattern2))
        
        if voice_files:
            print(f"Found {len(voice_files)} recent voice files:")
            for i, file in enumerate(sorted(voice_files, key=os.path.getmtime, reverse=True)[:3]):
                print(f"\n--- File {i+1} ---")
                analyze_audio_file(file)
        else:
            print("No recent voice files found in temp directory.")
            print("Usage: python debug_audio.py <audio_file_path>")
