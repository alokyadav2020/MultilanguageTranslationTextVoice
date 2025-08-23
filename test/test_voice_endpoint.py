#!/usr/bin/env python3
"""
Test the voice upload endpoint to see why it's returning 422 errors
"""
import requests
import json

def test_voice_upload():
    """Test voice upload with different scenarios"""
    
    base_url = "http://127.0.0.1:8000"
    
    # First, let's test if we can reach the endpoint
    print("🔍 Testing voice upload endpoint...")
    
    # Test 1: Check if endpoint exists
    try:
        # Create dummy audio data
        audio_data = b"fake_audio_data_for_testing"
        
        # Test with recipient_id (old format)
        print("\n📝 Test 1: Upload with recipient_id")
        
        files = {
            'audio': ('test.wav', audio_data, 'audio/wav')
        }
        data = {
            'language': 'en',
            'recipient_id': 2
        }
        
        # We need auth token, but let's see what error we get
        response = requests.post(f"{base_url}/api/voice/upload-message", files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 2: Check with chatroom_id
        print("\n📝 Test 2: Upload with chatroom_id")
        
        files = {
            'audio': ('test.wav', audio_data, 'audio/wav')
        }
        data = {
            'language': 'en',
            'chatroom_id': 1
        }
        
        response = requests.post(f"{base_url}/api/voice/upload-message", files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 3: Check what validation errors we get
        print("\n📝 Test 3: Invalid language")
        
        files = {
            'audio': ('test.wav', audio_data, 'audio/wav')
        }
        data = {
            'language': 'es',  # Invalid language
            'recipient_id': 2
        }
        
        response = requests.post(f"{base_url}/api/voice/upload-message", files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_voice_upload()
