#!/usr/bin/env python3
"""Test script to call the force cleanup endpoint"""

import requests
import json

def test_cleanup():
    # Your token from the browser (user 2)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzYWNoaW5Ac2FjaGluLmNvbSIsImV4cCI6MTc1NjAyNjM5Nn0.qrJC1U_gRZCgVB8CyeraRNYR0CNR4xVoCz4l99GR7Lw"
    
    url = "http://localhost:8000/api/voice-call/force-cleanup"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        print("🧹 Testing force cleanup endpoint...")
        response = requests.post(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cleanup successful: {data.get('message', 'No message')}")
        else:
            print(f"❌ Cleanup failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error calling cleanup endpoint: {e}")

if __name__ == "__main__":
    test_cleanup()
