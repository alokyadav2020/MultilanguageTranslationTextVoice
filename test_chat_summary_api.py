#!/usr/bin/env python3
"""
Test chat summary API after fixing the Message.translated_text issue
"""

import requests
import json

def test_chat_summary_api():
    """Test if the chat summary API is working"""
    
    base_url = "http://localhost:8000"
    
    # Test endpoints
    endpoints = [
        "/api/chat-summary/status",
        # Add more endpoints here if needed
    ]
    
    print("🧪 Testing Chat Summary API endpoints...")
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"\n📡 Testing: {url}")
            
            response = requests.get(url, timeout=10)
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                    print("   ✅ Success!")
                except Exception as e:
                    print(f"   Response: {response.text[:200]}...")
                    print(f"   ⚠️  Non-JSON response: {e}")
            else:
                print(f"   Error: {response.text[:200]}...")
                print("   ❌ Failed")
                
        except requests.exceptions.RequestException as e:
            print(f"   Connection Error: {e}")
            print("   ❌ Cannot connect to server")
        except Exception as e:
            print(f"   Unexpected Error: {e}")
            print("   ❌ Test failed")

if __name__ == "__main__":
    test_chat_summary_api()
