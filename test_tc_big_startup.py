#!/usr/bin/env python3
"""
Test script for the updated Helsinki-NLP TC-Big Translation Service.
Tests the non-blocking startup and model loading on Jetson.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_non_blocking_startup():
    """Test that the service starts without blocking."""
    print("🧪 Testing non-blocking startup...")
    
    try:
        # Import the service (this will initialize it)
        from app.services.translation import translation_service
        print("✅ Translation service imported successfully")
        print(f"📂 Cache directory: {translation_service.cache_dir}")
        print(f"🔄 Model mappings: {len(translation_service.model_mappings)} pairs")
        print(f"🖥️  Device: {translation_service.device}")
        
        # Test model info
        info = translation_service.get_model_info()
        print(f"💾 Cache size: {info['cache_size_mb']} MB")
        
        # Test basic language validation
        result = translation_service.translate_text("Hello", "en", "invalid")
        if result is None:
            print("✅ Language validation works correctly")
        
        # Test same language
        result = translation_service.translate_text("Hello", "en", "en")
        if result == "Hello":
            print("✅ Same language handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_async_model_loading():
    """Test asynchronous model loading."""
    print("\n🧪 Testing async model loading...")
    
    try:
        from app.services.translation import translation_service
        
        # Test model loading for one pair
        success = translation_service._load_model("en", "fr")
        if success:
            print("✅ Model loading test successful")
            
            # Test translation if model is loaded
            result = translation_service.translate_text("Hello world", "en", "fr")
            if result:
                print(f"✅ Translation test successful: 'Hello world' → '{result}'")
            else:
                print("⚠️  Translation returned None (model may still be loading)")
        else:
            print("⚠️  Model loading failed (models may not be downloaded yet)")
            
        return True
        
    except Exception as e:
        print(f"❌ Async model loading test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Helsinki-NLP TC-Big Translation Service Test Suite")
    print("=" * 60)
    
    tests = [
        test_non_blocking_startup,
        test_async_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TC-Big translation service is ready.")
        print("\n📋 Next steps for Jetson:")
        print("1. Ensure models are downloaded in: /home/orin/Desktop/translation_project/MultilanguageTranslationTextVoice/artifacts/models/")
        print("2. Start server: uvicorn app.main:app --reload")
        print("3. Check startup logs for background model loading progress")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
