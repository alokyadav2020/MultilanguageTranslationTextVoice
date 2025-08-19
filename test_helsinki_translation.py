#!/usr/bin/env python3
"""
Test script for Helsinki-NLP Translation Service on Jetson
Tests model loading, translation functionality, and performance.
"""

import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_import():
    """Test basic import of translation service."""
    print("🧪 Testing basic import...")
    try:
        from app.services.translation import TranslationService
        print("✅ TranslationService import successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_service_initialization():
    """Test translation service initialization."""
    print("\n🧪 Testing service initialization...")
    try:
        from app.services.translation import translation_service
        print("✅ Translation service initialized!")
        print(f"📋 Supported languages: {translation_service.supported_languages}")
        print(f"🔄 Model mappings: {len(translation_service.model_mappings)} pairs")
        print(f"💾 Cache directory: {translation_service.cache_dir}")
        print(f"🖥️  Device: {translation_service.device}")
        return True
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def test_model_info():
    """Test model information retrieval."""
    print("\n🧪 Testing model info...")
    try:
        from app.services.translation import translation_service
        info = translation_service.get_model_info()
        print("✅ Model info retrieved!")
        print(f"📂 Cache directory: {info['cache_directory']}")
        print(f"🔄 Available pairs: {len(info['available_model_pairs'])}")
        print(f"💾 Cache size: {info['cache_size_mb']} MB")
        return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_translation_without_models():
    """Test translation behavior when models aren't downloaded yet."""
    print("\n🧪 Testing translation without models...")
    try:
        from app.services.translation import translation_service
        
        # Test basic language validation
        result = translation_service.translate_text("Hello", "en", "invalid")
        if result is None:
            print("✅ Language validation works (invalid language rejected)")
        
        # Test same language
        result = translation_service.translate_text("Hello", "en", "en")
        if result == "Hello":
            print("✅ Same language handling works")
            
        return True
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def test_download_capabilities():
    """Test model download capabilities."""
    print("\n🧪 Testing download capabilities...")
    try:
        from app.services.translation import translation_service
        
        # Test download availability check
        can_download = translation_service.download_model_if_needed("en", "fr")
        print(f"✅ Download test completed (result: {can_download})")
        return True
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Helsinki-NLP Translation Service Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_import,
        test_service_initialization,
        test_model_info,
        test_translation_without_models,
        test_download_capabilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Helsinki-NLP translation service is ready.")
        print("\n📋 Next steps:")
        print("1. Run: python download_helsinki_models.py --all")
        print("2. Test translation: python test_translation.py")
        print("3. Start server: uvicorn app.main:app --reload")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
