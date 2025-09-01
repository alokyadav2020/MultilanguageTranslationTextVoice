#!/usr/bin/env python3
"""
SeamlessM4T Model Download Script
Downloads and tests SeamlessM4T models for real-time voice translation
"""

import sys
import torch
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import torchaudio
        logger.info(f"✅ PyTorch {torch.__version__} found")
        logger.info(f"✅ TorchAudio {torchaudio.__version__} found")
        
        try:
            # Just try to import to check availability
            __import__('seamless_communication')
            logger.info("✅ SeamlessM4T package found")
            return True
        except ImportError:
            logger.error("❌ SeamlessM4T not installed")
            logger.error("Install with: pip install seamless_communication")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        return False

def check_device():
    """Check available compute device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🎮 GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("💻 Using CPU (slower but functional)")
    
    return device

def check_cache_directory():
    """Check PyTorch cache directory"""
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    logger.info(f"📁 Cache directory: {cache_dir}")
    
    if cache_dir.exists():
        existing_files = list(cache_dir.glob("*seamless*"))
        if existing_files:
            logger.info(f"✅ Found {len(existing_files)} existing SeamlessM4T files:")
            for file in existing_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
            return True
        else:
            logger.info("📦 No existing SeamlessM4T models found")
    else:
        logger.info("📁 Cache directory will be created")
    
    return False

def download_models(device):
    """Download SeamlessM4T models"""
    logger.info("📥 Starting SeamlessM4T model download...")
    logger.info("⏳ This will download models automatically on first use...")
    
    try:
        from seamless_communication.models.inference import Translator
        
        start_time = time.time()
        
        # Use the official API - this will trigger model download if needed
        translator = Translator("seamlessM4T_large")
        
        download_time = time.time() - start_time
        logger.info(f"✅ Models loaded successfully in {download_time:.1f} seconds")
        
        return translator
        
    except Exception as e:
        logger.error(f"❌ Failed to download models: {e}")
        return None

def test_translation(translator, device):
    """Test translation functionality"""
    logger.info("🧪 Testing translation functionality...")
    
    try:
        # Create dummy audio (2 seconds of silence at 16kHz)
        dummy_audio = torch.zeros(32000).to(device)
        logger.info(f"📊 Test audio shape: {dummy_audio.shape}")
        
        # Test English to French translation using the official API
        logger.info("🔄 Testing English → French translation...")
        
        with torch.no_grad():
            text_out, audio_out = translator.predict(
                dummy_audio,
                src_lang="eng",   # English
                tgt_lang="fra"    # French
            )
        
        if audio_out is not None:
            logger.info(f"✅ Translation successful! Output shape: {audio_out.shape}")
            
            # Test Arabic translation
            logger.info("🔄 Testing English → Arabic translation...")
            
            with torch.no_grad():
                text_out_ar, audio_out_ar = translator.predict(
                    dummy_audio,
                    src_lang="eng",   # English
                    tgt_lang="arb"    # Arabic
                )
            
            if audio_out_ar is not None:
                logger.info("✅ Arabic translation test successful!")
                return True
            else:
                logger.error("❌ Arabic translation test failed")
                return False
        else:
            logger.error("❌ Translation test failed - no audio output")
            return False
            
    except Exception as e:
        logger.error(f"❌ Translation test failed: {e}")
        return False

def show_cache_info():
    """Show final cache information"""
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    
    if cache_dir.exists():
        all_files = list(cache_dir.glob("*"))
        seamless_files = [f for f in all_files if "seamless" in f.name.lower()]
        
        total_size = sum(f.stat().st_size for f in all_files) / (1024**3)
        seamless_size = sum(f.stat().st_size for f in seamless_files) / (1024**3)
        
        logger.info("📊 Cache Statistics:")
        logger.info(f"  Total files: {len(all_files)}")
        logger.info(f"  SeamlessM4T files: {len(seamless_files)}")
        logger.info(f"  Total cache size: {total_size:.2f} GB")
        logger.info(f"  SeamlessM4T size: {seamless_size:.2f} GB")

def main():
    """Main download and test function"""
    logger.info("🚀 SeamlessM4T Model Download & Test")
    logger.info("🌍 Supporting Arabic, English, and French")
    print()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check device
    device = check_device()
    
    # Check existing cache
    models_exist = check_cache_directory()
    
    print()
    
    if models_exist:
        logger.info("✅ Models already downloaded!")
        answer = input("Do you want to test existing models? (y/n): ").lower().strip()
        if answer != 'y':
            logger.info("👋 Exiting without testing")
            return
    else:
        logger.info("📥 Models need to be downloaded")
        answer = input("Do you want to proceed with download? (~4-6 GB) (y/n): ").lower().strip()
        if answer != 'y':
            logger.info("👋 Download cancelled")
            return
    
    print()
    
    # Download/load models
    translator = download_models(device)
    if not translator:
        logger.error("❌ Failed to load models")
        sys.exit(1)
    
    print()
    
    # Test translation
    test_success = test_translation(translator, device)
    
    print()
    
    # Show cache info
    show_cache_info()
    
    print()
    
    if test_success:
        logger.info("🎉 SeamlessM4T is ready for real-time voice translation!")
        logger.info("📋 Supported languages:")
        logger.info("  🇸🇦 Arabic (ar)")
        logger.info("  🇺🇸 English (en)")
        logger.info("  🇫🇷 French (fr)")
        print()
        logger.info("🚀 You can now start your application:")
        logger.info("  python -m uvicorn app.main:app --reload")
        logger.info("  Visit: http://localhost:8000/enhanced-voice-call")
    else:
        logger.error("❌ Translation test failed - please check the logs above")
        sys.exit(1)

if __name__ == "__main__":
    main()
