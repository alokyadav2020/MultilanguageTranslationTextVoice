#!/usr/bin/env python3
"""
Installation script for Whisper + Google Translate Translation Service
Installs all required dependencies for real-time voice translation
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command with proper error handling"""
    logger.info(f"🔄 {description}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - Success")
            return True
        else:
            logger.error(f"❌ {description} - Failed")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} - Timeout")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def install_whisper_dependencies():
    """Install Whisper and related dependencies"""
    logger.info("📦 Installing Whisper Translation Dependencies")
    
    # Core dependencies for Whisper translation service
    packages = [
        # Whisper and audio processing
        "openai-whisper",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.21.0",
        
        # Translation services
        "googletrans==4.0.0rc1",
        "gtts>=2.3.0",
        
        # Async and concurrent processing
        "asyncio",
        
        # Additional audio utilities
        "scipy>=1.9.0",
        "ffmpeg-python",
    ]
    
    # Install each package
    failed_packages = []
    
    for package in packages:
        success = run_command(
            f"pip install {package}",
            f"Installing {package}"
        )
        
        if not success:
            failed_packages.append(package)
            logger.warning(f"⚠️ Failed to install {package}, will try alternatives")
    
    # Try alternative installations for failed packages
    if failed_packages:
        logger.info("🔄 Trying alternative installations...")
        
        for package in failed_packages:
            if "googletrans" in package:
                # Try stable version
                run_command(
                    "pip install googletrans==3.1.0a0",
                    "Installing alternative googletrans version"
                )
            elif "whisper" in package:
                # Try direct GitHub installation
                run_command(
                    "pip install git+https://github.com/openai/whisper.git",
                    "Installing Whisper from GitHub"
                )
    
    return len(failed_packages) == 0

def install_system_dependencies():
    """Install system-level dependencies if needed"""
    logger.info("🔧 Checking system dependencies...")
    
    # Check if ffmpeg is available
    try:
        result = subprocess.run("ffmpeg -version", shell=True, capture_output=True)
        if result.returncode == 0:
            logger.info("✅ FFmpeg is available")
        else:
            logger.warning("⚠️ FFmpeg not found. Please install FFmpeg for better audio support:")
            logger.warning("   Windows: Download from https://ffmpeg.org/download.html")
            logger.warning("   Linux: sudo apt install ffmpeg")
            logger.warning("   macOS: brew install ffmpeg")
    except:
        logger.warning("⚠️ Could not check FFmpeg availability")
    
    return True

def test_installation():
    """Test the installation"""
    logger.info("🧪 Testing Whisper Translation Service installation...")
    
    try:
        # Test Whisper import
        import whisper
        logger.info("✅ Whisper import successful")
        
        # Test audio processing libraries
        import librosa
        import soundfile as sf
        import numpy as np
        logger.info("✅ Audio processing libraries available")
        
        # Test translation libraries
        from googletrans import Translator
        from gtts import gTTS
        logger.info("✅ Translation libraries available")
        
        # Test basic functionality
        logger.info("🔄 Testing basic functionality...")
        
        # Test Whisper model loading (small test)
        try:
            model = whisper.load_model("base")
            logger.info("✅ Whisper model loading successful")
        except Exception as e:
            logger.warning(f"⚠️ Whisper model loading failed: {e}")
            logger.warning("This might be due to network issues or system resources")
        
        # Test Google Translate
        try:
            translator = Translator()
            result = translator.translate("hello", dest='fr')
            if result and result.text:
                logger.info("✅ Google Translate connection successful")
            else:
                logger.warning("⚠️ Google Translate test failed")
        except Exception as e:
            logger.warning(f"⚠️ Google Translate test failed: {e}")
        
        # Test gTTS
        try:
            tts = gTTS(text="test", lang='en')
            logger.info("✅ gTTS initialization successful")
        except Exception as e:
            logger.warning(f"⚠️ gTTS test failed: {e}")
        
        logger.info("🎉 Installation test completed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Installation test failed: {e}")
        return False

def create_requirements_file():
    """Create requirements file for the Whisper service"""
    requirements_content = """# Whisper Translation Service Requirements
# Core Whisper and audio processing
openai-whisper>=20231117
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.21.0
scipy>=1.9.0

# Translation services
googletrans==4.0.0rc1
gtts>=2.3.0

# Audio utilities (optional but recommended)
ffmpeg-python

# Additional utilities
requests>=2.28.0
"""
    
    try:
        with open("whisper_requirements.txt", "w") as f:
            f.write(requirements_content)
        logger.info("✅ Created whisper_requirements.txt")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create requirements file: {e}")
        return False

def main():
    """Main installation function"""
    print("🎤 Whisper + Google Translate Installation")
    print("🌍 Real-time Voice Translation Service Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create requirements file
    create_requirements_file()
    
    # Install system dependencies
    if not install_system_dependencies():
        logger.warning("⚠️ System dependency check failed, continuing anyway...")
    
    # Install Python packages
    if not install_whisper_dependencies():
        logger.error("❌ Failed to install some dependencies")
        logger.info("💡 You can try manual installation using:")
        logger.info("   pip install -r whisper_requirements.txt")
        
        answer = input("\nDo you want to continue with testing? (y/n): ").lower().strip()
        if answer != 'y':
            sys.exit(1)
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("\n📋 Next steps:")
        print("1. 🚀 Start your FastAPI server:")
        print("   python -m uvicorn app.main:app --reload")
        print("\n2. 🌐 Open voice translation interface:")
        print("   http://localhost:8000/enhanced-voice-call")
        print("\n3. 🗣️ Test voice translation between:")
        print("   • Arabic (ar)")
        print("   • English (en)")
        print("   • French (fr)")
        
        print("\n📊 Service Features:")
        print("✅ Real-time audio chunk processing")
        print("✅ Concurrent/async translation pipeline")
        print("✅ Non-blocking I/O operations")
        print("✅ Automatic audio buffering")
        print("✅ Speech-to-text with Whisper")
        print("✅ Text translation with Google Translate")
        print("✅ Text-to-speech with gTTS")
        
    else:
        print("\n💥 Installation testing failed!")
        print("Please check the error messages above and try again.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure stable internet connection")
        print("2. Try: pip install --upgrade pip")
        print("3. Try: pip install -r whisper_requirements.txt")
        print("4. Check Python version (3.8+ required)")
        sys.exit(1)

if __name__ == "__main__":
    main()
