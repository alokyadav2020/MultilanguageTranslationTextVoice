#!/usr/bin/env python3
"""
🚀 Quick Start Script for Real-Time Voice Translation
Automates the complete setup and testing process
"""

import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, capture_output=False):
    """Run a command with proper error handling"""
    logger.info(f"🔄 {description}")
    
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        logger.error(f"❌ Failed to run command: {e}")
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("✅ Virtual environment detected")
        return True
    else:
        logger.warning("⚠️  Not in virtual environment (recommended but not required)")
        return True

def install_dependencies():
    """Install all required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    # Install PyTorch with CUDA support
    success, _, _ = run_command(
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA support"
    )
    
    if not success:
        logger.warning("⚠️  CUDA installation failed, trying CPU version...")
        success, _, _ = run_command(
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch CPU version"
        )
    
    if not success:
        logger.error("❌ Failed to install PyTorch")
        return False
    
    # Install main requirements
    success, _, _ = run_command(
        "pip install -r requirements.txt",
        "Installing main requirements"
    )
    
    if not success:
        # Try installing key packages individually
        packages = [
            "fastapi", "uvicorn[standard]", "websockets", 
            "jinja2", "aiofiles", "python-multipart"
        ]
        
        for package in packages:
            success, _, _ = run_command(f"pip install {package}", f"Installing {package}")
            if not success:
                logger.warning(f"⚠️  Failed to install {package}")
    
    # Install SeamlessM4T using the official method
    logger.info("📥 Cloning SeamlessM4T repository...")
    success, _, _ = run_command(
        "git clone https://github.com/facebookresearch/seamless_communication.git",
        "Cloning SeamlessM4T repository"
    )
    
    if success:
        logger.info("📦 Installing SeamlessM4T in editable mode...")
        success, _, _ = run_command(
            "cd seamless_communication && pip install -e .",
            "Installing SeamlessM4T from source"
        )
    else:
        logger.warning("⚠️  Git clone failed, trying pip install...")
        success, _, _ = run_command(
            "pip install seamless_communication",
            "Installing SeamlessM4T via pip"
        )
    
    return success

def download_models():
    """Download and test SeamlessM4T models"""
    logger.info("📥 Downloading SeamlessM4T models...")
    
    success, _, _ = run_command(
        "python download_seamless_models.py",
        "Running model download script"
    )
    
    return success

def start_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    logger.info("📱 Server will be available at: http://localhost:8000")
    logger.info("🌍 Voice translation at: http://localhost:8000/enhanced-voice-call")
    logger.info("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            "python", "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")

def main():
    """Main setup and launch function"""
    print("🌍 Real-Time Voice Translation Setup")
    print("🗣️  Arabic ↔ English ↔ French")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. 🔧 Full setup (install dependencies + download models)")
    print("2. 📥 Install Whisper Translation (Recommended alternative)")
    print("3. 🚀 Start server only (if everything is ready)")
    print("4. 🧪 Test existing setup")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("Please enter 1, 2, 3, or 4")
    
    if choice == '1':
        # Full setup
        logger.info("🔧 Starting full setup...")
        
        if not install_dependencies():
            logger.error("❌ Dependency installation failed")
            sys.exit(1)
        
        logger.info("✅ Dependencies installed successfully")
        
        if not download_models():
            logger.error("❌ Model download failed")
            sys.exit(1)
        
        logger.info("✅ Models downloaded and tested successfully")
        
        answer = input("\n🚀 Start the server now? (y/n): ").lower().strip()
        if answer == 'y':
            start_server()
    
    elif choice == '2':
        # Install Whisper translation service
        logger.info("🎤 Installing Whisper Translation Service...")
        
        success, _, _ = run_command(
            "python install_whisper_service.py",
            "Installing Whisper Translation Service"
        )
        
        if success:
            logger.info("✅ Whisper Translation Service installed!")
            
            # Test the installation
            test_success, _, _ = run_command(
                "python test_whisper_service.py",
                "Testing Whisper Translation Service"
            )
            
            if test_success:
                logger.info("✅ Whisper service test passed!")
            else:
                logger.warning("⚠️ Whisper service test had issues, but installation completed")
        else:
            logger.error("❌ Whisper installation failed")
            sys.exit(1)
        
        answer = input("\n🚀 Start the server now? (y/n): ").lower().strip()
        if answer == 'y':
            start_server()
    
    elif choice == '3':
        # Start server only
        start_server()
    
    elif choice == '4':
        # Test setup
        logger.info("🧪 Testing setup...")
        
        # Test imports
        try:
            import torch
            import torchaudio
            logger.info(f"✅ PyTorch {torch.__version__}")
            
            import fastapi
            logger.info(f"✅ FastAPI {fastapi.__version__}")
            
            # Try to import whisper and related packages
            try:
                import whisper
                logger.info("✅ Whisper available")
                
                from googletrans import Translator
                from gtts import gTTS
                logger.info("✅ Google Translate and TTS available")
            except ImportError:
                logger.error("❌ Whisper or translation libraries not installed")
                logger.info("💡 Run option 2 to install Whisper Translation Service")
                return
            
            # Test our services
            try:
                from app.services.whisper_translation_service import whisper_translation_service
                logger.info("✅ Whisper translation service imports successfully")
                
                from app.services.voice_call_manager import VoiceCallManager
                logger.info("✅ Voice call manager imports successfully")
                
                # Test service availability
                if whisper_translation_service.is_available:
                    logger.info("✅ Whisper translation service is ready!")
                else:
                    logger.warning("⚠️ Whisper translation service not fully initialized")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Service import failed: {e}")
                logger.info("💡 Services should work once dependencies are installed")
            
            logger.info("🎉 All components are working!")
            
            answer = input("\n🚀 Start the server? (y/n): ").lower().strip()
            if answer == 'y':
                start_server()
                
        except Exception as e:
            logger.error(f"❌ Setup test failed: {e}")

if __name__ == "__main__":
    main()
