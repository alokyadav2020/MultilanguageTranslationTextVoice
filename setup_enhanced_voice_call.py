#!/usr/bin/env python3
"""
Setup script for Enhanced Voice Call with SeamlessM4T Real-time Translation
Supports Arabic, English, and French voice-to-voice translation
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"🔄 {description}")
    logger.info(f"Running: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    return True

def check_gpu_support():
    """Check if GPU support is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"✅ GPU support available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            logger.info("ℹ️ No GPU detected, will use CPU (slower but functional)")
            return False
    except ImportError:
        logger.info("ℹ️ PyTorch not installed yet, will check GPU support after installation")
        return False

def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    # Check if CUDA is available
    cuda_available = False
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        cuda_available = result.returncode == 0
    except Exception:
        pass
    
    if cuda_available:
        logger.info("🎮 CUDA detected, installing PyTorch with GPU support")
        command = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        logger.info("💻 Installing PyTorch for CPU")
        command = "pip install torch torchaudio"
    
    return run_command(command, "Installing PyTorch")

def install_seamless():
    """Install SeamlessM4T and dependencies"""
    commands = [
        ("pip install seamless_communication", "Installing SeamlessM4T"),
        ("pip install librosa soundfile", "Installing audio processing libraries"),
        ("pip install accelerate transformers sentencepiece", "Installing additional dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def test_installation():
    """Test if SeamlessM4T installation works"""
    logger.info("🧪 Testing SeamlessM4T installation...")
    
    test_script = """
try:
    import torch
    import torchaudio
    from seamless_communication.models.inference import Translator
    
    print("✅ All imports successful")
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🎵 TorchAudio version: {torchaudio.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU device: {torch.cuda.get_device_name(0)}")
    
    print("✅ SeamlessM4T installation test passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ Installation test passed!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ Installation test failed:")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "app/static/audio",
        "app/static/uploads",
        "artifacts/models",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def main():
    """Main setup function"""
    logger.info("🚀 Starting Enhanced Voice Call Setup")
    logger.info("🌍 Supporting Arabic, English, and French real-time translation")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install PyTorch
    if not install_pytorch():
        logger.error("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Install SeamlessM4T
    if not install_seamless():
        logger.error("❌ Failed to install SeamlessM4T")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        logger.error("❌ Installation test failed")
        sys.exit(1)
    
    # Final checks
    check_gpu_support()
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info("1. Start your FastAPI server: python -m uvicorn app.main:app --reload")
    logger.info("2. Visit: http://localhost:8000/enhanced-voice-call")
    logger.info("3. The first run will download SeamlessM4T models (~2-4GB)")
    logger.info("4. Supported languages: Arabic (ar), English (en), French (fr)")
    logger.info("")
    logger.info("🔧 API endpoints:")
    logger.info("- POST /api/voice-call/initiate - Start a call")
    logger.info("- WebSocket /api/voice-call/ws/{call_id} - Real-time translation")
    logger.info("- GET /api/voice-call/translation/test - Test translation service")

if __name__ == "__main__":
    main()
