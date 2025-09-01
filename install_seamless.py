#!/usr/bin/env python3
"""
SeamlessM4T Installation Script
Follows the official installation method from Facebook Research
"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, cwd=None):
    """Run a command with proper error handling"""
    logger.info(f"🔄 {description}")
    
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - Success")
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            logger.error(f"❌ {description} - Failed")
            if result.stderr:
                logger.error(f"Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run command: {e}")
        return False

def check_git():
    """Check if git is available"""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True)
        if result.returncode == 0:
            logger.info("✅ Git is available")
            return True
        else:
            logger.error("❌ Git not found. Please install Git first.")
            return False
    except Exception:
        logger.error("❌ Git not found. Please install Git first.")
        return False

def install_seamless_communication():
    """Install SeamlessM4T following official method"""
    logger.info("🚀 Installing SeamlessM4T (Official Method)")
    
    # Check if git is available
    if not check_git():
        return False
    
    # Check if directory already exists
    seamless_dir = Path("seamless_communication")
    if seamless_dir.exists():
        logger.info("📁 seamless_communication directory already exists")
        answer = input("Do you want to remove it and reinstall? (y/n): ").lower().strip()
        if answer == 'y':
            import shutil
            shutil.rmtree(seamless_dir)
            logger.info("🗑️ Removed existing directory")
        else:
            logger.info("📦 Using existing installation")
            return install_in_existing_directory()
    
    # Step 1: Clone the repository
    success = run_command(
        "git clone https://github.com/facebookresearch/seamless_communication.git",
        "Cloning SeamlessM4T repository"
    )
    
    if not success:
        logger.error("❌ Failed to clone repository")
        return False
    
    # Step 2: Install in editable mode
    success = run_command(
        "pip install -e .",
        "Installing SeamlessM4T in editable mode",
        cwd="seamless_communication"
    )
    
    if not success:
        logger.error("❌ Failed to install SeamlessM4T")
        return False
    
    logger.info("✅ SeamlessM4T installed successfully!")
    return True

def install_in_existing_directory():
    """Install from existing directory"""
    seamless_dir = Path("seamless_communication")
    
    if not seamless_dir.exists():
        logger.error("❌ seamless_communication directory not found")
        return False
    
    # Pull latest changes
    success = run_command(
        "git pull",
        "Updating repository",
        cwd="seamless_communication"
    )
    
    # Install in editable mode
    success = run_command(
        "pip install -e .",
        "Installing SeamlessM4T in editable mode",
        cwd="seamless_communication"
    )
    
    return success

def test_installation():
    """Test the installation"""
    logger.info("🧪 Testing SeamlessM4T installation...")
    
    try:
        # Test import
        from seamless_communication.models.inference import Translator
        logger.info("✅ SeamlessM4T import successful")
        
        # Test model loading (this will download models on first use)
        logger.info("📥 Loading test model (this may download models)...")
        translator = Translator("seamlessM4T_large")
        logger.info("✅ Model loaded successfully")
        
        # Test basic translation
        logger.info("🔄 Testing basic translation...")
        import torch
        
        # Create dummy audio
        dummy_audio = torch.zeros(16000)  # 1 second of silence
        
        # Test translation
        text_out, audio_out = translator.predict(
            dummy_audio,
            src_lang="eng",
            tgt_lang="fra"
        )
        
        logger.info("✅ Translation test successful!")
        logger.info(f"📊 Audio output shape: {audio_out.shape if audio_out is not None else 'None'}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🌍 SeamlessM4T Official Installation")
    print("📋 Following: git clone + pip install -e .")
    print("=" * 50)
    
    # Install SeamlessM4T
    if not install_seamless_communication():
        logger.error("❌ Installation failed")
        sys.exit(1)
    
    # Test installation
    logger.info("🧪 Testing installation...")
    if test_installation():
        logger.info("🎉 SeamlessM4T is ready!")
        logger.info("📋 You can now use:")
        logger.info("   from seamless_communication.models.inference import Translator")
        logger.info("   translator = Translator('seamlessM4T_large')")
        logger.info("   text_out, audio_out = translator.predict(audio, src_lang='eng', tgt_lang='fra')")
    else:
        logger.error("❌ Installation test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
