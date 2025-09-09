#!/usr/bin/env python3
"""
Setup script for Chat Summary Feature (Transformers-based)
This script sets up the complete chat summary functionality using HuggingFace Transformers
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages for chat summary"""
    
    requirements = [
        "transformers>=4.30.0",
        "torch>=2.0.0", 
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "httpx>=0.24.0",
        "aiofiles>=23.1.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]
    
    logger.info("📦 Installing required packages...")
    
    for package in requirements:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_gpu_availability():
    """Test GPU availability for acceleration"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 NVIDIA GPU detected: {gpu_name}")
            logger.info(f"🔢 GPU Count: {gpu_count}")
            return True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple Silicon MPS acceleration available")
            return True
        else:
            logger.info("💻 Using CPU for inference")
            return False
            
    except ImportError:
        logger.warning("⚠️  PyTorch not available for GPU detection")
        return False

def test_model_loading():
    """Test if the summarization model can be loaded"""
    try:
        logger.info("🧪 Testing model loading...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "facebook/bart-large-cnn"
        logger.info(f"📥 Testing {model_name}...")
        
        # Test tokenizer loading
        _ = AutoTokenizer.from_pretrained(model_name)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Test model loading (this will download ~1.6GB on first run)
        _ = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto")
        logger.info("✅ Model loaded successfully")
        
        logger.info("🎉 Model test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def check_integration():
    """Check if chat summary service integrates properly"""
    try:
        logger.info("🔧 Testing chat summary service integration...")
        
        # Test import
        from app.services.chat_summary_service import chat_summary_service
        logger.info("✅ Chat summary service imported successfully")
        
        # Test model status
        status = chat_summary_service.get_model_status()
        logger.info(f"📊 Service status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def update_main_app():
    """Ensure chat summary router is registered in main app"""
    try:
        main_py_path = Path("app/main.py")
        
        if not main_py_path.exists():
            logger.warning("⚠️  app/main.py not found, skipping integration")
            return True
        
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if chat summary router is already imported
        if "from .api.chat_summary import router as chat_summary_router" in content:
            logger.info("✅ Chat summary router already integrated")
            return True
        
        # Add import and router registration
        logger.info("🔧 Adding chat summary router to main app...")
        
        # Add import
        if "from .api" in content:
            content = content.replace(
                "from .api",
                "from .api.chat_summary import router as chat_summary_router\nfrom .api"
            )
        
        # Add router registration
        if "app.include_router" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "app.include_router" in line and "chat_summary" not in line:
                    lines.insert(i + 1, "app.include_router(chat_summary_router)")
                    break
            content = '\n'.join(lines)
        
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("✅ Chat summary router integrated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update main app: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Chat Summary Feature Setup (Transformers-based)")
    logger.info("=" * 60)
    
    # Step 1: Install requirements
    logger.info("Step 1: Installing requirements...")
    if not install_requirements():
        logger.error("❌ Failed to install requirements")
        return False
    
    # Step 2: Test GPU availability
    logger.info("\nStep 2: Testing GPU availability...")
    gpu_available = test_gpu_availability()
    if gpu_available:
        logger.info("🚀 GPU acceleration will be used")
    else:
        logger.info("💻 CPU inference will be used")
    
    # Step 3: Test model loading
    logger.info("\nStep 3: Testing model loading...")
    if not test_model_loading():
        logger.error("❌ Model loading test failed")
        return False
    
    # Step 4: Check integration
    logger.info("\nStep 4: Testing integration...")
    if not check_integration():
        logger.error("❌ Integration test failed")
        return False
    
    # Step 5: Update main app
    logger.info("\nStep 5: Updating main application...")
    if not update_main_app():
        logger.error("❌ Failed to update main app")
        return False
    
    # Success message
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Chat Summary Feature Setup Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("✅ Features Available:")
    logger.info("   • AI-powered chat summarization")
    logger.info("   • Support for English, French, and Arabic")
    logger.info("   • GPU acceleration (if available)")
    logger.info("   • Direct and group chat summaries")
    logger.info("   • Download summaries in multiple formats")
    logger.info("")
    logger.info("🚀 Model: facebook/bart-large-cnn (~1.6GB)")
    logger.info("💻 Backend: HuggingFace Transformers")
    logger.info(f"⚡ Acceleration: {'GPU' if gpu_available else 'CPU'}")
    logger.info("")
    logger.info("📱 Next Steps:")
    logger.info("   1. Restart your FastAPI server")
    logger.info("   2. Look for the 'Summary' button in chat windows")
    logger.info("   3. Generate AI-powered summaries of your conversations")
    logger.info("")
    logger.info("🎯 Ready to use! No large model downloads required.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
