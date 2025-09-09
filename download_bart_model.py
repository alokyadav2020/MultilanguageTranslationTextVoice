#!/usr/bin/env python3
"""
Download facebook/bart-large-cnn model for chat summarization
Saves model to artifacts/models/bart-large-cnn/ folder
"""

import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_bart_model():
    """Download the BART-Large-CNN model to artifacts/models folder"""
    
    model_name = "facebook/bart-large-cnn"
    
    # Create artifacts/models directory
    artifacts_dir = Path("artifacts/models/bart-large-cnn")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🤖 BART-Large-CNN Model Downloader")
    logger.info("=" * 50)
    logger.info(f"📥 Downloading {model_name}...")
    logger.info("📊 Model size: ~1.6GB")
    logger.info(f"📁 Target directory: {artifacts_dir}")
    
    try:
        # Check if model already exists
        if (artifacts_dir / "config.json").exists() and (artifacts_dir / "pytorch_model.bin").exists():
            logger.info("✅ Model already exists in artifacts/models/bart-large-cnn")
            logger.info("🔄 Testing existing model...")
            
            # Load existing model
            tokenizer = AutoTokenizer.from_pretrained(str(artifacts_dir))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(artifacts_dir))
            
        else:
            # Check GPU availability
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory}GB)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("🍎 Apple Silicon MPS detected")
            else:
                device = "cpu"
                logger.info("💻 Using CPU")
            
            # Download tokenizer to artifacts folder
            logger.info("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(artifacts_dir))
            logger.info(f"✅ Tokenizer saved to {artifacts_dir}")
            
            # Download model to artifacts folder
            logger.info("📥 Downloading model (this may take a few minutes)...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            model.save_pretrained(str(artifacts_dir))
            logger.info(f"✅ Model saved to {artifacts_dir}")
        
        # Test the model
        logger.info("🧪 Testing model...")
        test_text = "This is a test conversation between two users discussing various topics."
        
        # Tokenize
        inputs = tokenizer.encode(test_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=50,
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info(f"🎯 Test summary: {summary}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 BART Model Download Complete!")
        logger.info("=" * 50)
        logger.info("✅ Model: facebook/bart-large-cnn")
        logger.info("✅ Size: ~1.6GB")
        logger.info(f"✅ Location: {artifacts_dir}")
        logger.info("✅ Ready for chat summarization")
        logger.info("\n📱 Next steps:")
        logger.info("   1. Use chat summary feature in your app")
        logger.info("   2. Model will load from artifacts/models folder")
        logger.info("   3. Enjoy fast AI-powered summaries!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error("💡 Try running: pip install transformers torch accelerate")
        return False

if __name__ == "__main__":
    success = download_bart_model()
    exit(0 if success else 1)
