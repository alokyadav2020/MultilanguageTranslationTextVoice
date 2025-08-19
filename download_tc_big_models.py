#!/usr/bin/env python3
"""
Download script for Helsinki-NLP OPUS tc-big translation models.
Downloads all required models for English, French, and Arabic translations.
Optimized for Nvidia Jetson systems.
"""

import os
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download_translation_model(repo_id, local_dir):
    """Download a single translation model using your exact structure."""
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"✓ {local_dir} already contains files — skipping download.")
        return True

    try:
        print(f"⏬ Downloading {repo_id} …")
        
        # Download tokenizer and model
        tok = AutoTokenizer.from_pretrained(repo_id)
        mod = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

        # Ensure directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Save to local directory
        tok.save_pretrained(local_dir)
        mod.save_pretrained(local_dir)
        
        print(f"✅ Saved model to {local_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {str(e)}")
        return False

def download_all_tc_big_models():
    """Download all required tc-big models for the translation service."""
    
    # Base directory for models (matching your Jetson structure)
    base_dir = "/home/orin/Desktop/translation_project/MultilanguageTranslationTextVoice/artifacts/models"
    
    # All tc-big models needed (matching your download script)
    models = {
        "Helsinki-NLP/opus-mt-tc-big-en-fr": "opus-mt-tc-big-en-fr",
        "Helsinki-NLP/opus-mt-tc-big-fr-en": "opus-mt-tc-big-fr-en",
        "Helsinki-NLP/opus-mt-tc-big-en-ar": "opus-mt-tc-big-en-ar",
        "Helsinki-NLP/opus-mt-tc-big-ar-en": "opus-mt-tc-big-ar-en",
        "Helsinki-NLP/opus-mt-tc-big-fr-ar": "opus-mt-tc-big-fr-ar",
        "Helsinki-NLP/opus-mt-tc-big-ar-fr": "opus-mt-tc-big-ar-fr",
    }
    
    print("🚀 Starting download of all Helsinki-NLP TC-Big translation models...")
    print(f"📂 Base directory: {base_dir}")
    print(f"📋 Models to download: {len(models)}")
    
    success_count = 0
    total_count = len(models)
    
    for repo_id, local_name in models.items():
        local_dir = os.path.join(base_dir, local_name)
        print(f"\n📥 Processing {repo_id}...")
        
        if download_translation_model(repo_id, local_dir):
            success_count += 1
        else:
            print(f"⚠️  Failed to download {repo_id}")
    
    print(f"\n🎉 Download complete! {success_count}/{total_count} models downloaded successfully.")
    
    if success_count == total_count:
        print("✅ All TC-Big models ready for translation service!")
        print("\n📋 Next steps:")
        print("1. Test the service: python test_tc_big_startup.py")
        print("2. Start the server: uvicorn app.main:app --reload")
    else:
        print("⚠️  Some models failed to download. Check internet connection and try again.")
    
    return success_count == total_count

def main():
    parser = argparse.ArgumentParser(description="Download Helsinki-NLP TC-Big translation models")
    parser.add_argument("--repo", help="Specific repository ID to download")
    parser.add_argument("--out", help="Output directory for specific model")
    parser.add_argument("--all", action="store_true", help="Download all required models")
    
    args = parser.parse_args()
    
    if args.all:
        download_all_tc_big_models()
    elif args.repo and args.out:
        download_translation_model(args.repo, args.out)
    else:
        print("Usage:")
        print("  Download all models: python download_tc_big_models.py --all")
        print("  Download specific model: python download_tc_big_models.py --repo REPO_ID --out OUTPUT_DIR")
        print("\nExample:")
        print("  python download_tc_big_models.py --repo Helsinki-NLP/opus-mt-tc-big-en-fr --out ./models/opus-mt-tc-big-en-fr")

if __name__ == "__main__":
    main()
