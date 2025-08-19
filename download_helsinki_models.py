#!/usr/bin/env python3
"""
Helsinki-NLP Model Downloader for Jetson Translation Project
Downloads all required OPUS translation models for EN/FR/AR language pairs.
"""

import os
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download_translation_model(repo_id, local_dir, revision="main"):
    """Download and save Helsinki-NLP OPUS model to local directory."""
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"✓ {local_dir} already contains files — skipping download.")
        return

    print(f"⏬ Downloading {repo_id} …")
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        mod = AutoModelForSeq2SeqLM.from_pretrained(repo_id, revision=revision)

        os.makedirs(local_dir, exist_ok=True)
        tok.save_pretrained(local_dir)
        mod.save_pretrained(local_dir)
        print(f"✅ Saved model to {local_dir}")
        
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {str(e)}")
        return False
    
    return True

def download_all_models(base_dir="/home/orin/Desktop/translation_project/artifacts/translation_llm"):
    """Download all Helsinki-NLP OPUS models for the translation service."""
    
    # Helsinki-NLP model mappings (matching translation service)
    models = [
        ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-en-fr"),
        ("Helsinki-NLP/opus-mt-fr-en", "Helsinki-NLP/opus-mt-fr-en"),
        ("Helsinki-NLP/opus-mt-en-ar", "Helsinki-NLP/opus-mt-en-ar"),
        ("Helsinki-NLP/opus-mt-ar-en", "Helsinki-NLP/opus-mt-ar-en"),
        ("Helsinki-NLP/opus-mt-fr-ar", "Helsinki-NLP/opus-mt-fr-ar"),
        ("Helsinki-NLP/opus-mt-ar-fr", "Helsinki-NLP/opus-mt-ar-fr"),
    ]
    
    print("🚀 Starting Helsinki-NLP OPUS models download...")
    print(f"📂 Base directory: {base_dir}")
    print(f"📋 Models to download: {len(models)}")
    print()
    
    os.makedirs(base_dir, exist_ok=True)
    
    success_count = 0
    for repo_id, local_subdir in models:
        local_dir = os.path.join(base_dir, local_subdir)
        print(f"📥 Processing: {repo_id}")
        
        if download_translation_model(repo_id, local_dir):
            success_count += 1
        print()
    
    print(f"✅ Download complete: {success_count}/{len(models)} models successful")
    
    if success_count == len(models):
        print("🎉 All Helsinki-NLP OPUS models are ready for translation service!")
    else:
        print("⚠️  Some models failed to download. Check your internet connection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Helsinki-NLP OPUS translation models")
    parser.add_argument("--repo", default=None, help="Specific model repo to download")
    parser.add_argument("--out", default=None, help="Output directory for specific model")
    parser.add_argument("--base-dir", default="/home/orin/Desktop/translation_project/artifacts/translation_llm", 
                        help="Base directory for all models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if args.all or (not args.repo and not args.out):
        # Download all models
        download_all_models(args.base_dir)
    elif args.repo and args.out:
        # Download specific model
        download_translation_model(args.repo, args.out)
    else:
        print("❌ Error: Either use --all flag or provide both --repo and --out arguments")
        print("Examples:")
        print("  python download_helsinki_models.py --all")
        print("  python download_helsinki_models.py --repo Helsinki-NLP/opus-mt-en-fr --out /path/to/save")
