#!/usr/bin/env python3
"""
Download Llama 3.1 8B Quantized Model from Hugging Face
This script downloads the quantized Llama 3.1 8B model for chat summarization
"""

import sys
from pathlib import Path
import requests
from tqdm import tqdm
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaModelDownloader:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.artifacts_path = self.base_path / "artifacts" / "models" / "llama3.1-8b-instruct-q4"
        self.model_repo = "microsoft/Llama-2-7b-chat-hf"  # Using a lighter alternative
        # For actual Llama 3.1 8B quantized, use: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        self.hf_token = None  # Add your HuggingFace token if needed
        
        # Model files to download (quantized GGUF format)
        self.model_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin",  # Main model file
            "generation_config.json"
        ]
        
        # Alternative: Use GGUF quantized files
        self.gguf_model_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        self.gguf_filename = "llama-3.1-8b-instruct-q4_k_m.gguf"
    
    def create_directories(self):
        """Create necessary directories"""
        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {self.artifacts_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def download_file(self, url: str, filename: str, description: str = "") -> bool:
        """Download a file with progress bar"""
        try:
            filepath = self.artifacts_path / filename
            
            # Skip if file already exists
            if filepath.exists():
                logger.info(f"⏭️  {filename} already exists, skipping download")
                return True
            
            logger.info(f"📥 Downloading {description or filename}...")
            
            headers = {}
            if self.hf_token:
                headers['Authorization'] = f'Bearer {self.hf_token}'
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def download_gguf_model(self) -> bool:
        """Download GGUF quantized model (recommended for CPU inference)"""
        logger.info("🚀 Starting GGUF model download...")
        
        success = self.download_file(
            self.gguf_model_url, 
            self.gguf_filename,
            "Llama 3.1 8B Instruct Q4_K_M (GGUF)"
        )
        
        if success:
            # Create model info file
            model_info = {
                "model_name": "Meta-Llama-3.1-8B-Instruct",
                "quantization": "Q4_K_M",
                "format": "GGUF",
                "file_path": str(self.artifacts_path / self.gguf_filename),
                "downloaded_at": datetime.now().isoformat(),
                "description": "Quantized Llama 3.1 8B model for chat summarization",
                "usage": "Use with llama-cpp-python for efficient CPU inference"
            }
            
            info_file = self.artifacts_path / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info("📄 Created model info file")
        
        return success
    
    def verify_download(self) -> bool:
        """Verify downloaded files"""
        try:
            model_file = self.artifacts_path / self.gguf_filename
            
            if not model_file.exists():
                logger.error("❌ Model file not found")
                return False
            
            file_size = model_file.stat().st_size
            logger.info(f"✅ Model file size: {file_size / (1024**3):.2f} GB")
            
            # Check minimum expected size (should be > 4GB for 8B model)
            if file_size < 3 * 1024**3:  # 3GB minimum
                logger.warning("⚠️  Model file seems smaller than expected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required packages for model inference"""
        try:
            logger.info("📦 Installing required packages...")
            
            # Basic packages
            packages = [
                "llama-cpp-python",
                "transformers",
                "accelerate",
                "requests",
                "tqdm"
            ]
            
            # GPU-specific packages
            gpu_packages = []
            
            # Check for NVIDIA GPU
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("🎮 NVIDIA GPU detected, adding CUDA support packages")
                    gpu_packages.extend([
                        "torch",
                        "bitsandbytes",
                        "nvidia-ml-py3"
                    ])
                    # Install CUDA-enabled llama-cpp-python
                    logger.info("Installing CUDA-enabled llama-cpp-python...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "llama-cpp-python[cuda]", "--force-reinstall", "--no-cache-dir"
                    ], check=False)
            except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
                logger.info("🖥️  No NVIDIA GPU detected or nvidia-smi not available")
            
            # Check for AMD GPU
            try:
                import subprocess
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("🎮 AMD GPU detected, adding ROCm support")
                    gpu_packages.extend([
                        "torch-rocm",
                    ])
            except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
                pass
            
            # Install all packages
            all_packages = packages + gpu_packages
            
            for package in all_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ {package} installed successfully")
                else:
                    logger.warning(f"⚠️  Failed to install {package}: {result.stderr}")
                    # Continue with other packages
            
            # Try to install OpenCL support for Intel/AMD GPUs
            try:
                logger.info("Installing OpenCL support...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "pyopencl"
                ], capture_output=True, text=True, check=False)
            except Exception:
                pass
            
            logger.info("✅ Dependency installation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def create_usage_example(self):
        """Create example usage script"""
        example_code = '''
"""
Example usage of the downloaded Llama 3.1 8B model for chat summarization
"""

from llama_cpp import Llama
import json

def load_model():
    """Load the quantized Llama model"""
    model_path = "artifacts/models/llama3.1-8b-instruct-q4/llama-3.1-8b-instruct-q4_k_m.gguf"
    
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # Context window
        n_threads=4,  # Number of CPU threads
        n_gpu_layers=0,  # Use 0 for CPU-only inference
        verbose=False
    )
    
    return llm

def summarize_chat(chat_messages, user_language="en"):
    """Summarize chat messages"""
    llm = load_model()
    
    # Format chat for summarization
    chat_text = "\\n".join([f"{msg['sender']}: {msg['text']}" for msg in chat_messages])
    
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that summarizes chat conversations. Create a concise summary in {user_language}.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please summarize this chat conversation:

{chat_text}

Provide a summary that includes:
1. Main topics discussed
2. Key decisions or agreements
3. Action items (if any)
4. Overall tone of the conversation

Summary:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["<|eot_id|>"]
    )
    
    return response['choices'][0]['text'].strip()

# Test the model
if __name__ == "__main__":
    test_messages = [
        {"sender": "Alice", "text": "Hi, how are you?"},
        {"sender": "Bob", "text": "I'm good, thanks! How about you?"},
        {"sender": "Alice", "text": "Great! Let's discuss the project."},
        {"sender": "Bob", "text": "Sure, I think we should focus on the API design first."}
    ]
    
    summary = summarize_chat(test_messages)
    print("Summary:", summary)
'''
        
        example_file = self.artifacts_path / "usage_example.py"
        with open(example_file, 'w') as f:
            f.write(example_code)
        
        logger.info("📝 Created usage example script")
    
    def run_download(self) -> bool:
        """Main download process"""
        logger.info("🚀 Starting Llama 3.1 8B model download process...")
        logger.info("=" * 60)
        
        # Step 1: Create directories
        if not self.create_directories():
            return False
        
        # Step 2: Install dependencies
        logger.info("📦 Installing dependencies...")
        if not self.install_dependencies():
            logger.warning("⚠️  Some dependencies failed to install. Continuing...")
        
        # Step 3: Download model
        logger.info("📥 Downloading quantized model...")
        if not self.download_gguf_model():
            return False
        
        # Step 4: Verify download
        logger.info("🔍 Verifying download...")
        if not self.verify_download():
            return False
        
        # Step 5: Create usage example
        self.create_usage_example()
        
        logger.info("🎉 Model download completed successfully!")
        logger.info("=" * 60)
        logger.info(f"📁 Model location: {self.artifacts_path}")
        logger.info(f"📄 Model file: {self.gguf_filename}")
        logger.info("📝 Usage example: usage_example.py")
        logger.info("🚀 Ready for chat summarization!")
        
        return True

def main():
    """Main function"""
    print("🦙 Llama 3.1 8B Quantized Model Downloader")
    print("=" * 50)
    
    downloader = LlamaModelDownloader()
    
    try:
        success = downloader.run_download()
        
        if success:
            print("\n✅ Download completed successfully!")
            print("\nNext steps:")
            print("1. Test the model with: python artifacts/models/llama3.1-8b-instruct-q4/usage_example.py")
            print("2. Integrate with chat summary service")
            print("3. Add summary buttons to chat interface")
        else:
            print("\n❌ Download failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
