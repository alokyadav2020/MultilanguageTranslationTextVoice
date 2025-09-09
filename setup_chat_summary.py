#!/usr/bin/env python3
"""
Chat Summary Setup Script
Installs and configures the chat summary functionality with Llama 3.1 8B
"""

import sys
import subprocess
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatSummarySetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "env"
        self.artifacts_path = self.project_root / "artifacts" / "models"
        
    def check_prerequisites(self):
        """Check if Python and pip are available"""
        try:
            # Check Python version
            result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
            python_version = result.stdout.strip()
            logger.info(f"✅ {python_version}")
            
            # Check if we're in a virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                logger.info("✅ Virtual environment detected")
            else:
                logger.warning("⚠️  Not in a virtual environment. Consider using one.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Python check failed: {e}")
            return False
    
    def install_dependencies(self):
        """Install required Python packages"""
        try:
            logger.info("📦 Installing Python dependencies...")
            
            packages = [
                "llama-cpp-python",
                "requests",
                "tqdm",
                "aiofiles",
                "asyncio",
            ]
            
            for package in packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ {package} installed successfully")
                else:
                    logger.error(f"❌ Failed to install {package}: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def download_model(self):
        """Download the Llama model"""
        try:
            logger.info("🦙 Downloading Llama 3.1 8B quantized model...")
            
            # Run the download script
            download_script = self.project_root / "download_llama_model.py"
            if not download_script.exists():
                logger.error("❌ download_llama_model.py not found")
                return False
            
            result = subprocess.run([
                sys.executable, str(download_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Model download completed")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Model download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model download error: {e}")
            return False
    
    def test_summary_service(self):
        """Test the chat summary service"""
        try:
            logger.info("🧪 Testing chat summary service...")
            
            # Import and test the service
            sys.path.append(str(self.project_root))
            
            from app.services.chat_summary_service import chat_summary_service
            
            # Test model loading
            status = chat_summary_service.get_model_status()
            
            if status["model_exists"]:
                logger.info("✅ Model file found")
            else:
                logger.error("❌ Model file not found")
                return False
            
            # Test with sample data
            test_messages = [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "sender_name": "Alice",
                    "message_type": "text",
                    "original_text": "Hello, how are you doing today?",
                    "original_language": "en"
                },
                {
                    "timestamp": "2024-01-01T10:01:00",
                    "sender_name": "Bob", 
                    "message_type": "text",
                    "original_text": "I'm doing great! Working on the new project.",
                    "original_language": "en"
                }
            ]
            
            async def run_test():
                result = await chat_summary_service.generate_chat_summary(
                    messages=test_messages,
                    user_language="en",
                    chat_type="direct",
                    participants=["Alice", "Bob"],
                    user_id=1
                )
                return result
            
            # Run async test
            result = asyncio.run(run_test())
            
            if result["success"]:
                logger.info("✅ Summary generation test successful")
                logger.info(f"Generated summary preview: {result['summary'][:100]}...")
                return True
            else:
                logger.error(f"❌ Summary generation failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Service test failed: {e}")
            return False
    
    def create_config_file(self):
        """Create configuration file for summary service"""
        try:
            config_content = f"""
# Chat Summary Configuration
# Generated by setup script

[summary_service]
model_path = {self.artifacts_path}/llama3.1-8b-instruct-q4/llama-3.1-8b-instruct-q4_k_m.gguf
max_context_length = 4096
max_tokens = 800
temperature = 0.7
top_p = 0.9

[supported_languages]
english = en
arabic = ar  
french = fr
spanish = es
german = de
italian = it
portuguese = pt
russian = ru
japanese = ja
korean = ko
chinese = zh

[api_endpoints]
direct_summary = /api/chat-summary/direct/{{user_id}}
group_summary = /api/chat-summary/group/{{group_id}}
download_direct = /api/chat-summary/direct/{{user_id}}/download
download_group = /api/chat-summary/group/{{group_id}}/download
status = /api/chat-summary/status
"""
            
            config_file = self.project_root / "chat_summary_config.ini"
            with open(config_file, 'w') as f:
                f.write(config_content.strip())
            
            logger.info(f"✅ Configuration file created: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config file creation failed: {e}")
            return False
    
    def create_usage_guide(self):
        """Create usage guide for the summary feature"""
        try:
            guide_content = """
# 💬 Chat Summary Feature - Usage Guide

## 🎯 Overview
The Chat Summary feature uses AI (Llama 3.1 8B) to generate intelligent summaries of your chat conversations in any supported language.

## 🚀 How to Use

### For Direct Chats:
1. Open any direct chat conversation
2. Click the "Summary" button in the chat header
3. AI will analyze the conversation and generate a summary
4. Download the summary in Markdown or Text format
5. Choose your preferred language for the summary

### For Group Chats:
1. Open any group chat
2. Click the "Group Summary" button
3. AI will analyze the group conversation
4. Download the summary with participant analysis

## 🌍 Supported Languages
- English (en)
- Arabic (ar) 
- French (fr)
- Spanish (es)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)

## 📊 What's Included in Summaries
- **Main Topics**: Key subjects discussed
- **Participation Analysis**: Who spoke most, engagement patterns
- **Communication Patterns**: Voice vs text usage, language patterns
- **Key Decisions**: Important conclusions or agreements
- **Action Items**: Follow-up tasks mentioned
- **Statistics**: Message counts, duration, language usage

## 🔧 API Endpoints

### Generate Summaries
- `GET /api/chat-summary/direct/{user_id}?language=en`
- `GET /api/chat-summary/group/{group_id}?language=en`

### Download Summaries  
- `GET /api/chat-summary/direct/{user_id}/download?format=markdown&language=en`
- `GET /api/chat-summary/group/{group_id}/download?format=markdown&language=en`

### Service Status
- `GET /api/chat-summary/status`

## 🎨 Frontend Integration

The summary functionality is automatically added to chat interfaces with:
- Summary generation buttons
- Download options
- Language selection
- Progress indicators
- Error handling

## 🔒 Privacy & Security
- Summaries are generated locally using your own AI model
- No data is sent to external services
- Summaries can be deleted after download
- User authentication is required for all operations

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure the model file exists in: `artifacts/models/llama3.1-8b-instruct-q4/`
- Check available disk space (model is ~4GB)
- Verify Python dependencies are installed

### Summary Generation Fails
- Check server logs for detailed error messages
- Ensure conversation has messages to summarize
- Try with a smaller conversation first

### Performance Issues
- Model runs on CPU by default (slower but more compatible)
- For faster performance, ensure adequate RAM (8GB+ recommended)
- Close other memory-intensive applications

## 📞 Support
- Check the application logs for detailed error messages
- Verify model installation with: `python -c "from app.services.chat_summary_service import chat_summary_service; print(chat_summary_service.get_model_status())"`
- Test the service with: `python app/services/chat_summary_service.py`

## 🎉 Enjoy Your AI-Powered Chat Summaries!
"""
            
            guide_file = self.project_root / "CHAT_SUMMARY_GUIDE.md"
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide_content.strip())
            
            logger.info(f"✅ Usage guide created: {guide_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Guide creation failed: {e}")
            return False
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        logger.info("🚀 Starting Chat Summary Setup")
        logger.info("=" * 60)
        
        steps = [
            ("Checking Prerequisites", self.check_prerequisites),
            ("Installing Dependencies", self.install_dependencies), 
            ("Downloading AI Model", self.download_model),
            ("Testing Summary Service", self.test_summary_service),
            ("Creating Configuration", self.create_config_file),
            ("Creating Usage Guide", self.create_usage_guide)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
            logger.info(f"✅ {step_name} completed")
            logger.info("-" * 40)
        
        logger.info("🎉 Chat Summary Setup Completed Successfully!")
        logger.info("=" * 60)
        logger.info("📚 Next Steps:")
        logger.info("1. Start your FastAPI application: python app/main.py") 
        logger.info("2. Open a chat conversation")
        logger.info("3. Click the 'Summary' button to test")
        logger.info("4. Read CHAT_SUMMARY_GUIDE.md for detailed usage")
        logger.info("=" * 60)
        
        return True

def main():
    """Main setup function"""
    print("🤖 Chat Summary Feature Setup")
    print("=" * 50)
    print("This script will install and configure AI-powered chat summarization")
    print("using Llama 3.1 8B quantized model for your translation app.")
    print()
    
    # Confirm setup
    try:
        confirm = input("Continue with setup? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Setup cancelled.")
            return 0
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 0
    
    # Run setup
    setup = ChatSummarySetup()
    
    try:
        success = setup.run_complete_setup()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected setup error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
