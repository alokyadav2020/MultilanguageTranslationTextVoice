# 🌍 Real-Time Voice Translation Setup Guide

## Overview
This guide will help you set up real-time voice-to-voice translation using SeamlessM4T for Arabic, English, and French languages during WebRTC voice calls.

## Prerequisites

### 1. Python Environment
- Python 3.8+ (3.9 or 3.10 recommended)
- Virtual environment activated

### 2. System Requirements
- **Minimum**: 8GB RAM, 10GB free disk space
- **Recommended**: 16GB RAM, CUDA GPU, 20GB free disk space
- **Network**: Stable internet for model download

## Installation Steps

### Step 1: Install Core Dependencies
```powershell
# Install PyTorch with CUDA support (if you have a GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU-only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI and WebRTC dependencies
pip install -r requirements.txt
```

### Step 2: Install SeamlessM4T (Official Method)
```powershell
# Method 1: Using the installation script (Recommended)
python install_seamless.py

# Method 2: Manual installation
git clone https://github.com/facebookresearch/seamless_communication.git
cd seamless_communication
pip install -e .

# Method 3: Alternative if git clone fails
pip install seamless_communication
```

### Step 3: Download and Test Models
```powershell
# Run the model download script
python download_seamless_models.py
```

This script will:
- ✅ Check all dependencies
- 🎮 Detect GPU/CPU
- 📥 Download models (~4-6 GB)
- 🧪 Test translation functionality
- 📊 Show cache statistics

### Step 4: Start the Application
```powershell
# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing the Voice Translation

### 1. Open Voice Call Interface
Visit: `http://localhost:8000/enhanced-voice-call`

### 2. Test with Sample Call
Visit: `http://localhost:8000/enhanced-voice-call?call_id=test&language=en`

### 3. Test Translation Languages
- **English**: `language=en`
- **Arabic**: `language=ar` 
- **French**: `language=fr`

## File Structure Overview

```
📁 MultilanguageTranslationTextVoice/
├── 🚀 app/
│   ├── 🔧 services/
│   │   ├── seamless_translation_service.py  # Core SeamlessM4T integration
│   │   └── voice_call_manager.py            # Call session management
│   ├── 🌐 api/
│   │   └── enhanced_voice_call.py           # WebSocket & REST API
│   ├── 🎨 static/js/
│   │   └── enhanced_voice_call.js           # Frontend voice call logic
│   └── 📄 templates/
│       └── enhanced_voice_call.html         # Voice call interface
├── 📥 download_seamless_models.py           # Model download & test script
├── ⚙️ setup_enhanced_voice_call.py          # Complete setup automation
└── 📖 README files with detailed documentation
```

## Key Features

### 🎯 Core Functionality
- **Real-time voice translation** between Arabic, English, and French
- **WebRTC peer-to-peer** voice calls
- **Chunked audio processing** (2-second chunks with 0.5s overlap)
- **WebSocket streaming** for low latency

### 🔧 Technical Components
- **SeamlessM4T v2_large**: State-of-the-art voice-to-voice translation
- **FastAPI backend**: High-performance async API
- **PyTorch**: ML model execution with GPU acceleration
- **16kHz audio processing**: Optimized for real-time performance

### 🌐 Language Support
| Language | Code | Flag |
|----------|------|------|
| Arabic   | `ar` | 🇸🇦  |
| English  | `en` | 🇺🇸  |
| French   | `fr` | 🇫🇷  |

## Troubleshooting

### Common Issues

#### 1. Model Download Fails
```powershell
# Check internet connection and retry
python download_seamless_models.py

# Clear cache and retry
python -c "import torch; import shutil; shutil.rmtree(torch.hub.get_dir(), ignore_errors=True)"
python download_seamless_models.py
```

#### 2. CUDA Out of Memory
```python
# Edit seamless_translation_service.py
# Change device selection:
device = torch.device("cpu")  # Force CPU usage
```

#### 3. Import Errors
```powershell
# Reinstall dependencies
pip uninstall seamless_communication
pip install seamless_communication

# Check Python environment
python -c "import torch, torchaudio; print('PyTorch OK')"
python -c "import seamless_communication; print('SeamlessM4T OK')"
```

#### 4. WebSocket Connection Issues
- Check firewall settings
- Ensure port 8000 is available
- Test with `localhost` instead of `0.0.0.0`

### Performance Optimization

#### GPU Acceleration
```python
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU usage during translation
nvidia-smi  # Run in separate terminal
```

#### Memory Management
- Close other applications during model download
- Use CPU if GPU memory is insufficient
- Consider smaller models for limited resources

## API Endpoints

### WebSocket Connection
```javascript
ws://localhost:8000/ws/enhanced-voice-call/{call_id}
```

### REST API
- `GET /enhanced-voice-call` - Voice call interface
- `POST /api/enhanced-voice-call/create` - Create new call
- `POST /api/enhanced-voice-call/join` - Join existing call
- `DELETE /api/enhanced-voice-call/{call_id}` - End call

## Development Mode

### Enable Debug Logging
```python
# In seamless_translation_service.py
logging.basicConfig(level=logging.DEBUG)
```

### Test Translation Manually
```python
python -c "
from app.services.seamless_translation_service import SeamlessTranslationService
service = SeamlessTranslationService()
print('Translation service ready!')
"
```

## Next Steps

1. **✅ Complete Setup**: Run `python download_seamless_models.py`
2. **🚀 Start Server**: `python -m uvicorn app.main:app --reload`
3. **🧪 Test Interface**: Visit `http://localhost:8000/enhanced-voice-call`
4. **📱 Test Translation**: Try voice calls between different languages
5. **🔧 Customize**: Modify language settings or add new features

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in the terminal
3. Test individual components with the provided scripts
4. Verify all dependencies are correctly installed

---

🎉 **Ready to translate voices in real-time!** 🌍🗣️
