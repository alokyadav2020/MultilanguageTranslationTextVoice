# Enhanced Voice Call with Real-time Translation

This module provides real-time voice-to-voice translation during voice calls using Facebook's SeamlessM4T model. It supports **Arabic**, **English**, and **French** with high-quality voice translation.

## 🌟 Features

- **Real-time Voice Translation**: Direct voice-to-voice translation without intermediate text
- **3 Languages**: Arabic (ar), English (en), French (fr)
- **WebRTC Integration**: Peer-to-peer voice calls with translation overlay
- **Low Latency**: ~2-3 second translation delay
- **High Quality**: Maintains voice characteristics and intonation
- **Visual Feedback**: Real-time audio visualization and transcription display

## 🏗️ Architecture

```
┌─────────────────┐    WebRTC P2P     ┌─────────────────┐
│   Client A      │◄─────────────────►│   Client B      │
│   (Arabic)      │                   │   (English)     │
└─────────────────┘                   └─────────────────┘
         │                                       │
         │ WebSocket                  WebSocket │
         │ (Audio chunks)          (Audio chunks)│
         ▼                                       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server                             │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Voice Call     │  │     SeamlessM4T             │  │
│  │  Manager        │  │   Translation Service       │  │
│  │                 │  │                              │  │
│  │ - WebSocket     │◄─┤ - Voice-to-Voice            │  │
│  │ - Call State    │  │ - Chunk Processing           │  │
│  │ - Participants  │  │ - Arabic ↔ English ↔ French │  │
│  └─────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
app/
├── api/
│   └── enhanced_voice_call.py      # Enhanced API with translation
├── services/
│   ├── seamless_translation_service.py  # SeamlessM4T integration
│   └── voice_call_manager.py       # Call management with translation
├── static/js/
│   └── enhanced_voice_call.js      # Frontend voice call manager
└── templates/
    └── enhanced_voice_call.html    # Enhanced UI with translation controls

setup_enhanced_voice_call.py        # Setup script
seamless_requirements.txt           # Additional requirements
```

## 🚀 Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
python setup_enhanced_voice_call.py
```

### Option 2: Manual Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchaudio

# For GPU support (if you have CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SeamlessM4T and dependencies
pip install seamless_communication
pip install librosa soundfile accelerate transformers sentencepiece
```

**Note**: First run will download SeamlessM4T models (~2-4GB). Ensure you have:
- Sufficient disk space (5GB+)
- Good internet connection
- Python 3.8+

## 🎯 Usage

### 1. Start Enhanced Voice Call

```python
# Navigate to enhanced voice call page
http://localhost:8000/enhanced-voice-call?call_id=test&language=en
```

### 2. API Endpoints

#### Create Call
```http
POST /api/voice-call/initiate
Content-Type: application/json
Authorization: Bearer <token>

{
    "target_user_id": 123
}
```

#### Join Call WebSocket
```javascript
ws://localhost:8000/api/voice-call/ws/{call_id}?token=<jwt_token>&language=en
```

#### Test Translation Service
```http
GET /api/voice-call/translation/test
```

### 3. Frontend Integration

```javascript
// Initialize enhanced voice call
const voiceCall = new EnhancedVoiceCall();

await voiceCall.init({
    callId: 'your-call-id',
    userLanguage: 'ar', // ar, en, or fr
    isInitiator: false
});
```

## 📡 WebSocket Protocol

### Client to Server Messages

```javascript
// Voice chunk for translation
{
    "type": "voice_chunk",
    "audio_data": "base64_encoded_audio",
    "language": "ar",
    "chunk_size": 32000,
    "timestamp": 1694558400000
}

// Translation settings update
{
    "type": "translation_settings",
    "settings": {
        "language": "fr",
        "enabled": true
    }
}

// WebRTC signaling
{
    "type": "webrtc_offer",
    "sdp": "...",
    "call_id": "call-123"
}
```

### Server to Client Messages

```javascript
// Translated voice
{
    "type": "voice_translation",
    "from_user": 456,
    "audio_data": "base64_encoded_translated_audio",
    "source_language": "ar",
    "target_language": "en",
    "timestamp": 1694558405000
}

// Connection confirmation
{
    "type": "connected",
    "call_id": "call-123",
    "translation_available": true,
    "supported_languages": {
        "ar": "Arabic",
        "en": "English", 
        "fr": "French"
    }
}
```

## ⚙️ Configuration

### Translation Settings

```python
# In seamless_translation_service.py
class SeamlessTranslationService:
    def __init__(self):
        self.sample_rate = 16000  # Required by SeamlessM4T
        self.chunk_duration = 2.0  # 2 seconds
        self.chunk_size = 32000    # ~2 seconds of audio
        
        # Supported languages (easily extensible)
        self.supported_languages = {
            'ar': 'arb',  # Arabic
            'en': 'eng',  # English
            'fr': 'fra',  # French
        }
```

### Audio Processing

```javascript
// In enhanced_voice_call.js
this.translationSettings = {
    enabled: true,
    chunkDuration: 2000, // 2 seconds
    overlapDuration: 500, // 0.5 second overlap
    minChunkSize: 16000   // Minimum audio size
};
```

## 🎮 GPU Support

SeamlessM4T performs significantly better with GPU support:

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Performance comparison:
# CPU: ~5-8 seconds translation latency
# GPU: ~2-3 seconds translation latency
```

## 🐛 Troubleshooting

### Common Issues

#### 1. SeamlessM4T Import Error
```bash
# Error: No module named 'seamless_communication'
pip install seamless_communication

# If still fails, try:
pip install --upgrade pip
pip install seamless_communication --no-cache-dir
```

#### 2. Audio Format Issues
```javascript
// Browser compatibility check
if (!MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
    // Fallback to default format
    this.mediaRecorder = new MediaRecorder(stream);
}
```

#### 3. Model Download Issues
```python
# Manual model download location:
# ~/.cache/torch/hub/checkpoints/
# Models: seamlessM4T_v2_large.pt (~2GB)

# Clear cache if corrupted:
import torch
torch.hub._get_cache_dir()  # Shows cache location
# Delete and re-download
```

#### 4. WebSocket Connection Issues
```javascript
// Check token validity
const token = localStorage.getItem('access_token');
if (!token) {
    window.location.href = '/login';
}

// WebSocket URL format
const wsUrl = `ws://localhost:8000/api/voice-call/ws/${callId}?token=${token}&language=en`;
```

## 📊 Performance Metrics

### Translation Latency
- **End-to-end latency**: 2-4 seconds
- **Audio chunk processing**: 2 seconds
- **Model inference**: 1-2 seconds
- **Network transmission**: <1 second

### Resource Usage
- **CPU**: High during translation (can use 50-80% on single core)
- **Memory**: ~4-6GB for loaded models
- **GPU VRAM**: ~2-4GB if using GPU acceleration
- **Network**: ~16-32 kbps per audio stream

### Supported Audio Formats
- **Input**: WebM, MP3, WAV
- **Processing**: 16kHz mono WAV (required by SeamlessM4T)
- **Output**: WAV (base64 encoded)

## 🔒 Security Considerations

1. **Authentication**: All WebSocket connections require valid JWT tokens
2. **Audio Data**: Processed in-memory, not permanently stored
3. **Rate Limiting**: Consider implementing per-user translation limits
4. **Content Filtering**: Add inappropriate content detection if needed

## 🎛️ Advanced Configuration

### Custom Language Support

To add more languages, update the language mappings:

```python
# In seamless_translation_service.py
self.supported_languages = {
    'ar': 'arb',  # Arabic
    'en': 'eng',  # English
    'fr': 'fra',  # French
    'es': 'spa',  # Spanish - ADD NEW LANGUAGES
    'de': 'deu',  # German
    # See SeamlessM4T docs for full language list
}
```

### Translation Quality Tuning

```python
# Adjust chunk processing for quality vs latency
self.chunk_duration = 3.0  # Longer chunks = better quality, higher latency
self.overlap_duration = 1.0  # More overlap = smoother transitions
```

## 📚 API Reference

### SeamlessTranslationService

```python
async def process_voice_chunk_realtime(
    call_id: str,
    user_id: int, 
    audio_data: str,  # base64 encoded
    source_language: str,  # 'ar', 'en', 'fr'
    target_language: str
) -> Dict
```

### VoiceCallManager

```python
async def handle_voice_translation(
    call_id: str,
    user_id: int,
    message: dict
)

async def join_call(
    call_id: str,
    user_id: int,
    websocket: WebSocket,
    language: str = "en"
) -> bool
```

## 🔗 Related Documentation

- [SeamlessM4T Official Docs](https://github.com/facebookresearch/seamless_communication)
- [WebRTC API Reference](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
- [FastAPI WebSocket Guide](https://fastapi.tiangolo.com/advanced/websockets/)

## 📄 License

This enhanced voice call feature is part of the Multilingual Translation Application. Please refer to the main project license for usage terms.
