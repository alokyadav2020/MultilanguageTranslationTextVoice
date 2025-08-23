#!/usr/bin/env python3
from app.services.voice_service import VoiceService

try:
    service = VoiceService()
    print('✅ Enhanced Voice Service loaded successfully')
    print(f'   Whisper model: {"Loaded" if service.whisper_model else "Not available"}')
    print(f'   Device: {service.whisper_device}')
    print(f'   Thread pool: {service.executor._max_workers} workers')
    print('🎉 Voice service is ready for enhanced processing!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
