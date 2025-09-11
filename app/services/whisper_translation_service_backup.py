"""
⚠️ DISABLED SERVICE - REPLACED BY SINGLETON voice_service.py ⚠️

This service has been disabled to prevent multiple Whisper model loading.
Use voice_service.py instead which implements singleton pattern with turbo model.

Original: Whisper + Google Translate Service for Real-Time Voice Translation
Optimized for concurrent, non-blocking, async processing of voice chunks
"""

# DISABLED: Commented out to prevent multiple Whisper model loading
# Only voice_service.py with singleton pattern should be used

"""
import whisper
import base64
import tempfile
import asyncio
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import threading

# Import translation libraries
from googletrans import Translator
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
import io
"""
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# DISABLED CLASSES - Replaced with dummy implementations

class AudioChunk:
    """DISABLED: Audio chunk data structure - Use voice_service.py instead"""
    def __init__(self, *args, **kwargs):
        logger.warning("⚠️ AudioChunk is DISABLED. Use voice_service.py instead.")

class TranslationResult:
    """DISABLED: Translation result data structure - Use voice_service.py instead"""
    def __init__(self, *args, **kwargs):
        logger.warning("⚠️ TranslationResult is DISABLED. Use voice_service.py instead.")

class AsyncAudioBuffer:
    """DISABLED: Thread-safe audio buffer - Use voice_service.py instead"""
    def __init__(self, *args, **kwargs):
        logger.warning("⚠️ AsyncAudioBuffer is DISABLED. Use voice_service.py instead.")
    
    def add_audio(self, audio_data):
        """DISABLED METHOD"""
        logger.warning("⚠️ add_audio is DISABLED. Use voice_service.py instead.")
        return False
    
    def get_chunk(self):
        """DISABLED METHOD"""
        logger.warning("⚠️ get_chunk is DISABLED. Use voice_service.py instead.")
        return None
    
    def clear(self):
        """DISABLED METHOD"""
        logger.warning("⚠️ clear is DISABLED. Use voice_service.py instead.")

class WhisperTranslationService:
    """
    DISABLED SERVICE - Use voice_service.py instead
    
    This service has been disabled to prevent multiple Whisper model loading.
    All methods return disabled status or redirect to voice_service.py
    """
    
    def __init__(self):
        logger.warning("⚠️ WhisperTranslationService is DISABLED. Use voice_service.py instead.")
        self.is_available = False
        self.whisper_model = None
    
    def _initialize_service(self):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _initialize_service is DISABLED. Use voice_service.py instead.")
        return False
    
    def _load_whisper_model(self) -> bool:
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _load_whisper_model is DISABLED. Use voice_service.py instead.")
        return False
    
    async def process_voice_chunk_realtime(
        self,
        call_id: str,
        user_id: int,
        audio_data: str,
        source_language: str,
        target_language: str
    ) -> Dict:
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ process_voice_chunk_realtime is DISABLED. Use voice_service.py instead.")
        return {
            "success": False,
            "error": "Service disabled. Use voice_service.py instead.",
            "status": "disabled"
        }
    
    async def _decode_audio_chunk(self, audio_data: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _decode_audio_chunk is DISABLED. Use voice_service.py instead.")
        return None
    
    def _add_to_buffer(self, buffer_key: str, audio_data) -> bool:
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _add_to_buffer is DISABLED. Use voice_service.py instead.")
        return False
    
    def _get_audio_chunk(self, buffer_key: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _get_audio_chunk is DISABLED. Use voice_service.py instead.")
        return None
    
    def _get_buffer_progress(self, buffer_key: str) -> float:
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _get_buffer_progress is DISABLED. Use voice_service.py instead.")
        return 0.0
    
    async def _translate_audio_chunk(self, audio_chunk, source_lang: str, target_lang: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _translate_audio_chunk is DISABLED. Use voice_service.py instead.")
        return TranslationResult(success=False, error="Service disabled")
    
    async def _transcribe_audio(self, audio_chunk, language: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _transcribe_audio is DISABLED. Use voice_service.py instead.")
        return None
    
    async def _translate_text(self, text: str, source_lang: str, target_lang: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _translate_text is DISABLED. Use voice_service.py instead.")
        return None
    
    async def _generate_speech(self, text: str, language: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _generate_speech is DISABLED. Use voice_service.py instead.")
        return None
    
    def _update_stats(self, success: bool, processing_time: float):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ _update_stats is DISABLED. Use voice_service.py instead.")
    
    def cleanup_call_buffers(self, call_id: str):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ cleanup_call_buffers is DISABLED. Use voice_service.py instead.")
    
    def get_service_status(self) -> Dict:
        """Return disabled status"""
        return {
            "status": "disabled",
            "message": "Service disabled. Use voice_service.py instead.",
            "is_available": False,
            "whisper_model_loaded": False,
            "replacement_service": "voice_service.py"
        }
    
    def shutdown(self):
        """DISABLED: Use voice_service.py instead"""
        logger.warning("⚠️ shutdown is DISABLED. Use voice_service.py instead.")

# Global service instance (returns disabled service)
whisper_translation_service = WhisperTranslationService()
                
            except Exception as e:
                logger.error(f"Google Translate error: {e}")
                return None
        
        # Run in translation thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.translation_executor,
            translate_sync
        )
        
        return result
    
    async def _generate_speech(self, text: str, language: str) -> Optional[str]:
        """Generate speech using gTTS (runs in thread pool)"""
        
        def tts_sync():
            try:
                # Get language code for gTTS
                lang_code = self.gtts_languages.get(language, 'en')
                
                # Create gTTS object
                tts = gTTS(
                    text=text,
                    lang=lang_code,
                    slow=False,
                    tld='com'  # Use .com for more natural voice
                )
                
                # Generate audio in memory
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Convert to base64
                audio_bytes = audio_buffer.getvalue()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                logger.debug(f"Generated TTS audio: {len(audio_bytes)} bytes")
                
                return audio_base64
                
            except Exception as e:
                logger.error(f"gTTS generation error: {e}")
                return None
        
        # Run in TTS thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.tts_executor,
            tts_sync
        )
        
        return result
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        self.stats['total_chunks'] += 1
        
        if success:
            self.stats['successful_translations'] += 1
        else:
            self.stats['failed_translations'] += 1
        
        # Update running average
        total = self.stats['total_chunks']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
    
    def cleanup_call_buffers(self, call_id: str):
        """Clean up buffers and resources for a call"""
        with self.buffer_lock:
            keys_to_remove = [
                key for key in self.audio_buffers.keys() 
                if key.startswith(f"{call_id}_")
            ]
            
            for key in keys_to_remove:
                self.audio_buffers[key].clear()
                del self.audio_buffers[key]
        
        logger.info(f"Cleaned up buffers for call {call_id}")
    
    def get_service_status(self) -> Dict:
        """Get service status and statistics"""
        return {
            "service_name": "WhisperTranslationService",
            "is_available": self.is_available,
            "model_loaded": self.whisper_model is not None,
            "active_buffers": len(self.audio_buffers),
            "statistics": self.stats.copy(),
            "supported_languages": list(self.whisper_languages.keys()),
            "thread_pools": {
                "whisper": self.whisper_executor._threads if self.whisper_executor else 0,
                "translation": self.translation_executor._threads if self.translation_executor else 0,
                "tts": self.tts_executor._threads if self.tts_executor else 0
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the service"""
        logger.info("🔄 Shutting down Whisper Translation Service...")
        
        # Clear all buffers
        with self.buffer_lock:
            for buffer in self.audio_buffers.values():
                buffer.clear()
            self.audio_buffers.clear()
        
        # Shutdown thread pools
        if self.whisper_executor:
            self.whisper_executor.shutdown(wait=True)
        if self.translation_executor:
            self.translation_executor.shutdown(wait=True)
        if self.tts_executor:
            self.tts_executor.shutdown(wait=True)
        
        logger.info("✅ Whisper Translation Service shutdown complete")

# Global service instance
whisper_translation_service = WhisperTranslationService()
