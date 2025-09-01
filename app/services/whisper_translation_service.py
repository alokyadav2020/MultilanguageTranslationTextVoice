"""
Whisper + Google Translate Service for Real-Time Voice Translation
Optimized for concurrent, non-blocking, async processing of voice chunks
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

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Audio chunk data structure"""
    data: np.ndarray
    timestamp: float
    user_id: int
    call_id: str
    source_lang: str
    target_lang: str

@dataclass
class TranslationResult:
    """Translation result data structure"""
    success: bool
    original_text: str = ""
    translated_text: str = ""
    translated_audio_base64: str = ""
    processing_time: float = 0.0
    error: str = ""
    timestamp: float = 0.0

class AsyncAudioBuffer:
    """Thread-safe audio buffer for accumulating chunks"""
    
    def __init__(self, buffer_duration: float = 2.0, sample_rate: int = 16000):
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.max_samples = int(buffer_duration * sample_rate)
        self.overlap_samples = int(0.5 * sample_rate)  # 0.5 second overlap
        
        self._buffer = deque(maxlen=self.max_samples * 2)  # Allow for overlap
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
    
    def add_audio(self, audio_data: np.ndarray) -> bool:
        """Add audio data to buffer, return True if ready for processing"""
        with self._lock:
            self._buffer.extend(audio_data)
            
            if len(self._buffer) >= self.max_samples:
                self._ready_event.set()
                return True
        return False
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """Get audio chunk for processing"""
        with self._lock:
            if len(self._buffer) >= self.max_samples:
                # Extract chunk with overlap handling
                chunk = np.array(list(self._buffer)[:self.max_samples])
                
                # Remove processed samples but keep overlap
                for _ in range(self.max_samples - self.overlap_samples):
                    if self._buffer:
                        self._buffer.popleft()
                
                # Reset event if buffer is too small now
                if len(self._buffer) < self.max_samples:
                    self._ready_event.clear()
                
                return chunk
        return None
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._buffer.clear()
            self._ready_event.clear()

class WhisperTranslationService:
    """
    High-performance real-time voice translation service using:
    - OpenAI Whisper for speech-to-text
    - Google Translate for text translation
    - gTTS for text-to-speech
    """
    
    def __init__(self):
        """Initialize the translation service with concurrent processing"""
        
        # Service availability
        self.is_available = False
        self._initialization_lock = threading.Lock()
        
        # Audio settings optimized for real-time processing
        self.sample_rate = 16000  # Standard for speech processing
        self.chunk_duration = 2.0  # 2 seconds for good balance
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Language mappings
        self.whisper_languages = {
            'ar': 'arabic',
            'en': 'english', 
            'fr': 'french'
        }
        
        self.translate_languages = {
            'ar': 'ar',
            'en': 'en',
            'fr': 'fr'
        }
        
        self.gtts_languages = {
            'ar': 'ar',
            'en': 'en',
            'fr': 'fr'
        }
        
        # Audio buffers for each call/user
        self.audio_buffers: Dict[str, AsyncAudioBuffer] = {}
        self.buffer_lock = threading.Lock()
        
        # Thread pools for concurrent processing
        self.whisper_executor = None
        self.translation_executor = None
        self.tts_executor = None
        
        # Models (loaded lazily)
        self.whisper_model = None
        self.translator = None
        
        # Processing stats
        self.stats = {
            'total_chunks': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'avg_processing_time': 0.0
        }
        
        # Initialize service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the translation service components"""
        try:
            logger.info("🔄 Initializing Whisper Translation Service...")
            
            # Initialize thread pools for concurrent processing
            self.whisper_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="whisper"
            )
            
            self.translation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=3,
                thread_name_prefix="translate" 
            )
            
            self.tts_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="tts"
            )
            
            # Initialize Google Translator
            self.translator = Translator()
            
            logger.info("✅ Thread pools initialized")
            
            # Test translator
            test_result = self.translator.translate("test", dest='fr')
            if test_result and test_result.text:
                logger.info("✅ Google Translate connection verified")
            else:
                raise Exception("Google Translate test failed")
            
            self.is_available = True
            logger.info("✅ Whisper Translation Service ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper Translation Service: {e}")
            self.is_available = False
    
    def _load_whisper_model(self) -> bool:
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            try:
                logger.info("📥 Loading Whisper model (base)...")
                start_time = time.time()
                
                # Use base model for faster processing (can upgrade to 'small' or 'medium')
                self.whisper_model = whisper.load_model("base")
                
                load_time = time.time() - start_time
                logger.info(f"✅ Whisper model loaded in {load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                return False
        return True
    
    async def process_voice_chunk_realtime(
        self,
        call_id: str,
        user_id: int,
        audio_data: str,
        source_language: str,
        target_language: str
    ) -> Dict:
        """
        Main entry point for real-time voice chunk processing
        Optimized for non-blocking, concurrent execution
        """
        
        if not self.is_available:
            return {
                "success": False,
                "error": "Whisper Translation Service not available"
            }
        
        if source_language == target_language:
            return {
                "success": False, 
                "error": "Source and target languages are the same"
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Decode and prepare audio (fast, CPU-bound)
            audio_array = await self._decode_audio_chunk(audio_data)
            if audio_array is None:
                return {
                    "success": False,
                    "error": "Failed to decode audio data"
                }
            
            # Step 2: Add to buffer and check if ready for processing
            buffer_key = f"{call_id}_{user_id}"
            ready_for_processing = self._add_to_buffer(buffer_key, audio_array)
            
            if not ready_for_processing:
                return {
                    "success": True,
                    "status": "buffering",
                    "buffer_progress": self._get_buffer_progress(buffer_key)
                }
            
            # Step 3: Process accumulated buffer asynchronously
            chunk_data = self._get_audio_chunk(buffer_key)
            if chunk_data is None:
                return {
                    "success": False,
                    "error": "Failed to get audio chunk from buffer"
                }
            
            # Step 4: Perform translation pipeline concurrently
            result = await self._translate_audio_chunk(
                chunk_data, source_language, target_language
            )
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(result.success, processing_time)
            
            if result.success:
                return {
                    "success": True,
                    "original_text": result.original_text,
                    "translated_text": result.translated_text,
                    "translated_audio": result.translated_audio_base64,
                    "processing_time": result.processing_time,
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "error": result.error,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            logger.error(f"Voice chunk processing error: {e}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }
    
    async def _decode_audio_chunk(self, audio_data: str) -> Optional[np.ndarray]:
        """Decode base64 WebM audio to numpy array (async, non-blocking)"""
        
        def decode_sync():
            try:
                # Decode base64 
                audio_bytes = base64.b64decode(audio_data)
                
                # Create temporary file for audio conversion
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name
                
                try:
                    # Load and resample audio using librosa
                    audio, sr = librosa.load(
                        temp_path, 
                        sr=self.sample_rate, 
                        mono=True,
                        dtype=np.float32
                    )
                    
                    # Normalize audio
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio)) * 0.9
                    
                    return audio
                    
                finally:
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.error(f"Audio decoding error: {e}")
                return None
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.translation_executor, 
            decode_sync
        )
        
        return result
    
    def _add_to_buffer(self, buffer_key: str, audio_data: np.ndarray) -> bool:
        """Add audio to buffer, return True if ready for processing"""
        with self.buffer_lock:
            if buffer_key not in self.audio_buffers:
                self.audio_buffers[buffer_key] = AsyncAudioBuffer(
                    buffer_duration=self.chunk_duration,
                    sample_rate=self.sample_rate
                )
            
            return self.audio_buffers[buffer_key].add_audio(audio_data)
    
    def _get_audio_chunk(self, buffer_key: str) -> Optional[np.ndarray]:
        """Get audio chunk from buffer"""
        with self.buffer_lock:
            if buffer_key in self.audio_buffers:
                return self.audio_buffers[buffer_key].get_chunk()
        return None
    
    def _get_buffer_progress(self, buffer_key: str) -> float:
        """Get buffer fill progress (0.0 to 1.0)"""
        with self.buffer_lock:
            if buffer_key in self.audio_buffers:
                buffer = self.audio_buffers[buffer_key]
                with buffer._lock:
                    return min(len(buffer._buffer) / buffer.max_samples, 1.0)
        return 0.0
    
    async def _translate_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """
        Translate audio chunk through the full pipeline:
        Audio → Text → Translation → Speech
        All steps run concurrently where possible
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Speech-to-Text using Whisper (CPU intensive)
            original_text = await self._transcribe_audio(audio_chunk, source_lang)
            
            if not original_text or len(original_text.strip()) < 2:
                return TranslationResult(
                    success=False,
                    error="No speech detected or text too short",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Translate text (network I/O)
            translated_text = await self._translate_text(
                original_text, source_lang, target_lang
            )
            
            if not translated_text:
                return TranslationResult(
                    success=False,
                    error="Text translation failed",
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Text-to-Speech (network I/O)
            translated_audio = await self._generate_speech(
                translated_text, target_lang
            )
            
            if not translated_audio:
                return TranslationResult(
                    success=False,
                    error="Speech generation failed", 
                    processing_time=time.time() - start_time
                )
            
            return TranslationResult(
                success=True,
                original_text=original_text,
                translated_text=translated_text,
                translated_audio_base64=translated_audio,
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Translation pipeline error: {e}")
            return TranslationResult(
                success=False,
                error=f"Pipeline error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    async def _transcribe_audio(self, audio_chunk: np.ndarray, language: str) -> Optional[str]:
        """Transcribe audio using Whisper (runs in thread pool)"""
        
        def transcribe_sync():
            try:
                # Ensure Whisper model is loaded
                if not self._load_whisper_model():
                    return None
                
                # Create temporary audio file for Whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Save audio as WAV file
                    sf.write(temp_path, audio_chunk, self.sample_rate)
                    
                    # Transcribe using Whisper
                    result = self.whisper_model.transcribe(
                        temp_path,
                        language=self.whisper_languages.get(language, 'english'),
                        fp16=False,  # Use fp32 for better compatibility
                        task='transcribe'
                    )
                    
                    text = result["text"].strip()
                    logger.debug(f"Whisper transcription: '{text}'")
                    
                    return text if text else None
                    
                finally:
                    Path(temp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
                return None
        
        # Run in Whisper thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.whisper_executor,
            transcribe_sync
        )
        
        return result
    
    async def _translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> Optional[str]:
        """Translate text using Google Translate (runs in thread pool)"""
        
        def translate_sync():
            try:
                # Prepare language codes
                src_code = self.translate_languages.get(source_lang, 'en')
                dest_code = self.translate_languages.get(target_lang, 'en')
                
                # Perform translation
                result = self.translator.translate(
                    text,
                    src=src_code,
                    dest=dest_code
                )
                
                translated = result.text.strip()
                logger.debug(f"Translation: '{text}' → '{translated}'")
                
                return translated if translated else None
                
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
