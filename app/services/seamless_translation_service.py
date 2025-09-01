import torch
import torchaudio
import base64
import asyncio
import logging
from typing import Dict, Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class SeamlessTranslationService:
    def __init__(self):
        self.translator = None
        self.streaming_translator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000  # SeamlessM4T expects 16kHz
        self.chunk_duration = 2.0  # 2 seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Audio buffers for each user/call
        self.audio_buffers = {}  # call_id_user_id -> audio_buffer
        
        # Supported languages (only Arabic, English, French)
        self.supported_languages = {
            'ar': 'arb',  # Arabic
            'en': 'eng',  # English
            'fr': 'fra',  # French
        }
        
        # Language names for display
        self.language_names = {
            'ar': 'Arabic',
            'en': 'English', 
            'fr': 'French'
        }
        
        # Initialize models in background
        self._model_loaded = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize SeamlessM4T models with download progress"""
        try:
            logger.info("🔄 Initializing SeamlessM4T models...")
            
            # Check if models are already cached
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints" 
            seamless_files = list(cache_dir.glob("*seamless*")) if cache_dir.exists() else []
            
            if seamless_files:
                logger.info(f"✅ Found cached models: {len(seamless_files)} files")
            else:
                logger.info("📥 Models not cached. Starting download (~4-6 GB)...")
                logger.info("⏳ This may take 10-30 minutes depending on your internet speed...")
            
            # Try to import seamless_communication
            try:
                from seamless_communication.models.inference import Translator
                logger.info("✅ SeamlessM4T package found, loading models...")
                
                # Load the translator using the official API format
                # Using seamlessM4T_large for better quality (can change to seamlessM4T_v2_large if available)
                self.translator = Translator("seamlessM4T_large")
                
                # Try to also load v2_large if available
                try:
                    self.translator_v2 = Translator("seamlessM4T_v2_large")
                    logger.info("✅ SeamlessM4T v2 large model loaded")
                except Exception:
                    logger.info("ℹ️ Using seamlessM4T_large model")
                    self.translator_v2 = None
                
                self._model_loaded = True
                logger.info(f"✅ SeamlessM4T models loaded successfully on {self.device}")
                
                # Log cache information
                if cache_dir.exists():
                    total_size = sum(f.stat().st_size for f in cache_dir.glob("*")) / (1024**3)
                    logger.info(f"💾 Total cache size: {total_size:.1f} GB")
                
            except ImportError:
                logger.warning("⚠️ SeamlessM4T not installed. Please install: pip install seamless_communication")
                self._model_loaded = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize SeamlessM4T: {e}")
            self._model_loaded = False
    
    def is_available(self) -> bool:
        """Check if SeamlessM4T is available and loaded"""
        return self._model_loaded and self.translator is not None
    
    async def process_voice_chunk_realtime(
        self,
        call_id: str,
        user_id: int,
        audio_data: str,  # base64 encoded WebM audio
        source_language: str,
        target_language: str
    ) -> Dict:
        """Process voice chunk for real-time voice-to-voice translation"""
        
        if not self.is_available():
            return {
                "success": False, 
                "error": "SeamlessM4T not available. Please install seamless_communication package."
            }
        
        # Validate languages
        if source_language not in self.supported_languages:
            return {"success": False, "error": f"Source language '{source_language}' not supported"}
        
        if target_language not in self.supported_languages:
            return {"success": False, "error": f"Target language '{target_language}' not supported"}
        
        # Skip if same language
        if source_language == target_language:
            return {"success": True, "status": "same_language", "audio_output": audio_data}
        
        try:
            # Decode and prepare audio
            audio_tensor = await self._prepare_audio_chunk(audio_data)
            if audio_tensor is None:
                return {"success": False, "error": "Failed to process audio data"}
            
            # Add to user's audio buffer
            buffer_key = f"{call_id}_{user_id}"
            if buffer_key not in self.audio_buffers:
                self.audio_buffers[buffer_key] = torch.empty(0, device=self.device)
            
            # Concatenate new audio chunk
            self.audio_buffers[buffer_key] = torch.cat([
                self.audio_buffers[buffer_key], 
                audio_tensor
            ])
            
            # Process if we have enough audio (2+ seconds)
            if len(self.audio_buffers[buffer_key]) >= self.chunk_size:
                return await self._translate_voice_buffer(
                    buffer_key, source_language, target_language
                )
            
            return {"success": True, "status": "buffering"}
            
        except Exception as e:
            logger.error(f"Real-time voice translation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _prepare_audio_chunk(self, audio_data: str) -> Optional[torch.Tensor]:
        """Convert base64 WebM audio to tensor format for SeamlessM4T"""
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary file for audio conversion
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Load audio using torchaudio
                waveform, original_sr = torchaudio.load(temp_path)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed (SeamlessM4T requirement)
                if original_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=original_sr, 
                        new_freq=self.sample_rate
                    )
                    waveform = resampler(waveform)
                
                # Remove batch dimension and move to device
                audio_tensor = waveform.squeeze(0).to(self.device)
                
                return audio_tensor
                
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio preparation error: {e}")
            return None
    
    async def _translate_voice_buffer(
        self,
        buffer_key: str,
        source_language: str,
        target_language: str
    ) -> Dict:
        """Translate accumulated audio buffer using SeamlessM4T voice-to-voice"""
        try:
            # Get audio buffer
            audio_tensor = self.audio_buffers[buffer_key]
            
            # Take first chunk_size samples
            chunk_tensor = audio_tensor[:self.chunk_size]
            
            # Keep remaining audio in buffer
            self.audio_buffers[buffer_key] = audio_tensor[self.chunk_size:]
            
            # Convert language codes to SeamlessM4T format
            source_lang = self.supported_languages[source_language]
            target_lang = self.supported_languages[target_language]
            
            # Perform voice-to-voice translation
            translated_audio = await self._translate_voice_to_voice(
                chunk_tensor, source_lang, target_lang
            )
            
            if translated_audio is not None:
                # Convert audio tensor to base64 for transmission
                audio_base64 = await self._tensor_to_base64_audio(translated_audio)
                
                return {
                    "success": True,
                    "audio_output": audio_base64,
                    "source_language": source_language,
                    "target_language": target_language,
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                return {"success": False, "error": "Translation failed"}
            
        except Exception as e:
            logger.error(f"Voice buffer translation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _translate_voice_to_voice(
        self,
        audio_tensor: torch.Tensor,
        source_lang: str,
        target_lang: str
    ) -> Optional[torch.Tensor]:
        """Translate voice to voice using SeamlessM4T"""
        try:
            # Run translation in thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def translate():
                with torch.no_grad():
                    # Use the official SeamlessM4T API format
                    # translator.predict returns (text_out, audio_out)
                    text_out, audio_out = self.translator.predict(
                        audio_tensor,  # Input audio tensor
                        src_lang=source_lang,
                        tgt_lang=target_lang
                    )
                    return text_out, audio_out
            
            text_result, audio_result = await loop.run_in_executor(None, translate)
            
            # Return the audio output
            if audio_result is not None:
                return audio_result
            
            return None
            
        except Exception as e:
            logger.error(f"Voice-to-voice translation error: {e}")
            return None
    
    async def _tensor_to_base64_audio(self, audio_tensor: torch.Tensor) -> str:
        """Convert audio tensor to base64 encoded audio"""
        try:
            # Create temporary file for audio encoding
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save as WAV file
                torchaudio.save(
                    temp_path,
                    audio_tensor.unsqueeze(0),  # Add channel dimension
                    self.sample_rate,
                    format="wav"
                )
                
                # Read and encode as base64
                with open(temp_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                
                return audio_base64
                
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio encoding error: {e}")
            return ""
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their display names"""
        return self.language_names
    
    def cleanup_call_buffers(self, call_id: str):
        """Clean up audio buffers when call ends"""
        keys_to_remove = [key for key in self.audio_buffers.keys() if key.startswith(f"{call_id}_")]
        for key in keys_to_remove:
            del self.audio_buffers[key]
        logger.info(f"Cleaned up buffers for call {call_id}")
    
    def get_translation_stats(self) -> Dict:
        """Get translation service statistics"""
        return {
            "model_loaded": self._model_loaded,
            "device": str(self.device),
            "supported_languages": self.language_names,
            "active_buffers": len(self.audio_buffers),
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration
        }
    
    def check_model_status(self) -> Dict:
        """Check if models are downloaded and ready"""
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        
        status = {
            "models_available": self._model_loaded,
            "cache_directory": str(cache_dir),
            "cache_exists": cache_dir.exists(),
            "seamless_files": [],
            "total_cache_size_gb": 0
        }
        
        if cache_dir.exists():
            seamless_files = list(cache_dir.glob("*seamless*"))
            status["seamless_files"] = [f.name for f in seamless_files]
            
            total_size = sum(f.stat().st_size for f in cache_dir.glob("*")) / (1024**3)
            status["total_cache_size_gb"] = round(total_size, 2)
        
        return status
    
    async def test_translation(self) -> Dict:
        """Test translation with dummy audio"""
        if not self.is_available():
            return {
                "success": False,
                "error": "SeamlessM4T not available"
            }
        
        try:
            # Create dummy audio (1 second of silence at 16kHz)
            dummy_audio = torch.zeros(16000).to(self.device)
            
            # Test English to Arabic translation
            result = await self._translate_voice_to_voice(dummy_audio, "eng", "arb")
            
            return {
                "success": True,
                "message": "Translation test successful",
                "device": str(self.device),
                "model_loaded": self._model_loaded,
                "output_shape": result.shape if result is not None else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation test failed: {e}"
            }

# Global service instance
seamless_translation_service = SeamlessTranslationService()
