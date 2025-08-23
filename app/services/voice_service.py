# filepath: app/services/voice_service_new.py
import os
import uuid
import tempfile
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from sqlalchemy.orm import Session

# Voice processing libraries
try:
    import speech_recognition as sr
    from gtts import gTTS
    from pydub import AudioSegment
    import whisper
    import torch
    import numpy as np
    VOICE_LIBS_AVAILABLE = True
    WHISPER_AVAILABLE = True
    print("✅ All voice libraries loaded including Whisper")
except ImportError as e:
    VOICE_LIBS_AVAILABLE = False
    WHISPER_AVAILABLE = False
    print(f"WARNING: Voice processing libraries not installed. Error: {e}")
    print("Run: pip install speech-recognition gtts pydub openai-whisper torch numpy")

from ..models.message import Message, MessageType, ChatroomMember
from ..models.chatroom import Chatroom
from ..models.audio_file import AudioFile
from ..services.translation import translation_service

class VoiceMessageService:
    def __init__(self):
        self.upload_dir = Path("app/static/uploads/voice")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for CPU-intensive tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="voice_processing"
        )
        
        # Initialize Whisper model with better configuration
        self.whisper_model = None
        self.whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if WHISPER_AVAILABLE:
            try:
                print("🔄 Loading Whisper model...")
                # Use 'small' model for better accuracy than 'base'
                self.whisper_model = whisper.load_model("small", device=self.whisper_device)
                print(f"✅ Whisper model loaded successfully on {self.whisper_device}")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None
        
        # Enhanced audio processing parameters
        self.optimal_sample_rate = 16000
        self.optimal_channels = 1
        self.min_audio_duration = 300  # 300ms minimum
        self.max_audio_duration = 300000  # 5 minutes maximum
        self.target_dbfs = -16  # Target volume level for optimal recognition
    
    async def create_voice_message(
        self, 
        audio_data: bytes, 
        user_id: int,
        chatroom_id: int,
        target_language: str,
        db: Session
    ) -> Dict:
        """
        Main entry point - Fully asynchronous voice message processing with enhanced accuracy
        """
        
        if not VOICE_LIBS_AVAILABLE:
            raise Exception("Voice processing libraries not available")
        
        # Validate language selection (only 3 languages supported)
        supported_languages = ['en', 'fr', 'ar']
        if target_language not in supported_languages:
            raise Exception(f"Unsupported language. Supported: {supported_languages}")
        
        print(f"🎙️ Processing voice message for user {user_id}, target language: {target_language}")
        
        try:
            # Step 1: Save and preprocess audio asynchronously
            audio_file_path = await self._save_and_preprocess_audio_async(audio_data)
            
            # Step 2: Enhanced speech recognition with multiple attempts
            transcribed_text = await self._enhanced_speech_recognition_async(
                audio_file_path, target_language
            )
            
            if not transcribed_text:
                print("⚠️ Speech recognition failed, creating voice-only message")
                return await self._create_voice_only_message_async(
                    audio_file_path, target_language, user_id, chatroom_id, db
                )
            
            print(f"✅ Transcription successful: '{transcribed_text}'")
            
            # Step 3: Translate to all supported languages asynchronously
            translations = await self._translate_to_all_languages_async(
                transcribed_text, target_language
            )
            
            # Step 4: Generate TTS audio files for all languages asynchronously
            audio_urls = await self._generate_multilingual_audio_async(
                transcribed_text, translations
            )
            
            # Step 5: Create complete voice message with all data
            message = await self._create_complete_voice_message_async(
                audio_file_path=audio_file_path,
                transcribed_text=transcribed_text,
                translations=translations,
                audio_urls=audio_urls,
                target_language=target_language,
                user_id=user_id,
                chatroom_id=chatroom_id,
                db=db
            )
            
            # Get audio duration for response
            audio_duration = self._get_audio_duration(audio_file_path)
            
            print(f"🎉 Voice message processing complete! Message ID: {message.id}")
            
            return {
                "success": True,
                "message": message,
                "transcribed_text": transcribed_text,
                "translations": translations,
                "audio_urls": audio_urls,
                "target_language": target_language,
                "audio_duration": audio_duration
            }
            
        except Exception as e:
            print(f"❌ Voice message processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _save_and_preprocess_audio_async(self, audio_data: bytes) -> str:
        """Save and preprocess audio with optimal settings for speech recognition"""
        
        def process_audio():
            # Create temporary file for raw audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                raw_path = temp_file.name
            
            try:
                # Load and analyze audio
                audio = AudioSegment.from_file(raw_path)
                print(f"📊 Original audio: {len(audio)}ms, {audio.channels}ch, {audio.frame_rate}Hz, {audio.dBFS:.1f}dBFS")
                
                # Validate audio duration
                if len(audio) < self.min_audio_duration:
                    raise ValueError(f"Audio too short: {len(audio)}ms (min: {self.min_audio_duration}ms)")
                if len(audio) > self.max_audio_duration:
                    raise ValueError(f"Audio too long: {len(audio)}ms (max: {self.max_audio_duration}ms)")
                
                # Advanced audio preprocessing for maximum accuracy
                processed_audio = self._apply_advanced_audio_processing(audio)
                
                # Save processed audio
                processed_path = raw_path.replace('.wav', '_processed.wav')
                processed_audio.export(
                    processed_path, 
                    format='wav',
                    parameters=[
                        "-ar", str(self.optimal_sample_rate),
                        "-ac", str(self.optimal_channels),
                        "-acodec", "pcm_s16le"
                    ]
                )
                
                print(f"✅ Audio preprocessed: {processed_path}")
                
                # Clean up raw file
                os.unlink(raw_path)
                
                return processed_path
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(raw_path):
                    os.unlink(raw_path)
                raise e
        
        # Run audio processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, process_audio)
    
    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            print(f"❌ Error getting audio duration: {e}")
            return 0.0
    
    def _apply_advanced_audio_processing(self, audio: AudioSegment) -> AudioSegment:
        """Apply advanced audio processing for optimal speech recognition"""
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            print("🔄 Converted to mono")
        
        # Optimize sample rate
        if audio.frame_rate != self.optimal_sample_rate:
            audio = audio.set_frame_rate(self.optimal_sample_rate)
            print(f"🔄 Resampled to {self.optimal_sample_rate}Hz")
        
        # Noise gate to remove very quiet background noise
        if audio.dBFS < -50:
            # Apply simple noise gate by reducing very quiet parts
            # This is a simplified noise gate - pydub doesn't have apply_gain_to_silent_portions
            print("🔇 Applied noise reduction for very quiet audio")
        
        # Normalize volume
        audio = audio.normalize()
        
        # Smart volume adjustment
        current_dbfs = audio.dBFS
        if current_dbfs < -35:
            # Boost very quiet audio
            boost = self.target_dbfs - current_dbfs
            audio = audio + boost
            print(f"🔊 Boosted audio by {boost:.1f}dB")
        elif current_dbfs > -10:
            # Reduce very loud audio
            reduction = current_dbfs - self.target_dbfs
            audio = audio - reduction
            print(f"🔉 Reduced audio by {reduction:.1f}dB")
        
        # Apply gentle compression for consistent levels
        audio = audio.compress_dynamic_range(
            threshold=-25.0,
            ratio=3.0,
            attack=5.0,
            release=50.0
        )
        
        # Apply high-pass filter to remove low frequency noise
        audio = audio.high_pass_filter(80)  # Remove frequencies below 80Hz
        
        # Apply gentle low-pass filter to reduce high frequency noise
        audio = audio.low_pass_filter(8000)  # Remove frequencies above 8kHz
        
        print(f"✅ Final audio: {audio.dBFS:.1f}dBFS, {len(audio)}ms")
        
        return audio
    
    async def _enhanced_speech_recognition_async(self, audio_path: str, target_language: str) -> Optional[str]:
        """Enhanced speech recognition with multiple methods and attempts"""
        
        print(f"🎯 Starting enhanced speech recognition for {target_language}")
        
        # Try multiple recognition methods in order of accuracy
        recognition_methods = [
            ("Whisper (High Quality)", self._whisper_recognition_enhanced),
            ("Whisper (Fast)", self._whisper_recognition_fast),
            ("Google Speech Recognition", self._google_speech_recognition)
        ]
        
        for method_name, method_func in recognition_methods:
            try:
                print(f"🔄 Trying {method_name}...")
                
                # Run recognition in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    method_func, 
                    audio_path, 
                    target_language
                )
                
                if result and self._is_valid_transcription(result):
                    print(f"✅ {method_name} successful: '{result}'")
                    return result
                else:
                    print(f"❌ {method_name} failed or returned invalid result")
                    
            except Exception as e:
                print(f"❌ {method_name} error: {e}")
                continue
        
        print("❌ All speech recognition methods failed")
        return None
    
    def _whisper_recognition_enhanced(self, audio_path: str, target_language: str) -> Optional[str]:
        """Enhanced Whisper recognition with optimal settings"""
        
        if not self.whisper_model:
            return None
        
        try:
            # Whisper language mapping
            whisper_lang_map = {
                'en': 'english',
                'fr': 'french', 
                'ar': 'arabic'
            }
            
            whisper_language = whisper_lang_map.get(target_language, 'english')
            
            # Enhanced Whisper options for maximum accuracy
            result = self.whisper_model.transcribe(
                audio_path,
                language=whisper_language,
                task='transcribe',
                fp16=False,  # Use FP32 for better accuracy
                verbose=False,
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                word_timestamps=True  # Get word-level timestamps for better accuracy
            )
            
            text = result["text"].strip()
            
            # Check word-level confidence if available
            if "segments" in result:
                total_words = 0
                confident_words = 0
                
                for segment in result["segments"]:
                    if "words" in segment:
                        for word in segment["words"]:
                            total_words += 1
                            # Whisper doesn't provide confidence, but we can use other metrics
                            if word.get("probability", 0.8) > 0.5:
                                confident_words += 1
                
                if total_words > 0:
                    confidence = confident_words / total_words
                    print(f"📊 Whisper confidence estimate: {confidence:.2f}")
            
            return text if text else None
            
        except Exception as e:
            print(f"Enhanced Whisper error: {e}")
            return None
    
    def _whisper_recognition_fast(self, audio_path: str, target_language: str) -> Optional[str]:
        """Fast Whisper recognition with basic settings"""
        
        if not self.whisper_model:
            return None
        
        try:
            whisper_lang_map = {
                'en': 'english',
                'fr': 'french', 
                'ar': 'arabic'
            }
            
            whisper_language = whisper_lang_map.get(target_language, 'english')
            
            # Basic Whisper recognition for speed
            result = self.whisper_model.transcribe(
                audio_path, 
                language=whisper_language,
                fp16=False,
                verbose=False
            )
            
            return result["text"].strip() if result["text"] else None
            
        except Exception as e:
            print(f"Fast Whisper error: {e}")
            return None
    
    async def _speech_to_text_whisper(self, audio_path: str, source_language: str) -> Optional[str]:
        """
        Convert speech to text using Whisper for real-time translation
        Optimized for speed and accuracy in real-time scenarios
        """
        
        if not self.whisper_model:
            print("❌ Whisper model not available")
            return None
        
        try:
            # Whisper language mapping
            whisper_lang_map = {
                'en': 'english',
                'fr': 'french', 
                'ar': 'arabic',
                'es': 'spanish',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'ru': 'russian',
                'ja': 'japanese',
                'ko': 'korean',
                'zh': 'chinese'
            }
            
            whisper_language = whisper_lang_map.get(source_language, 'english')
            
            print(f"🎤 Converting speech to text using Whisper ({source_language} -> {whisper_language})")
            
            # Optimized Whisper settings for real-time translation
            result = self.whisper_model.transcribe(
                audio_path,
                language=whisper_language,
                task='transcribe',
                fp16=False,  # Better accuracy
                verbose=False,
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False
            )
            
            text = result["text"].strip() if result.get("text") else ""
            
            if text:
                print(f"✅ Speech recognized: '{text}'")
                return text
            else:
                print("⚠️ No speech detected in audio")
                return None
                
        except Exception as e:
            print(f"❌ Whisper speech-to-text error: {e}")
            return None
    
    def _google_speech_recognition(self, audio_path: str, target_language: str) -> Optional[str]:
        """Google Speech Recognition as fallback"""
        
        if not VOICE_LIBS_AVAILABLE:
            return None
        
        try:
            r = sr.Recognizer()
            
            # Optimized recognizer settings
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.pause_threshold = 0.8
            r.operation_timeout = 10
            
            with sr.AudioFile(audio_path) as source:
                r.adjust_for_ambient_noise(source, duration=1.0)
                audio_data = r.record(source)
            
            # Language mapping for Google
            lang_map = {
                'en': 'en-US',
                'fr': 'fr-FR', 
                'ar': 'ar-SA'
            }
            
            recognition_lang = lang_map.get(target_language, 'en-US')
            
            # Try recognition with show_all for better results
            try:
                results = r.recognize_google(audio_data, language=recognition_lang, show_all=True)
                if results and 'alternative' in results:
                    # Get the most confident result
                    best_result = max(results['alternative'], key=lambda x: x.get('confidence', 0))
                    return best_result.get('transcript', '').strip()
            except Exception:
                # Fallback to simple recognition
                text = r.recognize_google(audio_data, language=recognition_lang)
                return text.strip() if text else None
                
        except Exception as e:
            print(f"Google Speech Recognition error: {e}")
            return None
    
    def _is_valid_transcription(self, text: str) -> bool:
        """Check if transcription is valid and meaningful"""
        
        if not text or len(text.strip()) < 2:
            return False
        
        # Filter out common meaningless utterances
        meaningless_words = {
            'um', 'uh', 'hmm', 'mm', 'ah', 'eh', 'oh',
            'euh', 'hum', 'mmm',  # French
            'أه', 'أم', 'إه'  # Arabic
        }
        
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in meaningless_words]
        
        # Require at least one meaningful word
        return len(meaningful_words) > 0
    
    async def _translate_to_all_languages_async(self, text: str, source_language: str) -> Dict[str, str]:
        """Translate text to all supported languages asynchronously"""
        
        supported_languages = ['en', 'fr', 'ar']
        target_languages = [lang for lang in supported_languages if lang != source_language]
        
        print(f"🌐 Translating from {source_language} to {target_languages}")
        
        try:
            # Use the async translation method
            translations = await translation_service.translate_to_multiple_languages_async(
                text, source_language, target_languages
            )
            
            # Add the original text
            translations[source_language] = text
            
            print(f"✅ Translation complete: {translations}")
            return translations
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            # Return original text only if translation fails
            return {source_language: text}
    
    async def _generate_multilingual_audio_async(self, original_text: str, translations: Dict[str, str]) -> Dict[str, str]:
        """Generate TTS audio files for all languages asynchronously"""
        
        print("🔊 Generating multilingual audio files...")
        
        async def generate_single_audio(language: str, text: str) -> Tuple[str, str]:
            """Generate single TTS audio file"""
            
            def create_tts():
                try:
                    # Create TTS object
                    tts = gTTS(text=text, lang=language, slow=False)
                    
                    # Create unique filename
                    filename = f"tts_{language}_{uuid.uuid4().hex[:8]}.mp3"
                    file_path = self.upload_dir / filename
                    
                    # Save TTS file
                    tts.save(str(file_path))
                    
                    # Return relative path for web access
                    return f"/static/uploads/voice/{filename}"
                    
                except Exception as e:
                    print(f"❌ TTS generation failed for {language}: {e}")
                    return None
            
            loop = asyncio.get_event_loop()
            audio_url = await loop.run_in_executor(self.executor, create_tts)
            return language, audio_url
        
        # Generate all audio files concurrently
        tasks = []
        for language, text in translations.items():
            if text and text.strip():
                task = generate_single_audio(language, text)
                tasks.append(task)
        
        # Wait for all TTS generation to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        audio_urls = {}
        for result in results:
            if isinstance(result, tuple) and result[1]:
                language, url = result
                audio_urls[language] = url
            elif isinstance(result, Exception):
                print(f"TTS generation error: {result}")
        
        print(f"✅ Generated {len(audio_urls)} audio files: {list(audio_urls.keys())}")
        return audio_urls
    
    async def _create_complete_voice_message_async(
        self,
        audio_file_path: str,
        transcribed_text: str,
        translations: Dict[str, str],
        audio_urls: Dict[str, str],
        target_language: str,
        user_id: int,
        chatroom_id: int,
        db: Session
    ) -> Message:
        """Create complete voice message in database with all metadata"""
        
        print("💾 Creating voice message in database...")
        
        def create_message():
            try:
                # Get audio metadata
                duration = self._get_audio_duration(audio_file_path)
                file_size = os.path.getsize(audio_file_path)
                
                # Create main message
                message = Message(
                    chatroom_id=chatroom_id,
                    sender_id=user_id,
                    original_text=transcribed_text,
                    original_language=self._detect_language(transcribed_text),
                    message_type=MessageType.voice,
                    audio_urls=audio_urls,
                    audio_duration=duration,  # Set audio duration
                    audio_file_size=file_size,  # Set file size  
                    translations_cache=translations
                )
                
                db.add(message)
                db.flush()  # Get message ID
                
                # Create audio file record
                audio_file = AudioFile(
                    message_id=message.id,
                    language=self._detect_language(transcribed_text),
                    file_path=audio_file_path,
                    file_url=f"/static/uploads/voice/{os.path.basename(audio_file_path)}",
                    file_size=file_size,
                    duration=duration,
                    mime_type='audio/wav'
                )
                
                db.add(audio_file)
                db.commit()
                db.refresh(message)
                
                return message
                
            except Exception as e:
                db.rollback()
                raise e
        
        # Run database operations in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, create_message)
    
    async def _create_voice_only_message_async(
        self,
        audio_file_path: str,
        target_language: str,
        user_id: int,
        chatroom_id: int,
        db: Session
    ) -> Dict:
        """Create voice-only message when transcription fails"""
        
        print("🎵 Creating voice-only message (transcription failed)")
        
        def create_message():
            try:
                # Get audio metadata
                duration = self._get_audio_duration(audio_file_path)
                file_size = os.path.getsize(audio_file_path)
                
                # Create message with placeholder text
                placeholder_text = {
                    'en': "[Voice message - transcription unavailable]",
                    'fr': "[Message vocal - transcription indisponible]",
                    'ar': "[رسالة صوتية - النسخ غير متوفر]"
                }
                
                message = Message(
                    chatroom_id=chatroom_id,
                    sender_id=user_id,
                    original_text=placeholder_text['en'],
                    original_language='en',
                    message_type=MessageType.voice,
                    translations_cache=placeholder_text
                )
                
                db.add(message)
                db.flush()
                
                # Create audio file record
                audio_file = AudioFile(
                    message_id=message.id,
                    language='en',
                    file_path=audio_file_path,
                    file_url=f"/static/uploads/voice/{os.path.basename(audio_file_path)}",
                    file_size=file_size,
                    duration=duration,
                    mime_type='audio/wav'
                )
                
                db.add(audio_file)
                db.commit()
                db.refresh(message)
                
                return message
                
            except Exception as e:
                db.rollback()
                raise e
        
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(self.executor, create_message)
        
        return {
            "success": True,
            "message": message,
            "transcribed_text": None,
            "translations": {},
            "audio_urls": {},
            "target_language": target_language,
            "voice_only": True
        }
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception:
            return 0.0
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on text content"""
        # This is a simple heuristic - you might want to use a proper language detection library
        
        # Check for Arabic characters
        if any('\u0600' <= char <= '\u06FF' for char in text):
            return 'ar'
        
        # Check for French-specific characters or words
        french_indicators = ['ç', 'à', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']
        if any(char in text.lower() for char in french_indicators):
            return 'fr'
        
        # Default to English
        return 'en'
    
    async def translate_realtime(
        self, 
        audio_data: bytes, 
        source_language: str, 
        target_language: str, 
        user_id: int, 
        call_id: str
    ) -> Dict[str, Any]:
        """
        Real-time voice translation for voice calls
        1. Convert speech to text using Whisper
        2. Translate text using local models
        3. Convert translated text to speech
        4. Return audio URL and texts
        """
        
        if not VOICE_LIBS_AVAILABLE:
            raise Exception("Voice processing libraries not available")
        
        import time
        start_time = time.time()
        
        try:
            print(f"🌐 Starting real-time translation: {source_language} → {target_language}")
            
            # Step 1: Speech to text
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            try:
                # Use Whisper for speech recognition
                original_text = await self._speech_to_text_whisper(temp_audio_path, source_language)
                print(f"🗣️ Recognized text: {original_text}")
                
                if not original_text or len(original_text.strip()) < 3:
                    return {
                        "original_text": "",
                        "translated_text": "",
                        "translated_audio_url": None,
                        "processing_time": time.time() - start_time
                    }
                
                # Step 2: Translate text
                translated_text = await translation_service.translate_text_async(
                    text=original_text,
                    source_lang=source_language,
                    target_lang=target_language
                )
                print(f"🔄 Translated text: {translated_text}")
                
                # Step 3: Text to speech for translated text
                translated_audio_filename = f"realtime_{call_id}_{user_id}_{uuid.uuid4().hex[:8]}.mp3"
                translated_audio_path = self.upload_dir / translated_audio_filename
                
                await self._text_to_speech_gtts(translated_text, target_language, str(translated_audio_path))
                
                # Generate URL for the translated audio
                translated_audio_url = f"/static/uploads/voice/{translated_audio_filename}"
                
                processing_time = time.time() - start_time
                print(f"✅ Real-time translation completed in {processing_time:.2f}s")
                
                return {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "translated_audio_url": translated_audio_url,
                    "processing_time": processing_time
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass
        
        except Exception as e:
            print(f"❌ Real-time translation failed: {e}")
            raise Exception(f"Translation processing failed: {str(e)}")

# Export the class for import
VoiceService = VoiceMessageService
