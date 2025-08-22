# filepath: app/services/voice_service.py
import os
import uuid
import tempfile
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

# Voice processing libraries (install with: pip install speech-recognition gtts pydub openai-whisper)
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
from ..models.user import User
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
        
    async def process_voice_message_async(
        self, 
        audio_data: bytes, 
        user_id: int,
        chatroom_id: int,
        target_language: str,
        db: Session
    ) -> Dict:
        """
        Fully asynchronous voice message processing with enhanced accuracy
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
            
            print(f"🎉 Voice message processing complete! Message ID: {message.id}")
            
            return {
                "success": True,
                "message": message,
                "transcribed_text": transcribed_text,
                "translations": translations,
                "audio_urls": audio_urls,
                "target_language": target_language
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
            # Apply noise gate
            chunks = audio[::100]  # Sample every 100ms
            noise_level = min(chunk.dBFS for chunk in chunks if len(chunk) > 50)
            gate_threshold = noise_level + 10  # 10dB above noise floor
            
            # Simple noise gate implementation
            silent_threshold = gate_threshold
            audio = audio.apply_gain_to_silent_portions(silent_threshold, 0, -60)
            print(f"🔇 Applied noise gate at {gate_threshold:.1f}dBFS")
        
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
                    content=transcribed_text,
                    translated_content=translations.get(target_language, transcribed_text),
                    original_language=self._detect_language(transcribed_text),
                    target_language=target_language,
                    message_type=MessageType.voice
                )
                
                db.add(message)
                db.flush()  # Get message ID
                
                # Create audio file record
                audio_file = AudioFile(
                    message_id=message.id,
                    file_path=audio_file_path,
                    duration=duration,
                    file_size=file_size,
                    format='wav'
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
                    content=placeholder_text['en'],
                    translated_content=placeholder_text.get(target_language, placeholder_text['en']),
                    original_language='en',
                    target_language=target_language,
                    message_type=MessageType.voice
                )
                
                db.add(message)
                db.flush()
                
                # Create audio file record
                audio_file = AudioFile(
                    message_id=message.id,
                    file_path=audio_file_path,
                    duration=duration,
                    file_size=file_size,
                    format='wav'
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
        except:
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

# Create service instance
VoiceService = VoiceMessageService()
        """Convert speech to text using OpenAI Whisper"""
        if not WHISPER_AVAILABLE or not self.whisper_model:
            print("Whisper not available, skipping...")
            return None
            
        print(f"🎙️ Starting Whisper speech recognition for: {audio_path}, language: {language}")
        
        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
        
        try:
            # Whisper language mapping
            whisper_lang_map = {
                'en': 'english',
                'fr': 'french', 
                'ar': 'arabic'
            }
            
            whisper_language = whisper_lang_map.get(language, 'english')
            print(f"Using Whisper language: {whisper_language}")
            
            # Quick audio quality check
            try:
                quick_check = AudioSegment.from_file(audio_path)
                print(f"Audio check - Duration: {len(quick_check)}ms, Volume: {quick_check.dBFS:.1f} dBFS")
                
                if len(quick_check) < 300:
                    print("Audio too short for Whisper recognition")
                    return None
            except Exception as e:
                print(f"Error checking audio quality: {e}")
                return None
            
            # Preprocess audio for better Whisper performance
            print("Preprocessing audio for Whisper...")
            processed_path = await self.preprocess_audio_for_whisper(audio_path)
            
            # Use Whisper to transcribe
            print("Running Whisper transcription...")
            result = self.whisper_model.transcribe(
                processed_path, 
                language=whisper_language,
                fp16=False,  # Use FP32 for better compatibility
                verbose=False
            )
            
            text = result["text"].strip()
            
            print(f"✅ Whisper transcription successful: '{text}'")
            
            # Clean up processed file if different from original
            if processed_path != audio_path and os.path.exists(processed_path):
                os.unlink(processed_path)
            
            # Return text if it's meaningful (not just empty or very short)
            if len(text) >= 2 and text.lower() not in ['', 'um', 'uh', 'mm', 'hmm']:
                return text
            else:
                print(f"Whisper result too short or meaningless: '{text}'")
                return None
                
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
    async def preprocess_audio_for_whisper(self, audio_path: str) -> str:
        """Preprocess audio for optimal Whisper performance"""
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Whisper works best with 16kHz mono audio
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Boost quiet audio more aggressively for Whisper
            if audio.dBFS < -30:
                boost_db = -15 - audio.dBFS  # Target around -15 dBFS for Whisper
                audio = audio + boost_db
                print(f"Boosted audio for Whisper: {boost_db:.1f}dB, new level: {audio.dBFS:.1f} dBFS")
            
            # Create optimized file for Whisper
            whisper_path = audio_path.replace('.wav', '_whisper.wav')
            audio.export(whisper_path, format='wav', parameters=[
                "-ar", "16000",  # Sample rate
                "-ac", "1",      # Mono
                "-acodec", "pcm_s16le"  # 16-bit PCM
            ])
            
            print(f"Preprocessed audio for Whisper: {whisper_path}")
            return whisper_path
            
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            return audio_path  # Return original if preprocessing fails
    
    def speech_to_text_google(self, audio_path: str, language: str) -> Optional[str]:
        """Convert speech to text using Google Speech Recognition (fallback)"""
        print(f"🔄 Fallback: Using Google Speech Recognition for: {audio_path}")
        return self.speech_to_text(audio_path, language)

    def speech_to_text(self, audio_path: str, language: str) -> Optional[str]:
        """Convert speech to text using speech_recognition"""
        if not VOICE_LIBS_AVAILABLE:
            print("Voice libraries not available")
            return None
            
        print(f"Starting speech recognition for: {audio_path}, language: {language}")
        
        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
        
        # Quick audio quality check
        try:
            quick_check = AudioSegment.from_file(audio_path)
            print(f"Initial audio check - Duration: {len(quick_check)}ms, Volume: {quick_check.dBFS:.1f} dBFS")
            
            if len(quick_check) < 300:
                print("Audio too short for recognition")
                return None
                
            if quick_check.dBFS < -70:
                print("Audio extremely quiet - may not contain speech")
                # Continue anyway but warn
                
        except Exception as e:
            print(f"Error checking audio quality: {e}")
            return None
            
        r = sr.Recognizer()
        
        try:
            print("Converting audio file to WAV format...")
            # Convert to WAV format for speech recognition
            audio = AudioSegment.from_file(audio_path)
            print(f"Audio duration: {len(audio)}ms, channels: {audio.channels}, sample rate: {audio.frame_rate}")
            
            # Audio preprocessing for better recognition
            # Normalize audio levels first
            audio = audio.normalize()
            print(f"After normalization: {audio.dBFS:.1f} dBFS")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("Converted to mono audio")
            
            # Adjust sample rate to 16kHz (optimal for speech recognition)
            original_rate = audio.frame_rate
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                print(f"Adjusted sample rate to 16000Hz from {original_rate}Hz")
            
            # Check if audio is too short (less than 0.5 seconds)
            if len(audio) < 500:
                print(f"Audio too short: {len(audio)}ms")
                return None
            
            # Boost very quiet audio more aggressively
            if audio.dBFS < -35:
                # Calculate boost needed to get to around -20 dBFS
                boost_db = -20 - audio.dBFS
                audio = audio + boost_db
                print(f"Boosted quiet audio by {boost_db:.1f}dB, new level: {audio.dBFS:.1f} dBFS")
            elif audio.dBFS < -25:
                # Modest boost for moderately quiet audio
                boost_db = -20 - audio.dBFS
                audio = audio + boost_db
                print(f"Boosted audio by {boost_db:.1f}dB, new level: {audio.dBFS:.1f} dBFS")
            
            # Apply compression to even out volume levels
            # This helps with varying speech volumes
            compressed = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            if compressed.dBFS > audio.dBFS - 3:  # Only use if it doesn't make it much quieter
                audio = compressed
                print(f"Applied dynamic compression, level: {audio.dBFS:.1f} dBFS")
            
            wav_path = audio_path.replace('.webm', '.wav').replace('.mp3', '.wav').replace('.m4a', '.wav')
            audio.export(wav_path, format='wav', parameters=["-ar", "16000", "-ac", "1"])
            print(f"Exported preprocessed WAV to: {wav_path}")
            
            # Perform speech recognition with improved settings
            print("Performing speech recognition...")
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise with longer duration
                r.adjust_for_ambient_noise(source, duration=1.0)
                
                # Increase energy threshold for better noise handling
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                
                # Record the entire audio file
                audio_data = r.record(source)
            
            # Language mapping for Google Speech Recognition (only English, French, Arabic)
            lang_map = {
                'en': 'en-US',
                'fr': 'fr-FR', 
                'ar': 'ar-SA'
            }
            
            recognition_lang = lang_map.get(language, 'en-US')
            print(f"Using recognition language: {recognition_lang}")
            
            # Try multiple recognition attempts with different settings
            text = None
            attempts = [
                # Attempt 1: Standard recognition
                {"show_all": False},
                # Attempt 2: Get all results and pick best
                {"show_all": True},
                # Attempt 3: Try with default language if specific language fails
                {"language": "en-US", "show_all": False} if recognition_lang != "en-US" else None
            ]
            
            for i, attempt_params in enumerate(attempts):
                if attempt_params is None:
                    continue
                    
                try:
                    print(f"Recognition attempt {i+1} with params: {attempt_params}")
                    
                    if attempt_params.get("show_all"):
                        # Get all possible transcriptions
                        results = r.recognize_google(audio_data, language=recognition_lang, show_all=True)
                        if results and len(results) > 0:
                            # Pick the most confident result
                            text = results[0]["transcript"]
                            confidence = results[0].get("confidence", 0)
                            print(f"Got transcription with confidence {confidence}: {text}")
                            break
                    else:
                        # Standard recognition
                        lang_to_use = attempt_params.get("language", recognition_lang)
                        text = r.recognize_google(audio_data, language=lang_to_use)
                        print(f"Recognition attempt {i+1} successful: {text}")
                        break
                        
                except sr.UnknownValueError:
                    print(f"Recognition attempt {i+1} failed: Could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"Recognition attempt {i+1} failed: Request error: {e}")
                    continue
                except Exception as e:
                    print(f"Recognition attempt {i+1} failed: {e}")
                    continue
            
            if not text:
                print("All recognition attempts failed")
                
            print(f"Final speech recognition result: {text}")
            
            # Clean up WAV file (but keep for debugging if recognition failed)
            if os.path.exists(wav_path):
                if text:
                    os.unlink(wav_path)
                    print("Cleaned up WAV file")
                else:
                    print(f"Recognition failed. WAV file kept for debugging: {wav_path}")
                    # Optionally, you can move it to a debug folder
            
            return text
            
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio (final attempt)")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            return None
        except Exception as e:
            print(f"Speech recognition error: {e}")
            # Print more detailed error info
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
    async def translate_text(self, text: str, source_lang: str) -> Dict[str, str]:
        """Translate text to supported languages (English, French, Arabic only)"""
        target_languages = ['en', 'fr', 'ar']
        translations = {}
        
        for target_lang in target_languages:
            if target_lang != source_lang:
                try:
                    # Use your existing translation service (sync version)
                    translated = translation_service.translate_text(text, source_lang, target_lang)
                    translations[target_lang] = translated
                except Exception as e:
                    print(f"Translation error for {target_lang}: {e}")
                    translations[target_lang] = text  # Fallback to original
        
        return translations
    
    async def generate_voice_files(self, original_text: str, translations: Dict[str, str], original_lang: str) -> Dict[str, str]:
        """Generate TTS audio files for original and translated text"""
        audio_urls = {}
        
        if not VOICE_LIBS_AVAILABLE:
            return audio_urls
        
        try:
            # Generate for original language
            audio_urls[original_lang] = await self.text_to_speech(original_text, original_lang)
            
            # Generate for translations
            for lang, text in translations.items():
                if text and lang != original_lang:
                    audio_urls[lang] = await self.text_to_speech(text, lang)
                    
        except Exception as e:
            print(f"TTS generation error: {e}")
        
        return audio_urls
    
    async def create_voice_only_message(
        self, 
        audio_file_path: str, 
        language: str, 
        sender_id: int, 
        recipient_id: int,
        db: Session
    ) -> Dict:
        """Create a voice message when transcription fails"""
        
        # Create a placeholder text based on language
        placeholder_texts = {
            'en': "[Voice message - transcription unavailable]",
            'fr': "[Message vocal - transcription indisponible]", 
            'ar': "[رسالة صوتية - النسخ غير متاح]"
        }
        
        placeholder_text = placeholder_texts.get(language, placeholder_texts['en'])
        
        # Save the original audio file to uploads directory
        audio_url = await self.save_original_audio(audio_file_path, language)
        
        # Get audio metadata
        duration = self.get_audio_duration(audio_file_path)
        file_size = os.path.getsize(audio_file_path)
        
        # Get or create chatroom
        chatroom = self.get_or_create_direct_chatroom(sender_id, recipient_id, db)
        
        # Create message with placeholder text
        message = Message(
            chatroom_id=chatroom.id,
            sender_id=sender_id,
            original_text=placeholder_text,
            original_language=language,
            message_type=MessageType.voice,
            original_audio_path=audio_file_path,
            audio_urls={language: audio_url},  # Only original language
            audio_duration=duration,
            audio_file_size=file_size,
            translations_cache={}  # No translations available
        )
        
        db.add(message)
        db.commit()
        db.refresh(message)
        
        # Create audio file record
        audio_file = AudioFile(
            message_id=message.id,
            language=language,
            file_path=audio_url,
            file_url=audio_url,
            file_size=file_size,
            duration=duration,
            mime_type="audio/wav"
        )
        
        db.add(audio_file)
        db.commit()
        
        return {
            "message": message,
            "transcribed_text": placeholder_text,
            "translations": {},
            "audio_urls": {language: audio_url},
            "duration": duration
        }
    
    async def save_original_audio(self, audio_file_path: str, language: str) -> str:
        """Save original audio file to uploads directory"""
        try:
            # Generate unique filename
            filename = f"voice_{uuid.uuid4().hex}_{language}.wav"
            save_path = self.upload_dir / filename
            
            # Copy the processed audio file
            import shutil
            shutil.copy2(audio_file_path, save_path)
            
            return f"/static/uploads/voice/{filename}"
            
        except Exception as e:
            print(f"Error saving original audio: {e}")
            return ""
    
    async def text_to_speech(self, text: str, language: str) -> Optional[str]:
        """Convert text to speech using gTTS (English, French, Arabic only)"""
        if not VOICE_LIBS_AVAILABLE:
            return None
            
        try:
            # Language mapping for gTTS (only supported languages)
            lang_map = {
                'en': 'en',
                'fr': 'fr',
                'ar': 'ar'
            }
            
            tts_lang = lang_map.get(language, 'en')
            
            # Create TTS object
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            
            # Generate unique filename
            filename = f"tts_{uuid.uuid4().hex}_{language}.mp3"
            file_path = self.upload_dir / filename
            
            # Save TTS audio
            tts.save(str(file_path))
            
            # Return URL for accessing the file
            return f"/static/uploads/voice/{filename}"
            
        except Exception as e:
            print(f"TTS error for {language}: {e}")
            return None
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        if not VOICE_LIBS_AVAILABLE:
            return 0.0
            
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0
    
    def get_or_create_direct_chatroom(self, user1_id: int, user2_id: int, db: Session):
        """Get or create direct chatroom between two users"""
        
        # Create unique chatroom name for direct messages
        direct_key = f"direct:{min(user1_id, user2_id)}:{max(user1_id, user2_id)}"
        
        # Check if chatroom exists
        chatroom = db.query(Chatroom).filter(
            Chatroom.chatroom_name == direct_key,
            Chatroom.is_group_chat.is_(False)
        ).first()
        
        if chatroom:
            return chatroom
        
        # Create new chatroom
        chatroom = Chatroom(
            chatroom_name=direct_key,
            is_group_chat=False
        )
        db.add(chatroom)
        db.flush()  # Get ID without committing
        
        # Add members
        member1 = ChatroomMember(chatroom_id=chatroom.id, user_id=user1_id)
        member2 = ChatroomMember(chatroom_id=chatroom.id, user_id=user2_id)
        
        db.add(member1)
        db.add(member2)
        db.commit()
        db.refresh(chatroom)
        
        return chatroom
    
    async def create_audio_file_records(self, message: Message, audio_urls: Dict[str, str], db: Session):
        """Create detailed AudioFile records for each generated audio file"""
        
        for language, url in audio_urls.items():
            if url:
                # Extract filename from URL
                filename = url.split("/")[-1]
                file_path = self.upload_dir / filename
                
                # Get file details
                file_size = 0
                duration = 0.0
                
                if file_path.exists():
                    file_size = os.path.getsize(file_path)
                    duration = self.get_audio_duration(str(file_path))
                
                # Create AudioFile record
                audio_file = AudioFile(
                    message_id=message.id,
                    language=language,
                    file_path=str(file_path),
                    file_url=url,
                    file_size=file_size,
                    duration=duration,
                    mime_type="audio/mp3"
                )
                
                db.add(audio_file)
        
        db.commit()

# Create service instance
voice_service = VoiceMessageService()
