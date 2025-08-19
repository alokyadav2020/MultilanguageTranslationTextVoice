"""
Translation service using Facebook M2M100 model for chat application.
Supports English, French, and Arabic translations with local caching.
Uses single lightweight model (418M) optimized for Nvidia Jetson compatibility.
"""

import os
import logging
import asyncio
import concurrent.futures
from typing import Dict, Optional, Any, List
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translation service that manages local Hugging Face models for translation
    between English, French, and Arabic languages.
    """
    
    def __init__(self):
        # Single model and tokenizer for all translations
        self.model = None
        self.tokenizer = None
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "models")
        
        # Use single M2M100 model for all translations (Jetson-optimized)
        self.model_name = "facebook/m2m100_418M"
        self.model_cache_path = os.path.join(self.cache_dir, "m2m100_418M")
        
        # Logging configuration
        self.verbose_logging = False  # Detailed logs for debugging
        self.essential_logging = True  # Essential logs for monitoring (always enabled)
        
        # Translation statistics for monitoring
        self.translation_stats = {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "batch_translations": 0,
            "arabic_translations": 0
        }
        
        # Thread pool for async translation processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="translation")
        
        # Device configuration - use CUDA if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log device information (startup only)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 M2M100 Translation service: CUDA GPU detected ({gpu_name})")
        else:
            logger.info("🖥️  M2M100 Translation service: Using CPU")
        
        # Language mappings for M2M100 (single model supports all pairs)
        self.supported_languages = {"en", "fr", "ar"}
        self.language_codes = {
            "en": "en",
            "fr": "fr", 
            "ar": "ar"
        }
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_cache_path, exist_ok=True)
        
        # Load the single M2M100 model at startup
        self._load_m2m100_model()
        
        if self.essential_logging:
            logger.info(f"M2M100 Translation service initialized with cache directory: {self.cache_dir}")
            logger.info("✅ Single M2M100 model supports all language pairs: EN↔FR↔AR")
    
    def _load_m2m100_model(self):
        """Load the M2M100 model with proper error handling for security and compatibility."""
        try:
            if self.essential_logging:
                logger.info(f"� Loading M2M100 model: {self.model_name}")
                logger.info(f"📂 Cache directory: {self.cache_dir}")
            
            # Check if model is already cached locally
            model_cache_path = os.path.join(self.cache_dir, "models--facebook--m2m100_418M")
            if os.path.exists(model_cache_path):
                if self.essential_logging:
                    logger.info("� Loading from local cache...")
                
                # Load from local cache with safetensors preference
                self.tokenizer = M2M100Tokenizer.from_pretrained(
                    model_cache_path,
                    local_files_only=True,
                    use_safetensors=True  # Prefer safetensors format
                )
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    model_cache_path,
                    local_files_only=True,
                    use_safetensors=True,  # Prefer safetensors format
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                )
            else:
                if self.essential_logging:
                    logger.info("📥 Downloading model from Hugging Face...")
                
                # Download with safetensors preference
                self.tokenizer = M2M100Tokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_safetensors=True
                )
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_safetensors=True,  # Use safe tensor format
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Store the actual cache path for future use
            self.model_cache_path = model_cache_path if os.path.exists(model_cache_path) else os.path.join(self.cache_dir, "models--facebook--m2m100_418M")
            
            if self.essential_logging:
                logger.info(f"✅ M2M100 model loaded successfully on {self.device}")
                logger.info(f"📊 Model parameters: ~418M")
                logger.info(f"💾 Memory usage: ~1.9GB")
            
        except ImportError as e:
            if "protobuf" in str(e).lower():
                logger.error("❌ Protobuf library missing! Install with: pip install protobuf>=4.25.0")
                logger.error("   This is required for M2M100 tokenizer to work.")
            else:
                logger.error(f"❌ Import error: {str(e)}")
            self.model = None
            self.tokenizer = None
            raise
            
        except ValueError as e:
            if "torch.load" in str(e) and "v2.6" in str(e):
                logger.error("❌ PyTorch version too old! Upgrade with: pip install torch>=2.6.0")
                logger.error("   This fixes a critical security vulnerability (CVE-2025-32434)")
                logger.error("   Alternatively, the model will try to use safetensors format automatically.")
            else:
                logger.error(f"❌ Value error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
            raise
            
        except Exception as e:
            logger.error(f"❌ Failed to load M2M100 model: {str(e)}")
            logger.error("   Possible solutions:")
            logger.error("   1. Upgrade PyTorch: pip install torch>=2.6.0")
            logger.error("   2. Install protobuf: pip install protobuf>=4.25.0")
            logger.error("   3. Clear cache: rm -rf ~/.cache/huggingface/")
            logger.error("   4. Check internet connection for model download")
            self.model = None
            self.tokenizer = None
            raise

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language using M2M100.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, fr, ar)
            target_lang: Target language code (en, fr, ar)
            
        Returns:
            Translated text or None if translation fails
        """
        # Update statistics
        self.translation_stats["total_translations"] += 1
        
        # Essential logging for monitoring
        if self.essential_logging and self.translation_stats["total_translations"] % 10 == 0:
            logger.info(f"📊 Translation stats: {self.translation_stats['total_translations']} total, "
                       f"{self.translation_stats['successful_translations']} successful, "
                       f"{self.translation_stats['arabic_translations']} Arabic")
        
        # Detailed logging for debugging
        if self.verbose_logging:
            logger.info(f"🔄 M2M100 Translation: {source_lang} -> {target_lang} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Validate languages
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            if self.essential_logging:
                logger.warning(f"❌ Unsupported language pair: {source_lang} -> {target_lang}")
            self.translation_stats["failed_translations"] += 1
            return None
        
        # Skip translation if languages are the same
        if source_lang == target_lang:
            if self.verbose_logging:
                logger.info("⏭️  Same language detected, returning original text")
            self.translation_stats["successful_translations"] += 1
            return text
        
        # Check if M2M100 model is loaded
        if self.model is None or self.tokenizer is None:
            if self.essential_logging:
                logger.error("❌ M2M100 model not loaded")
            self.translation_stats["failed_translations"] += 1
            return None
        
        try:
            # Use M2M100 for translation
            result = self._translate_m2m100(text, source_lang, target_lang)
            if result and result != text:
                self.translation_stats["successful_translations"] += 1
                if source_lang == "ar" or target_lang == "ar":
                    self.translation_stats["arabic_translations"] += 1
                if self.verbose_logging:
                    logger.info(f"✅ M2M100 translation successful: {source_lang} → {target_lang}")
                return result
            else:
                self.translation_stats["failed_translations"] += 1
                if self.essential_logging:
                    logger.warning(f"⚠️  M2M100 translation failed: {source_lang} → {target_lang}")
                return None
                
        except Exception as e:
            self.translation_stats["failed_translations"] += 1
            if self.essential_logging:
                logger.error(f"❌ M2M100 translation error: {source_lang} → {target_lang}: {e}")
            return None
    
    def _translate_m2m100(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using M2M100 model.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, fr, ar)
            target_lang: Target language code (en, fr, ar)
            
        Returns:
            Translated text or None if translation fails
        """
        try:
            # Set source language for the tokenizer
            self.tokenizer.src_lang = self.language_codes[source_lang]
            
            # Tokenize the input text
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move tensors to the same device as the model
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate translation
            with torch.no_grad():
                # Force the target language
                forced_bos_token_id = self.tokenizer.get_lang_id(self.language_codes[target_lang])
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the translation
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            if self.verbose_logging:
                logger.info(f"🤖 M2M100 translation: '{text[:30]}...' → '{translation[:30]}...'")
            
            return translation.strip()
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ M2M100 translation error: {e}")
            return None

    async def translate_text_async(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Asynchronously translate text from source language to target language.
        This method runs the translation in a background thread to avoid blocking.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, fr, ar)
            target_lang: Target language code (en, fr, ar)
            
        Returns:
            Translated text or None if translation fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.translate_text, 
            text, 
            source_lang, 
            target_lang
        )
    
    async def translate_batch_async(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """
        Asynchronously translate multiple texts using optimized batch processing.
        This uses the synchronous batch translation in a background thread for maximum efficiency.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Update batch statistics
        self.translation_stats["batch_translations"] += 1
        
        # Essential logging for batch operations
        if self.essential_logging:
            logger.info(f"🔄 Batch translation: {len(texts)} texts | {source_lang} -> {target_lang}")
        
        # Skip if same language
        if source_lang == target_lang:
            return texts
        
        # Use optimized batch translation in background thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.translate_batch,
            texts,
            source_lang,
            target_lang
        )
        
        # Log batch completion
        if self.essential_logging:
            success_count = sum(1 for orig, trans in zip(texts, result) if trans != orig)
            logger.info(f"✅ Batch completed: {success_count}/{len(texts)} successful translations")
        
        return result
    
    def get_available_translations(self, source_lang: str) -> list:
        """
        Get list of available target languages for a source language.
        
        Args:
            source_lang: Source language code
            
        Returns:
            List of available target language codes
        """
        if source_lang not in self.supported_languages:
            return []
        
        targets = []
        for lang in self.supported_languages:
            if lang != source_lang:
                targets.append(lang)
        
        return targets
    
    def translate_batch(self, texts: list, source_lang: str, target_lang: str) -> list:
        """
        Translate multiple texts in a single batch using M2M100.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Log batch start
        if self.verbose_logging:
            logger.info(f"🔄 M2M100 batch translation: {len(texts)} texts | {source_lang} -> {target_lang}")
        
        # Skip if same language
        if source_lang == target_lang:
            if self.verbose_logging:
                logger.info("⏭️  Same language batch, returning original texts")
            return texts
        
        # Handle single text efficiently
        if len(texts) == 1:
            result = self.translate_text(texts[0], source_lang, target_lang)
            return [result if result else texts[0]]
        
        # Check if M2M100 model is loaded
        if self.model is None or self.tokenizer is None:
            if self.essential_logging:
                logger.error("❌ M2M100 model not loaded for batch translation")
            return texts
        
        try:
            # Set source language for M2M100 tokenizer
            self.tokenizer.src_lang = self.language_codes[source_lang]
            
            # Batch tokenization
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch generation with M2M100
            with torch.no_grad():
                # Force the target language
                forced_bos_token_id = self.tokenizer.get_lang_id(self.language_codes[target_lang])
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Batch decode
            translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            if self.verbose_logging:
                logger.info(f"✅ M2M100 batch translation completed: {len(translated_texts)} texts")
            
            return translated_texts
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ M2M100 batch translation failed: {str(e)}")
            # Fall back to individual translations
            results = []
            for text in texts:
                result = self.translate_text(text, source_lang, target_lang)
                results.append(result if result else text)
            return results

    def get_statistics(self) -> dict:
        """
        Get translation service statistics.
        
        Returns:
            Dictionary containing translation statistics
        """
        return self.translation_stats.copy()
    
    def translate_to_multiple_languages(self, text: str, source_lang: str, target_langs: list) -> Dict[str, str]:
        """
        Translate text to multiple target languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            
        Returns:
            Dictionary mapping target language codes to translated text
        """
        translations = {}
        
        # Group target languages by translation method for batch processing
        direct_langs = []
        arabic_from_langs = []
        arabic_to_langs = []
        
        for target_lang in target_langs:
            if target_lang != source_lang:
                if source_lang == "ar":
                    arabic_from_langs.append(target_lang)
                elif target_lang == "ar":
                    arabic_to_langs.append(target_lang)
                else:
                    direct_langs.append(target_lang)
        
        # Batch process direct translations (en<->fr)
        if direct_langs:
            for target_lang in direct_langs:
                translated = self.translate_text(text, source_lang, target_lang)
                if translated:
                    translations[target_lang] = translated
        
        # Batch process Arabic translations
        if arabic_from_langs:
            for target_lang in arabic_from_langs:
                translated = self.translate_text(text, source_lang, target_lang)
                if translated:
                    translations[target_lang] = translated
        
        if arabic_to_langs:
            for target_lang in arabic_to_langs:
                translated = self.translate_text(text, source_lang, target_lang)
                if translated:
                    translations[target_lang] = translated
        
        return translations
    
    async def translate_to_multiple_languages_async(self, text: str, source_lang: str, target_langs: list) -> Dict[str, str]:
        """
        Asynchronously translate text to multiple target languages using parallel processing.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            
        Returns:
            Dictionary mapping target language codes to translated text
        """
        if not target_langs:
            return {}
        
        # Filter out same language
        valid_target_langs = [lang for lang in target_langs if lang != source_lang]
        if not valid_target_langs:
            return {source_lang: text}
        
        # Create async tasks for each target language
        loop = asyncio.get_event_loop()
        tasks = []
        task_langs = []
        
        for target_lang in valid_target_langs:
            task = loop.run_in_executor(
                self.executor,
                self.translate_text,
                text,
                source_lang,
                target_lang
            )
            tasks.append(task)
            task_langs.append(target_lang)
        
        # Execute all translations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        translations = {}
        for i, (target_lang, result) in enumerate(zip(task_langs, results)):
            if isinstance(result, Exception):
                if self.verbose_logging:
                    logger.error(f"Translation to {target_lang} failed: {str(result)}")
                translations[target_lang] = text  # Fallback to original
            else:
                translations[target_lang] = result if result else text
        
        # Include original text for same language if requested
        if source_lang in target_langs:
            translations[source_lang] = text
        
        return translations
    
    def preload_models(self):
        """
        Preload common translation models for faster response times.
        This focuses on the most frequently used language pairs.
        """
        if self.verbose_logging:
            logger.info("🚀 Preloading common translation models...")
        
        # Priority language pairs (most commonly used)
        priority_pairs = [
            ("en", "fr"),  # English to French
            ("fr", "en"),  # French to English
            ("en", "ar"),  # English to Arabic
            ("fr", "ar"),  # French to Arabic
        ]
        
        # Load priority models first
        for source_lang, target_lang in priority_pairs:
            if self.verbose_logging:
                logger.info(f"📥 Preloading: {source_lang} -> {target_lang}")
            self._load_model(source_lang, target_lang)
        
        logger.info("✅ Common translation models preloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and cache status.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "cache_directory": self.cache_dir,
            "supported_languages": self.supported_languages,
            "loaded_models": list(self.models.keys()),
            "available_model_pairs": list(self.model_mappings.keys()),
            "cache_size_mb": 0
        }
        
        # Calculate cache size
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logger.warning(f"Could not calculate cache size: {str(e)}")
        
        return info
    
    def _try_mbart_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Try translation using Facebook's mBART model (offline, optimized)."""
        try:
            if self.verbose_logging:
                logger.info(f"🤖 Trying mBART translation: {source_lang} -> {target_lang}")
                
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            
            # Cache the model locally
            model_cache_dir = os.path.join(self.cache_dir, "mbart")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name, cache_dir=model_cache_dir)
            model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=model_cache_dir)
            
            # Move model to the correct device
            model = model.to(self.device)
            
            # Set source language
            tokenizer.src_lang = "ar_AR" if source_lang == "ar" else f"{source_lang}_XX"
            
            encoded_text = tokenizer(text, return_tensors="pt")
            # Move tensors to device
            encoded_text = {k: v.to(self.device) for k, v in encoded_text.items()}
            
            # Generate translation
            target_lang_code = "en_XX" if target_lang == "en" else f"{target_lang}_XX"
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_text,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code],
                    max_length=512,
                    num_beams=2,  # Reduced for speed
                    early_stopping=True
                )
            
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return result
            
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"mBART translation failed: {str(e)}")
            return None
    
    def _pattern_based_arabic_translation(self, text: str, target_lang: str) -> Optional[str]:
        """Simple pattern-based translation for common Arabic words (offline, optimized)."""
        
        if self.verbose_logging:
            logger.info(f"📖 Trying pattern-based Arabic translation to {target_lang}: '{text[:30]}...'")
        
        # Simple dictionary for common Arabic words (offline)
        arabic_to_english = {
            "مرحبا": "hello",
            "مرحباً": "hello",
            "شكرا": "thank you", 
            "شكراً": "thank you",
            "نعم": "yes",
            "لا": "no",
            "السلام عليكم": "peace be upon you",
            "وعليكم السلام": "and upon you peace",
            "كيف حالك": "how are you",
            "أنا بخير": "I am fine",
            "صباح الخير": "good morning",
            "مساء الخير": "good evening",
            "تصبح على خير": "good night",
            "إلى اللقاء": "goodbye",
            "ما اسمك": "what is your name",
            "اسمي": "my name is"
        }
        
        arabic_to_french = {
            "مرحبا": "bonjour",
            "مرحباً": "bonjour", 
            "شكرا": "merci",
            "شكراً": "merci",
            "نعم": "oui",
            "لا": "non"
        }
        
        translation_dict = arabic_to_english if target_lang == "en" else arabic_to_french
        
        # Check if the text matches any known patterns
        text_clean = text.strip()
        if text_clean in translation_dict:
            result = translation_dict[text_clean]
            if self.verbose_logging:
                logger.info(f"✅ Exact pattern match: '{text_clean}' -> '{result}'")
            return result
        
        # Check for partial matches
        for arabic, translation in translation_dict.items():
            if arabic in text:
                result = text.replace(arabic, translation)
                if self.verbose_logging:
                    logger.info(f"✅ Partial pattern match: '{arabic}' -> '{translation}'")
                return result
        
        if self.verbose_logging:
            logger.info(f"❌ No pattern match found for: '{text_clean}'")
        return None

    def get_translation_stats(self) -> Dict[str, Any]:
        """
        Get translation statistics for monitoring and debugging.
        
        Returns:
            Dictionary with translation statistics
        """
        return {
            "statistics": self.translation_stats.copy(),
            "success_rate": (
                self.translation_stats["successful_translations"] / 
                max(self.translation_stats["total_translations"], 1) * 100
            ),
            "device": str(self.device),
            "loaded_models": len(self.models),
            "verbose_logging": self.verbose_logging,
            "essential_logging": self.essential_logging
        }

# Global translation service instance
translation_service = TranslationService()
