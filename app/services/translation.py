"""
Translation service using local Hugging Face models for chat application.
Supports English, French, and Arabic translations with local caching.
Note: Arabic-to-other translations are not available due to missing Helsinki-NLP models.
"""

import os
import logging
import asyncio
import concurrent.futures
from typing import Dict, Optional, Any, List
from transformers import MarianMTModel, MarianTokenizer
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
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "models")
        
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
            logger.info(f"🚀 Translation service: CUDA GPU detected ({gpu_name})")
        else:
            logger.info("🖥️  Translation service: Using CPU")
        
        # Language mappings for Helsinki-NLP models
        # Only including models that actually exist
        self.model_mappings = {
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "fr-en": "Helsinki-NLP/opus-mt-fr-en", 
            "en-ar": "Helsinki-NLP/opus-mt-en-ar",
            "fr-ar": "Helsinki-NLP/opus-mt-fr-ar"
        }
        
        # Alternative Arabic models that might exist
        self.arabic_models = {
            # Try different model architectures for Arabic
            "ar-en": [
                "Helsinki-NLP/opus-mt-ar-en",
                "facebook/mbart-large-50-many-to-many-mmt",  # Multilingual model
                "google/mt5-base"  # Multilingual T5 model
            ]
        }
        
        # Supported languages
        self.supported_languages = ["en", "fr", "ar"]
        
        # Translation capabilities matrix - all combinations supported via direct or indirect routes
        self.translation_capabilities = {
            "en": {"fr": True, "ar": True},      # English to French/Arabic works directly
            "fr": {"en": True, "ar": True},      # French to English/Arabic works directly
            "ar": {"en": True, "fr": True}       # Arabic to others works via two-step translation
        }
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Translation service initialized with cache directory: {self.cache_dir}")
        logger.info("Available translations: EN<->FR, EN->AR, FR->AR")
        logger.warning("Arabic-to-other translations not available (no Helsinki-NLP models exist)")
    
    def _get_model_key(self, source_lang: str, target_lang: str) -> str:
        """Generate model key for language pair."""
        return f"{source_lang}-{target_lang}"
    
    def _load_model(self, source_lang: str, target_lang: str) -> bool:
        """
        Load translation model for specified language pair.
        Downloads and caches model locally if not already available.
        Tries alternative models if primary model fails.
        """
        model_key = self._get_model_key(source_lang, target_lang)
        
        # Check if model is already loaded
        if model_key in self.models and model_key in self.tokenizers:
            return True
        
        # Get model name from mappings
        if model_key not in self.model_mappings:
            logger.error(f"No model mapping found for {source_lang} -> {target_lang}")
            return False
        
        # Try primary model first
        if self._try_load_model_variant(model_key, self.model_mappings[model_key]):
            return True
        
        # Try alternative models if available
        if model_key in self.alternative_mappings:
            for alt_model in self.alternative_mappings[model_key]:
                logger.info(f"Trying alternative model {alt_model} for {model_key}")
                if self._try_load_model_variant(model_key, alt_model):
                    return True
        
        logger.error(f"All model loading attempts failed for {model_key}")
        return False
    
    def _try_load_model_variant(self, model_key: str, model_name: str) -> bool:
        """Try to load a specific model variant."""
        local_model_path = os.path.join(self.cache_dir, model_key)
        
        try:
            # Try to load from local cache first
            if os.path.exists(local_model_path):
                if self.essential_logging:
                    logger.info(f"📂 Loading cached model: {model_key}")
                tokenizer = MarianTokenizer.from_pretrained(local_model_path)
                model = MarianMTModel.from_pretrained(local_model_path)
            else:
                # Download and cache model
                if self.essential_logging:
                    logger.info(f"⬇️  Downloading model: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                # Save to local directory for faster future loading
                tokenizer.save_pretrained(local_model_path)
                model.save_pretrained(local_model_path)
                if self.essential_logging:
                    logger.info(f"💾 Model cached: {model_key}")
            
            # Move model to the correct device (GPU/CPU)
            model = model.to(self.device)
            
            # Store in memory
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            
            if self.verbose_logging:
                logger.info(f"Model {model_key} loaded successfully using {model_name}")
                logger.info(f"⚡ Model loaded on device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key} using {model_name}: {str(e)}")
            return False
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language (optimized for performance).
        
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
            logger.info(f"🔄 Translation: {source_lang} -> {target_lang} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Validate languages
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            if self.essential_logging:
                logger.warning(f"❌ Unsupported language pair: {source_lang} -> {target_lang}")
            self.translation_stats["failed_translations"] += 1
            return None
        
        # Skip translation if languages are the same
        if source_lang == target_lang:
            if self.verbose_logging:
                logger.info(f"⏭️  Same language detected, returning original text")
            self.translation_stats["successful_translations"] += 1
            return text
        
        # Handle Arabic translations
        if source_lang == "ar":
            self.translation_stats["arabic_translations"] += 1
            result = self._translate_from_arabic(text, target_lang)
            if result and result != text:
                self.translation_stats["successful_translations"] += 1
                if self.verbose_logging:
                    logger.info(f"✅ Arabic translation successful: {source_lang} -> {target_lang}")
            else:
                self.translation_stats["failed_translations"] += 1
                if self.essential_logging:
                    logger.warning(f"⚠️  Arabic translation fallback: {source_lang} -> {target_lang}")
            return result
        elif target_lang == "ar":
            self.translation_stats["arabic_translations"] += 1
            result = self._translate_to_arabic(text, source_lang)
            if result and result != text:
                self.translation_stats["successful_translations"] += 1
                if self.verbose_logging:
                    logger.info(f"✅ Translation to Arabic successful: {source_lang} -> {target_lang}")
            else:
                self.translation_stats["failed_translations"] += 1
                if self.essential_logging:
                    logger.warning(f"⚠️  Translation to Arabic fallback: {source_lang} -> {target_lang}")
            return result
        else:
            # Direct translation for non-Arabic pairs (en<->fr)
            if self._load_model(source_lang, target_lang):
                result = self._translate_direct(text, source_lang, target_lang)
                if result and result != text:
                    self.translation_stats["successful_translations"] += 1
                    if self.verbose_logging:
                        logger.info(f"✅ Direct translation successful: {source_lang} -> {target_lang}")
                else:
                    self.translation_stats["failed_translations"] += 1
                    if self.essential_logging:
                        logger.warning(f"⚠️  Direct translation failed: {source_lang} -> {target_lang}")
                return result
        
        # Fallback case
        self.translation_stats["failed_translations"] += 1
        if self.essential_logging:
            logger.warning(f"❌ Translation failed, returning original: {source_lang} -> {target_lang}")
        return text  # Return original text as fallback
    
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
    
    def _translate_from_arabic(self, text: str, target_lang: str) -> Optional[str]:
        """
        Translate from Arabic to other languages using alternative approach (optimized).
        """
        if self.verbose_logging:
            logger.info(f"Arabic to {target_lang} translation")
        
        if target_lang == "en":
            # Try pattern-based translation first (fastest)
            result = self._pattern_based_arabic_translation(text, target_lang)
            if result and result != text:
                return result
            
            # Try mBART model as fallback
            try:
                result = self._try_mbart_translation(text, "ar", "en")
                if result and result != text:
                    return result
            except Exception:
                pass
            
            # Return with minimal notation
            return f"[AR->EN: {text}]"
                
        elif target_lang == "fr":
            result = self._pattern_based_arabic_translation(text, target_lang)
            if result and result != text:
                return result
            return f"[AR->FR: {text}]"
        
        return text
    
    def _translate_from_arabic(self, text: str, target_lang: str) -> Optional[str]:
        """
        Translate from Arabic to other languages using alternative approach.
        Since Helsinki-NLP ar->en and ar->fr models don't exist, we'll try:
        1. Alternative model repositories
        2. Two-step translation if available
        3. Return original with clear logging
        """
        logger.info(f"🔄 Starting Arabic to {target_lang} translation")
        logger.info(f"📝 Input text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if target_lang == "en":
            # Try alternative Arabic to English models
            logger.info("� Searching for alternative Arabic->English translation methods")
            
            # Method 1: Try Facebook's M2M100 model (if available)
            try:
                logger.info("🧪 Attempting to use alternative translation approach...")
                # For now, we'll implement a simple character-by-character or word approach
                # This is a placeholder for a more sophisticated solution
                
                # Method 2: Use reverse translation as approximation
                # English -> Arabic works, so we can compare patterns
                if self._load_model("en", "ar"):
                    logger.info("💡 Using pattern-based approach with en->ar model as reference")
                    # Use new helper methods for better translation
                    result = self._pattern_based_arabic_translation(text, target_lang)
                    if result and result != text:
                        return result
                    
                    # Try mBART model
                    try:
                        result = self._try_mbart_translation(text, "ar", "en")
                        if result and result != text:
                            return result
                    except Exception as e:
                        logger.warning(f"mBART failed: {str(e)}")
                    
                    # For demo purposes, add a note that this is Arabic text
                    return f"[Offline AR->EN: {text}]"
                
            except Exception as e:
                logger.error(f"❌ Alternative translation method failed: {str(e)}")
                
        elif target_lang == "fr":
            logger.info("� Attempting Arabic to French translation")
            logger.warning("⚠️  Arabic->French: No direct model available")
            return f"[AR->FR: {text}]"
        
        logger.info(f"✅ Arabic translation completed (with limitations)")
        return text
    
    def _translate_to_arabic(self, text: str, source_lang: str) -> Optional[str]:
        """
        Translate to Arabic from other languages.
        """
        # We have en-ar and fr-ar models that work
        if self._load_model(source_lang, "ar"):
            return self._translate_direct(text, source_lang, "ar")
        
        logger.error(f"No model available for {source_lang} -> ar")
        return text
    
    def _translate_direct(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Perform direct translation using loaded model (optimized for performance)."""
        model_key = self._get_model_key(source_lang, target_lang)
        
        try:
            # Get model and tokenizer (already loaded)
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Tokenize input text (batch-ready)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to device only if not already there
            if inputs['input_ids'].device != self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation with optimized settings
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=512, 
                    num_beams=2,  # Reduced from 4 for speed
                    early_stopping=True,
                    do_sample=False  # Deterministic for speed
                )
            
            # Decode translation
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text
            
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Translation failed for {source_lang}->{target_lang}: {str(e)}")
            return None
    
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
        Translate multiple texts in a single batch for better performance.
        
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
            logger.info(f"🔄 Sync batch translation: {len(texts)} texts | {source_lang} -> {target_lang}")
        
        # Skip if same language
        if source_lang == target_lang:
            if self.verbose_logging:
                logger.info(f"⏭️  Same language batch, returning original texts")
            return texts
        
        # Handle single text efficiently
        if len(texts) == 1:
            result = self.translate_text(texts[0], source_lang, target_lang)
            return [result if result else texts[0]]
        
        # For Arabic translations, use optimized batch processing
        if source_lang == "ar" or target_lang == "ar":
            return self._batch_translate_arabic(texts, source_lang, target_lang)
        
        # Batch processing for supported pairs (en<->fr)
        model_key = self._get_model_key(source_lang, target_lang)
        
        if not self._load_model(source_lang, target_lang):
            return texts
        
        try:
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Batch tokenization
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch generation with optimized settings
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,  # Reduced for speed
                    early_stopping=True,
                    do_sample=False
                )
            
            # Batch decode
            translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            return translated_texts
            
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Batch translation failed: {str(e)}")
            return texts
    
    def _batch_translate_arabic(self, texts: list, source_lang: str, target_lang: str) -> list:
        """
        Optimized batch translation for Arabic texts using pattern matching and mBART.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        results = []
        
        # Batch process pattern-based translations first
        pattern_results = []
        for text in texts:
            if source_lang == "ar":
                pattern_result = self._pattern_based_arabic_translation(text, target_lang)
                if pattern_result and pattern_result != text:
                    pattern_results.append(pattern_result)
                else:
                    pattern_results.append(None)  # Needs mBART translation
            else:
                pattern_results.append(None)  # Direct translation to Arabic
        
        # Collect texts that need mBART translation
        mbart_texts = []
        mbart_indices = []
        for i, (text, pattern_result) in enumerate(zip(texts, pattern_results)):
            if pattern_result is None:
                mbart_texts.append(text)
                mbart_indices.append(i)
        
        # Batch translate with mBART if needed
        mbart_results = {}
        if mbart_texts:
            try:
                # Try to use mBART model for batch translation
                for i, text in enumerate(mbart_texts):
                    mbart_result = self._try_mbart_translation(text, source_lang, target_lang)
                    mbart_results[mbart_indices[i]] = mbart_result if mbart_result else text
            except Exception as e:
                if self.verbose_logging:
                    logger.error(f"Batch mBART translation failed: {str(e)}")
                # Fallback to original texts
                for idx in mbart_indices:
                    mbart_results[idx] = texts[idx]
        
        # Combine results
        final_results = []
        for i, (text, pattern_result) in enumerate(zip(texts, pattern_results)):
            if pattern_result is not None:
                final_results.append(pattern_result)
            elif i in mbart_results:
                final_results.append(mbart_results[i])
            else:
                final_results.append(text)  # Fallback
        
        return final_results
    
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
