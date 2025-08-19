"""
Translation service using Helsinki-NLP OPUS models for chat application.
Supports English, French, and Arabic translations with local caching.
Uses multiple specialized models optimized for Nvidia Jetson compatibility.
"""

import os
import logging
import asyncio
import concurrent.futures
from typing import Dict, Optional, Any, List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translation service that manages local Helsinki-NLP OPUS models for translation
    between English, French, and Arabic languages.
    """
    
    def __init__(self):
        # Multiple models and tokenizers for specific translation pairs
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Updated cache directory structure to match your Jetson setup
        self.cache_dir = "/home/orin/Desktop/translation_project/artifacts/translation_llm"
        
        # Helsinki-NLP model mappings (matching your downloaded models)
        self.model_mappings = {
            ("en", "fr"): {
                "repo_id": "Helsinki-NLP/opus-mt-en-fr",
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-en-fr")
            },
            ("fr", "en"): {
                "repo_id": "Helsinki-NLP/opus-mt-fr-en", 
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-fr-en")
            },
            ("en", "ar"): {
                "repo_id": "Helsinki-NLP/opus-mt-en-ar",
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-en-ar")
            },
            ("ar", "en"): {
                "repo_id": "Helsinki-NLP/opus-mt-ar-en",
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-ar-en")
            },
            ("fr", "ar"): {
                "repo_id": "Helsinki-NLP/opus-mt-fr-ar",
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-fr-ar")
            },
            ("ar", "fr"): {
                "repo_id": "Helsinki-NLP/opus-mt-ar-fr",
                "local_path": os.path.join(self.cache_dir, "Helsinki-NLP/opus-mt-ar-fr")
            }
        }
        
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
            logger.info(f"🚀 Helsinki-NLP Translation service: CUDA GPU detected ({gpu_name})")
        else:
            logger.info("🖥️  Helsinki-NLP Translation service: Using CPU")
        
        # Language mappings
        self.supported_languages = {"en", "fr", "ar"}
        
        # Ensure cache directory structure exists
        os.makedirs(self.cache_dir, exist_ok=True)
        for lang_pair, config in self.model_mappings.items():
            os.makedirs(config["local_path"], exist_ok=True)
        
        if self.essential_logging:
            logger.info(f"Helsinki-NLP Translation service initialized")
            logger.info(f"📂 Cache directory: {self.cache_dir}")
            logger.info(f"✅ Supporting language pairs: {list(self.model_mappings.keys())}")

    def _get_model_key(self, source_lang: str, target_lang: str) -> str:
        """Get the key for storing the model in memory."""
        return f"{source_lang}-{target_lang}"

    def _load_model(self, source_lang: str, target_lang: str) -> bool:
        """
        Load Helsinki-NLP model for the given language pair.
        Uses your local downloaded models first, downloads if needed.
        """
        model_key = self._get_model_key(source_lang, target_lang)
        
        # Return True if already loaded
        if model_key in self.models:
            return True
        
        # Check if we have this language pair mapping
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.model_mappings:
            if self.essential_logging:
                logger.warning(f"⚠️  No model mapping found for {source_lang}→{target_lang}")
            return False
        
        config = self.model_mappings[lang_pair]
        local_path = config["local_path"]
        repo_id = config["repo_id"]
        
        try:
            # First try to load from local path (your downloaded models)
            if os.path.exists(local_path) and os.listdir(local_path):
                if self.essential_logging:
                    logger.info(f"📂 Loading cached model {source_lang}→{target_lang} from {local_path}")
                
                tokenizer = AutoTokenizer.from_pretrained(local_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
                
            else:
                # Download if not available locally
                if self.essential_logging:
                    logger.info(f"📥 Downloading model {repo_id} to {local_path}")
                
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
                
                # Save to local path for future use
                os.makedirs(local_path, exist_ok=True)
                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)
            
            # Move model to device and set to eval mode
            model.to(self.device)
            model.eval()
            
            # Store in memory
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            
            if self.essential_logging:
                logger.info(f"✅ Model {source_lang}→{target_lang} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ Failed to load model {source_lang}→{target_lang}: {str(e)}")
            return False

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language using Helsinki-NLP models.
        
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
            logger.info(f"🔄 Helsinki-NLP Translation: {source_lang} -> {target_lang} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
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
        
        # Load model if needed
        if not self._load_model(source_lang, target_lang):
            self.translation_stats["failed_translations"] += 1
            return None
        
        try:
            # Use Helsinki-NLP model for translation
            result = self._translate_direct(text, source_lang, target_lang)
            if result and result != text:
                self.translation_stats["successful_translations"] += 1
                if source_lang == "ar" or target_lang == "ar":
                    self.translation_stats["arabic_translations"] += 1
                if self.verbose_logging:
                    logger.info(f"✅ Helsinki-NLP translation successful: {source_lang} → {target_lang}")
                return result
            else:
                self.translation_stats["failed_translations"] += 1
                if self.essential_logging:
                    logger.warning(f"⚠️  Helsinki-NLP translation failed: {source_lang} → {target_lang}")
                return None
                
        except Exception as e:
            self.translation_stats["failed_translations"] += 1
            if self.essential_logging:
                logger.error(f"❌ Helsinki-NLP translation error: {source_lang} → {target_lang}: {e}")
            return None

    def _translate_direct(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Perform direct translation using loaded Helsinki-NLP model.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if translation fails
        """
        model_key = self._get_model_key(source_lang, target_lang)
        
        try:
            # Get model and tokenizer (already loaded)
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode translation
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if self.verbose_logging:
                logger.info(f"🔄 Helsinki-NLP: '{text[:30]}...' → '{translated_text[:30]}...'")
            
            return translated_text.strip()
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ Direct translation error: {e}")
            return None

    async def translate_text_async(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Asynchronously translate text from source language to target language.
        This method runs the translation in a background thread to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.translate_text, 
            text, 
            source_lang, 
            target_lang
        )

    def translate_batch(self, texts: list, source_lang: str, target_lang: str) -> list:
        """
        Translate multiple texts in a single batch using Helsinki-NLP models.
        
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
            logger.info(f"🔄 Helsinki-NLP batch translation: {len(texts)} texts | {source_lang} -> {target_lang}")
        
        # Skip if same language
        if source_lang == target_lang:
            if self.verbose_logging:
                logger.info("⏭️  Same language batch, returning original texts")
            return texts
        
        # Handle single text efficiently
        if len(texts) == 1:
            result = self.translate_text(texts[0], source_lang, target_lang)
            return [result if result else texts[0]]
        
        # Load model if needed
        if not self._load_model(source_lang, target_lang):
            if self.essential_logging:
                logger.error(f"❌ Model not available for batch translation: {source_lang} -> {target_lang}")
            return texts
        
        try:
            model_key = self._get_model_key(source_lang, target_lang)
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
            
            # Batch generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Batch decode
            translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            if self.verbose_logging:
                logger.info(f"✅ Helsinki-NLP batch translation completed: {len(translated_texts)} texts")
            
            return translated_texts
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ Helsinki-NLP batch translation failed: {str(e)}")
            # Fall back to individual translations
            results = []
            for text in texts:
                result = self.translate_text(text, source_lang, target_lang)
                results.append(result if result else text)
            return results

    async def translate_batch_async(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """
        Asynchronously translate multiple texts using optimized batch processing.
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
        """
        if source_lang not in self.supported_languages:
            return []
        
        available = []
        for lang_pair in self.model_mappings.keys():
            if lang_pair[0] == source_lang:
                available.append(lang_pair[1])
        
        return available

    def translate_to_multiple_languages(self, text: str, source_lang: str, target_langs: list) -> Dict[str, str]:
        """
        Translate text to multiple target languages.
        """
        translations = {}
        
        for target_lang in target_langs:
            if target_lang != source_lang:
                translated = self.translate_text(text, source_lang, target_lang)
                if translated:
                    translations[target_lang] = translated
        
        return translations

    async def translate_to_multiple_languages_async(self, text: str, source_lang: str, target_langs: list) -> Dict[str, str]:
        """
        Asynchronously translate text to multiple target languages using parallel processing.
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

    def get_statistics(self) -> dict:
        """
        Get translation service statistics.
        """
        return self.translation_stats.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and cache status.
        """
        info = {
            "cache_directory": self.cache_dir,
            "supported_languages": self.supported_languages,
            "loaded_models": list(self.models.keys()),
            "available_model_pairs": list(self.model_mappings.keys()),
            "model_mappings": self.model_mappings,
            "device": str(self.device),
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

    def preload_models(self):
        """
        Preload common translation models for faster response times.
        """
        if self.essential_logging:
            logger.info("🚀 Preloading Helsinki-NLP translation models...")
        
        # Load all available models
        for (source_lang, target_lang) in self.model_mappings.keys():
            if self.verbose_logging:
                logger.info(f"📥 Preloading: {source_lang} -> {target_lang}")
            self._load_model(source_lang, target_lang)
        
        if self.essential_logging:
            logger.info("✅ Helsinki-NLP models preloaded")

    def download_model_if_needed(self, source_lang: str, target_lang: str) -> bool:
        """
        Download model using your download script structure if not available locally.
        """
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.model_mappings:
            return False
        
        config = self.model_mappings[lang_pair]
        local_path = config["local_path"]
        repo_id = config["repo_id"]
        
        # Check if already downloaded
        if os.path.exists(local_path) and os.listdir(local_path):
            if self.essential_logging:
                logger.info(f"✓ {local_path} already contains files — skipping download.")
            return True
        
        try:
            if self.essential_logging:
                logger.info(f"⏬ Downloading {repo_id} to {local_path} …")
            
            # Use your download structure
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
            
            os.makedirs(local_path, exist_ok=True)
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            
            if self.essential_logging:
                logger.info(f"✅ Saved model to {local_path}")
            return True
            
        except Exception as e:
            if self.essential_logging:
                logger.error(f"❌ Failed to download {repo_id}: {str(e)}")
            return False

# Global translation service instance
translation_service = TranslationService()
