"""
Translation service using local Hugging Face models for chat application.
Supports English, French, and Arabic translations with local caching.
"""

import os
import logging
from typing import Dict, Optional, Any
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
        
        # Language mappings for Helsinki-NLP models
        self.model_mappings = {
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "fr-en": "Helsinki-NLP/opus-mt-fr-en", 
            "en-ar": "Helsinki-NLP/opus-mt-en-ar",
            "ar-en": "Helsinki-NLP/opus-mt-ar-en",
            "fr-ar": "Helsinki-NLP/opus-mt-fr-ar",
            "ar-fr": "Helsinki-NLP/opus-mt-ar-fr"
        }
        
        # Supported languages
        self.supported_languages = ["en", "fr", "ar"]
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Translation service initialized with cache directory: {self.cache_dir}")
    
    def _get_model_key(self, source_lang: str, target_lang: str) -> str:
        """Generate model key for language pair."""
        return f"{source_lang}-{target_lang}"
    
    def _load_model(self, source_lang: str, target_lang: str) -> bool:
        """
        Load translation model for specified language pair.
        Downloads and caches model locally if not already available.
        """
        model_key = self._get_model_key(source_lang, target_lang)
        
        # Check if model is already loaded
        if model_key in self.models and model_key in self.tokenizers:
            return True
        
        # Get model name from mappings
        if model_key not in self.model_mappings:
            logger.error(f"No model mapping found for {source_lang} -> {target_lang}")
            return False
        
        model_name = self.model_mappings[model_key]
        local_model_path = os.path.join(self.cache_dir, model_key)
        
        try:
            # Try to load from local cache first
            if os.path.exists(local_model_path):
                logger.info(f"Loading cached model from {local_model_path}")
                tokenizer = MarianTokenizer.from_pretrained(local_model_path)
                model = MarianMTModel.from_pretrained(local_model_path)
            else:
                # Download and cache model
                logger.info(f"Downloading model {model_name} to {local_model_path}")
                tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                # Save to local directory for faster future loading
                tokenizer.save_pretrained(local_model_path)
                model.save_pretrained(local_model_path)
                logger.info(f"Model cached successfully at {local_model_path}")
            
            # Store in memory
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            
            logger.info(f"Model {model_key} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {str(e)}")
            return False
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, fr, ar)
            target_lang: Target language code (en, fr, ar)
            
        Returns:
            Translated text or None if translation fails
        """
        # Validate languages
        if source_lang not in self.supported_languages:
            logger.error(f"Unsupported source language: {source_lang}")
            return None
            
        if target_lang not in self.supported_languages:
            logger.error(f"Unsupported target language: {target_lang}")
            return None
        
        # Skip translation if languages are the same
        if source_lang == target_lang:
            return text
        
        # Load model if not already loaded
        if not self._load_model(source_lang, target_lang):
            return None
        
        model_key = self._get_model_key(source_lang, target_lang)
        
        try:
            # Get model and tokenizer
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode translation
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Successfully translated from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation failed for {source_lang} -> {target_lang}: {str(e)}")
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
        
        for target_lang in target_langs:
            if target_lang != source_lang:
                translated = self.translate_text(text, source_lang, target_lang)
                if translated:
                    translations[target_lang] = translated
        
        return translations
    
    def preload_models(self):
        """
        Preload all translation models for faster response times.
        This can be called during application startup.
        """
        logger.info("Preloading all translation models...")
        
        for source_lang in self.supported_languages:
            for target_lang in self.supported_languages:
                if source_lang != target_lang:
                    logger.info(f"Preloading model: {source_lang} -> {target_lang}")
                    self._load_model(source_lang, target_lang)
        
        logger.info("All models preloaded successfully")
    
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

# Global translation service instance
translation_service = TranslationService()
