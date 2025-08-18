"""
API endpoints for translation management and functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from ..core.database import get_db
from ..api.deps import get_current_user
from ..models.user import User
from ..models.message import Message
from ..services.translation import translation_service
from ..services.chatroom import chatroom_service

router = APIRouter(prefix="/api/translation", tags=["translation"])


@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "fr", "name": "Français"},
            {"code": "ar", "name": "العربية"}
        ],
        "supported_pairs": list(translation_service.model_mappings.keys())
    }


@router.post("/translate")
async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    current_user: User = Depends(get_current_user)
):
    """
    Translate text from source language to target language.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if source_lang not in translation_service.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_lang}")
    
    if target_lang not in translation_service.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}")
    
    try:
        # Use async translation for better performance
        translated_text = await translation_service.translate_text_async(text, source_lang, target_lang)
        
        if translated_text is None:
            raise HTTPException(status_code=500, detail="Translation failed")
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@router.post("/translate/multiple")
async def translate_to_multiple_languages(
    text: str,
    source_lang: str,
    target_langs: List[str],
    current_user: User = Depends(get_current_user)
):
    """
    Translate text to multiple target languages.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if source_lang not in translation_service.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_lang}")
    
    # Validate target languages
    for lang in target_langs:
        if lang not in translation_service.supported_languages:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {lang}")
    
    try:
        # Use async batch translation for better performance
        async_tasks = []
        import asyncio
        
        # Create async tasks for each target language
        for target_lang in target_langs:
            if target_lang != source_lang:  # Skip same language
                task = translation_service.translate_text_async(text, source_lang, target_lang)
                async_tasks.append((target_lang, task))
        
        # Execute all translations in parallel
        translations = {}
        if async_tasks:
            results = await asyncio.gather(*[task for _, task in async_tasks], return_exceptions=True)
            
            for i, (target_lang, _) in enumerate(async_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    translations[target_lang] = text  # Fallback to original
                else:
                    translations[target_lang] = result if result else text
        
        # Include original text for same language
        for target_lang in target_langs:
            if target_lang == source_lang:
                translations[target_lang] = text
        
        return {
            "original_text": text,
            "source_language": source_lang,
            "translations": translations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@router.post("/translate/batch")
async def translate_batch(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    current_user: User = Depends(get_current_user)
):
    """
    Translate multiple texts in a single batch for better performance.
    """
    if not texts or all(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="At least one text must be provided")
    
    if source_lang not in translation_service.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_lang}")
    
    if target_lang not in translation_service.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}")
    
    try:
        # Use async batch translation for maximum performance
        translated_texts = await translation_service.translate_batch_async(texts, source_lang, target_lang)
        
        return {
            "original_texts": texts,
            "translated_texts": translated_texts,
            "source_language": source_lang,
            "target_language": target_lang,
            "count": len(texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")


@router.get("/message/{message_id}")
async def get_message_translations(
    message_id: int,
    target_lang: str = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get translations for a specific message.
    If target_lang is provided, generate missing translation if needed.
    """
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    result = {
        "message_id": message.id,
        "original_text": message.original_text,
        "original_language": message.original_language,
        "translations_cache": message.translations_cache or {}
    }
    
    # Generate missing translation if requested
    if target_lang and target_lang not in (message.translations_cache or {}):
        if target_lang in translation_service.supported_languages:
            try:
                translated_text = chatroom_service.generate_missing_translation(
                    message, target_lang, db
                )
                if translated_text:
                    result["translations_cache"][target_lang] = translated_text
            except Exception:
                # Don't fail the request if translation fails
                pass
    
    return result


@router.get("/stats/chatroom/{chatroom_id}")
async def get_chatroom_translation_stats(
    chatroom_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get translation statistics for a chatroom.
    """
    try:
        stats = chatroom_service.get_translation_statistics(chatroom_id, db)
        languages_used = chatroom_service.get_chatroom_languages(chatroom_id, db)
        
        return {
            "chatroom_id": chatroom_id,
            "statistics": stats,
            "languages_used": languages_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.get("/info")
async def get_translation_service_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about the translation service.
    """
    try:
        info = translation_service.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting service info: {str(e)}")


@router.get("/stats")
async def get_translation_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get translation service statistics and performance metrics.
    """
    try:
        stats = translation_service.get_translation_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting translation stats: {str(e)}")


@router.post("/preload")
async def preload_models(
    current_user: User = Depends(get_current_user)
):
    """
    Preload all translation models (admin operation).
    """
    try:
        translation_service.preload_models()
        return {"message": "All models preloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preloading models: {str(e)}")
