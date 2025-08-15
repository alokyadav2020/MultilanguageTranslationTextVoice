"""
Chatroom service helper for managing chat functionality with translation support.
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from ..models.message import Message
from ..services.translation import translation_service


class ChatroomService:
    """Service for managing chatroom operations with translation support."""
    
    @staticmethod
    def get_messages_with_translations(
        chatroom_id: int, 
        db: Session, 
        user_language: str = "en",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get chatroom messages with translation support.
        
        Args:
            chatroom_id: ID of the chatroom
            db: Database session
            user_language: Preferred language for the user
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with translation data
        """
        messages = (
            db.query(Message)
            .filter(Message.chatroom_id == chatroom_id)
            .order_by(Message.timestamp.asc())
            .limit(limit)
            .all()
        )
        
        result = []
        for msg in messages:
            message_data = {
                "id": msg.id,
                "sender_id": msg.sender_id,
                "original_text": msg.original_text,
                "original_language": msg.original_language,
                "translations_cache": msg.translations_cache or {},
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "message_type": msg.message_type.value if hasattr(msg.message_type, "value") else str(msg.message_type),
            }
            
            # Add display text based on user's preferred language
            display_text = ChatroomService.get_display_text(
                msg.original_text,
                msg.original_language,
                msg.translations_cache or {},
                user_language
            )
            message_data["display_text"] = display_text
            message_data["is_translated"] = (display_text != msg.original_text)
            
            result.append(message_data)
        
        return result
    
    @staticmethod
    def get_display_text(
        original_text: str,
        original_language: str,
        translations_cache: Dict[str, str],
        user_language: str
    ) -> str:
        """
        Get the appropriate text to display for a user based on their language preference.
        
        Args:
            original_text: Original message text
            original_language: Language of the original message
            translations_cache: Cached translations
            user_language: User's preferred language
            
        Returns:
            Text to display (original or translated)
        """
        # If user's language matches original language, return original
        if user_language == original_language:
            return original_text
        
        # If translation exists in cache, return it
        if user_language in translations_cache:
            return translations_cache[user_language]
        
        # Otherwise return original text
        return original_text
    
    @staticmethod
    def generate_missing_translation(
        message: Message,
        target_language: str,
        db: Session
    ) -> Optional[str]:
        """
        Generate a missing translation for a message and update the cache.
        
        Args:
            message: Message object to translate
            target_language: Target language for translation
            db: Database session
            
        Returns:
            Translated text or None if translation fails
        """
        # Check if translation already exists
        if message.translations_cache and target_language in message.translations_cache:
            return message.translations_cache[target_language]
        
        # Generate translation
        translated_text = translation_service.translate_text(
            message.original_text,
            message.original_language,
            target_language
        )
        
        if translated_text:
            # Update cache in database
            if not message.translations_cache:
                message.translations_cache = {}
            
            message.translations_cache[target_language] = translated_text
            db.commit()
            
            return translated_text
        
        return None
    
    @staticmethod
    def get_chatroom_languages(chatroom_id: int, db: Session) -> List[str]:
        """
        Get all languages used in a chatroom based on message history.
        
        Args:
            chatroom_id: ID of the chatroom
            db: Database session
            
        Returns:
            List of language codes used in the chatroom
        """
        languages = (
            db.query(Message.original_language)
            .filter(Message.chatroom_id == chatroom_id)
            .distinct()
            .all()
        )
        
        return [lang[0] for lang in languages if lang[0]]
    
    @staticmethod
    def get_translation_statistics(chatroom_id: int, db: Session) -> Dict[str, Any]:
        """
        Get translation statistics for a chatroom.
        
        Args:
            chatroom_id: ID of the chatroom
            db: Database session
            
        Returns:
            Dictionary with translation statistics
        """
        messages = db.query(Message).filter(Message.chatroom_id == chatroom_id).all()
        
        total_messages = len(messages)
        languages_used = set()
        messages_with_translations = 0
        translation_count = 0
        
        for msg in messages:
            if msg.original_language:
                languages_used.add(msg.original_language)
            
            if msg.translations_cache:
                messages_with_translations += 1
                translation_count += len(msg.translations_cache)
        
        return {
            "total_messages": total_messages,
            "languages_used": list(languages_used),
            "messages_with_translations": messages_with_translations,
            "total_translations": translation_count,
            "translation_coverage": (messages_with_translations / total_messages * 100) if total_messages > 0 else 0
        }


# Global chatroom service instance
chatroom_service = ChatroomService()
