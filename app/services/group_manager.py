# filepath: app/services/group_manager.py
import json
from typing import Dict, List, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class GroupManager:
    def __init__(self):
        # group_id -> set of user_ids
        self.group_connections: Dict[int, Set[int]] = {}
        # user_id -> websocket
        self.user_websockets: Dict[int, WebSocket] = {}
        # user_id -> set of group_ids they're connected to
        self.user_groups: Dict[int, Set[int]] = {}
        # typing indicators: group_id -> set of user_ids currently typing
        self.typing_users: Dict[int, Set[int]] = {}
        # connection tracking for duplicate prevention
        self.connection_timestamps: Dict[int, float] = {}
    
    async def connect_user_to_groups(self, user_id: int, websocket: WebSocket, group_ids: List[int]):
        """Connect a user to multiple groups with duplicate prevention"""
        import time
        
        current_time = time.time()
        
        # Check for duplicate connection (within 1 second)
        if user_id in self.connection_timestamps:
            if current_time - self.connection_timestamps[user_id] < 1.0:
                logger.warning(f"Preventing duplicate connection for user {user_id}")
                return False
        
        # Close existing connection if any
        if user_id in self.user_websockets:
            try:
                old_websocket = self.user_websockets[user_id]
                await old_websocket.close(code=1000, reason="New connection established")
                logger.info(f"Closed old connection for user {user_id}")
            except Exception as e:
                logger.error(f"Error closing old connection: {e}")
        
        # Set new connection
        self.user_websockets[user_id] = websocket
        self.connection_timestamps[user_id] = current_time
        
        if user_id not in self.user_groups:
            self.user_groups[user_id] = set()
        
        for group_id in group_ids:
            # Add user to group
            if group_id not in self.group_connections:
                self.group_connections[group_id] = set()
            self.group_connections[group_id].add(user_id)
            self.user_groups[user_id].add(group_id)
            
            logger.info(f"User {user_id} connected to group {group_id}")
        
        return True
    
    async def disconnect_user(self, user_id: int):
        """Disconnect user from all groups with cleanup"""
        if user_id in self.user_groups:
            for group_id in self.user_groups[user_id]:
                if group_id in self.group_connections:
                    self.group_connections[group_id].discard(user_id)
                    if not self.group_connections[group_id]:
                        del self.group_connections[group_id]
                
                # Remove from typing indicators
                if group_id in self.typing_users:
                    self.typing_users[group_id].discard(user_id)
                    if not self.typing_users[group_id]:
                        del self.typing_users[group_id]
            
            del self.user_groups[user_id]
        
        if user_id in self.user_websockets:
            del self.user_websockets[user_id]
        
        if user_id in self.connection_timestamps:
            del self.connection_timestamps[user_id]
            
        logger.info(f"User {user_id} disconnected from all groups")
    
    async def join_group(self, user_id: int, group_id: int):
        """Add user to a specific group (for dynamic joining)"""
        if user_id in self.user_websockets:
            if group_id not in self.group_connections:
                self.group_connections[group_id] = set()
            self.group_connections[group_id].add(user_id)
            
            if user_id not in self.user_groups:
                self.user_groups[user_id] = set()
            self.user_groups[user_id].add(group_id)
            
            logger.info(f"User {user_id} joined group {group_id}")
    
    async def leave_group(self, user_id: int, group_id: int):
        """Remove user from a specific group"""
        if group_id in self.group_connections:
            self.group_connections[group_id].discard(user_id)
            if not self.group_connections[group_id]:
                del self.group_connections[group_id]
        
        if user_id in self.user_groups:
            self.user_groups[user_id].discard(group_id)
        
        # Remove from typing indicators
        if group_id in self.typing_users:
            self.typing_users[group_id].discard(user_id)
            if not self.typing_users[group_id]:
                del self.typing_users[group_id]
        
        logger.info(f"User {user_id} left group {group_id}")
    
    async def broadcast_to_group(self, group_id: int, message_data: dict, exclude_user: int = None):
        """Broadcast message to all members of a group"""
        if group_id not in self.group_connections:
            return
        
        message_json = json.dumps(message_data)
        disconnected_users = []
        
        for user_id in self.group_connections[group_id]:
            if exclude_user and user_id == exclude_user:
                continue
                
            if user_id in self.user_websockets:
                try:
                    websocket = self.user_websockets[user_id]
                    await websocket.send_text(message_json)
                except Exception as e:
                    logger.error(f"Failed to send message to user {user_id} in group {group_id}: {e}")
                    disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect_user(user_id)
    
    async def broadcast_voice_message_to_group(self, group_id: int, message_data: dict, members_query, translations: dict, audio_urls: dict, exclude_user: int = None):
        """Enhanced broadcast for voice messages with member-specific audio and translations"""
        if group_id not in self.group_connections:
            logger.warning(f"Group {group_id} not found in connections")
            return
        
        logger.info(f"Broadcasting voice message to group {group_id}")
        logger.info(f"Connected users in group: {list(self.group_connections[group_id])}")
        logger.info(f"Available translations: {list(translations.keys())}")
        logger.info(f"Available audio URLs: {list(audio_urls.keys())}")
        
        disconnected_users = []
        
        for user_id in self.group_connections[group_id]:
            if exclude_user and user_id == exclude_user:
                continue
                
            if user_id in self.user_websockets:
                try:
                    # Find member's preferred languages
                    member_text_language = 'en'  # default
                    member_voice_language = 'en'  # default
                    for member in members_query:
                        if member.user_id == user_id:
                            member_text_language = member.preferred_language or 'en'
                            member_voice_language = member.voice_language or 'en'
                            break
                    
                    logger.info(f"👤 User {user_id} languages: text={member_text_language}, voice={member_voice_language}")
                    
                    # Create personalized message for this user
                    personalized_message = message_data.copy()
                    
                    # Set personalized text content
                    personalized_message["preferred_content"] = translations.get(member_text_language, message_data["content"])
                    personalized_message["user_language"] = member_text_language
                    
                    # Set personalized audio URL based on user's voice language preference
                    user_audio_url = None
                    if member_voice_language in audio_urls:
                        user_audio_url = audio_urls[member_voice_language]
                        logger.info(f"Using voice translation for user {user_id}: {member_voice_language}")
                    elif message_data["original_language"] in audio_urls:
                        user_audio_url = audio_urls[message_data["original_language"]]
                        logger.info(f"Using original voice for user {user_id}: {message_data['original_language']}")
                    else:
                        # Fallback to any available audio
                        available_audios = list(audio_urls.values())
                        if available_audios:
                            user_audio_url = available_audios[0]
                            logger.info(f"Using fallback audio for user {user_id}")
                    
                    # Update the voice file paths for this user
                    if user_audio_url:
                        personalized_message["voice_file"] = user_audio_url
                        # Create a personalized audio_urls dict with user's preferred audio as primary
                        personalized_audio_urls = audio_urls.copy()
                        personalized_message["audio_urls"] = personalized_audio_urls
                        personalized_message["voice_translations"] = personalized_audio_urls
                    
                    logger.info(f"Sending voice message to user {user_id}:")
                    logger.info(f"  - Text: '{personalized_message['preferred_content'][:30]}...'")
                    logger.info(f"  - Audio: {user_audio_url}")
                    
                    websocket = self.user_websockets[user_id]
                    message_json = json.dumps(personalized_message)
                    await websocket.send_text(message_json)
                    
                    logger.info(f"Voice message sent successfully to user {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error sending voice message to user {user_id}: {e}")
                    disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect_user(user_id)
    
    async def broadcast_to_group_enhanced(self, group_id: int, message_data: dict, members_query, translations: dict, exclude_user: int = None):
        """Enhanced broadcast with member-specific translations"""
        if group_id not in self.group_connections:
            logger.warning(f"Group {group_id} not found in connections")
            return
        
        logger.info(f"Broadcasting enhanced message to group {group_id}")
        logger.info(f"Connected users in group: {list(self.group_connections[group_id])}")
        logger.info(f"Available translations: {list(translations.keys())}")
        
        disconnected_users = []
        
        for user_id in self.group_connections[group_id]:
            if exclude_user and user_id == exclude_user:
                continue
                
            if user_id in self.user_websockets:
                try:
                    # Find member's preferred language
                    member_language = 'en'  # default
                    for member in members_query:
                        if member.user_id == user_id:
                            member_language = member.preferred_language or 'en'
                            break
                    
                    logger.info(f"👤 User {user_id} preferred language: {member_language}")
                    
                    # Create personalized message with member's preferred translation
                    personalized_message = message_data.copy()
                    personalized_message["preferred_content"] = translations.get(member_language, message_data["content"])
                    personalized_message["user_language"] = member_language
                    
                    logger.info(f"Sending to user {user_id}: original='{message_data['content'][:30]}...' preferred='{personalized_message['preferred_content'][:30]}...'")
                    
                    websocket = self.user_websockets[user_id]
                    message_json = json.dumps(personalized_message)
                    await websocket.send_text(message_json)
                    
                    logger.info(f"Message sent successfully to user {user_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to send enhanced message to user {user_id} in group {group_id}: {e}")
                    disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect_user(user_id)
            
        logger.info(f"Broadcast completed. Sent to {len(self.group_connections[group_id]) - len(disconnected_users)} users, {len(disconnected_users)} disconnected")
    
    async def notify_group(self, group_id: int, notification_data: dict):
        """Send notification to group members"""
        notification_data["type"] = "group_notification"
        notification_data["group_id"] = group_id
        await self.broadcast_to_group(group_id, notification_data)
    
    async def handle_typing(self, user_id: int, group_id: int, is_typing: bool, user_name: str):
        """Handle typing indicators"""
        if group_id not in self.typing_users:
            self.typing_users[group_id] = set()
        
        if is_typing:
            self.typing_users[group_id].add(user_id)
        else:
            self.typing_users[group_id].discard(user_id)
        
        # Broadcast typing update to other group members
        typing_data = {
            "type": "typing_update",
            "group_id": group_id,
            "user_id": user_id,
            "user_name": user_name,
            "is_typing": is_typing,
            "typing_users": [uid for uid in self.typing_users[group_id] if uid != user_id]
        }
        
        await self.broadcast_to_group(group_id, typing_data, exclude_user=user_id)
    
    def get_group_members(self, group_id: int) -> Set[int]:
        """Get list of currently connected members in a group"""
        return self.group_connections.get(group_id, set())
    
    def get_user_groups(self, user_id: int) -> Set[int]:
        """Get list of groups user is connected to"""
        return self.user_groups.get(user_id, set())
    
    def is_user_online(self, user_id: int) -> bool:
        """Check if user is online"""
        return user_id in self.user_websockets
    
    def get_online_members(self, group_id: int) -> Set[int]:
        """Get list of online members in a group"""
        if group_id not in self.group_connections:
            return set()
        return self.group_connections[group_id]

# Global group manager instance
group_manager = GroupManager()
