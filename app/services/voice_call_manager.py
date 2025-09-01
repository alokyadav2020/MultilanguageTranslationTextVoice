import json
import logging
import uuid
import time
from typing import Dict, Optional
from fastapi import WebSocket
from ..services.whisper_translation_service import whisper_translation_service

logger = logging.getLogger(__name__)

class CallParticipant:
    def __init__(self, user_id: int, websocket: WebSocket, language: str = "en"):
        self.user_id = user_id
        self.websocket = websocket
        self.language = language
        self.connected_at = time.time()
        self.translation_enabled = True
        
class CallSession:
    def __init__(self, call_id: str, initiator_id: int):
        self.call_id = call_id
        self.initiator_id = initiator_id
        self.participants: Dict[int, CallParticipant] = {}
        self.created_at = time.time()
        self.is_active = True
        self.translation_pairs: Dict[int, str] = {}  # user_id -> target_language
        
    def add_participant(self, user_id: int, websocket: WebSocket, language: str = "en"):
        """Add participant to call session"""
        participant = CallParticipant(user_id, websocket, language)
        self.participants[user_id] = participant
        logger.info(f"Added participant {user_id} to call {self.call_id} with language {language}")
        
    def remove_participant(self, user_id: int):
        """Remove participant from call session"""
        if user_id in self.participants:
            del self.participants[user_id]
            logger.info(f"Removed participant {user_id} from call {self.call_id}")
            
    def get_participant_count(self) -> int:
        """Get number of active participants"""
        return len(self.participants)
        
    def get_other_participants(self, user_id: int) -> Dict[int, CallParticipant]:
        """Get all participants except the specified user"""
        return {pid: p for pid, p in self.participants.items() if pid != user_id}
        
    def set_translation_target(self, user_id: int, target_language: str):
        """Set target language for user's translation"""
        if user_id in self.participants:
            self.translation_pairs[user_id] = target_language
            logger.info(f"Set translation target for user {user_id}: {target_language}")

class VoiceCallManager:
    def __init__(self):
        self.active_calls: Dict[str, CallSession] = {}
        self.user_calls: Dict[int, str] = {}  # user_id -> call_id
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 300  # 5 minutes
        
    async def create_call(self, initiator_id: int, target_user_id: int = None) -> str:
        """Create a new voice call session"""
        call_id = str(uuid.uuid4())
        call_session = CallSession(call_id, initiator_id)
        
        self.active_calls[call_id] = call_session
        self.user_calls[initiator_id] = call_id
        
        logger.info(f"Created call {call_id} by user {initiator_id}")
        return call_id
        
    async def join_call(self, call_id: str, user_id: int, websocket: WebSocket, language: str = "en") -> bool:
        """Join an existing call"""
        if call_id not in self.active_calls:
            logger.warning(f"Call {call_id} not found")
            return False
            
        call_session = self.active_calls[call_id]
        
        # Check if user is already in another call
        if user_id in self.user_calls and self.user_calls[user_id] != call_id:
            await self.leave_call(self.user_calls[user_id], user_id)
            
        call_session.add_participant(user_id, websocket, language)
        self.user_calls[user_id] = call_id
        
        # Notify other participants about new user
        await self._broadcast_participant_joined(call_id, user_id)
        
        return True
        
    async def leave_call(self, call_id: str, user_id: int):
        """Leave a call"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        call_session.remove_participant(user_id)
        
        if user_id in self.user_calls:
            del self.user_calls[user_id]
            
        # Clean up translation buffers
        whisper_translation_service.cleanup_call_buffers(call_id)
        
        # Notify other participants
        await self._broadcast_participant_left(call_id, user_id)
        
        # Clean up empty calls
        if call_session.get_participant_count() == 0:
            del self.active_calls[call_id]
            logger.info(f"Cleaned up empty call {call_id}")
            
    async def handle_webrtc_offer(self, call_id: str, user_id: int, message: dict):
        """Handle WebRTC offer and forward to other participants"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        other_participants = call_session.get_other_participants(user_id)
        
        for participant_id, participant in other_participants.items():
            try:
                await participant.websocket.send_text(json.dumps({
                    "type": "webrtc_offer",
                    "from_user": user_id,
                    "sdp": message.get("sdp"),
                    "call_id": call_id
                }))
            except Exception as e:
                logger.error(f"Failed to send WebRTC offer to {participant_id}: {e}")
                
    async def handle_webrtc_answer(self, call_id: str, user_id: int, message: dict):
        """Handle WebRTC answer and forward to other participants"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        other_participants = call_session.get_other_participants(user_id)
        
        for participant_id, participant in other_participants.items():
            try:
                await participant.websocket.send_text(json.dumps({
                    "type": "webrtc_answer",
                    "from_user": user_id,
                    "sdp": message.get("sdp"),
                    "call_id": call_id
                }))
            except Exception as e:
                logger.error(f"Failed to send WebRTC answer to {participant_id}: {e}")
                
    async def handle_ice_candidate(self, call_id: str, user_id: int, message: dict):
        """Handle ICE candidate and forward to other participants"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        other_participants = call_session.get_other_participants(user_id)
        
        for participant_id, participant in other_participants.items():
            try:
                await participant.websocket.send_text(json.dumps({
                    "type": "ice_candidate",
                    "from_user": user_id,
                    "candidate": message.get("candidate"),
                    "call_id": call_id
                }))
            except Exception as e:
                logger.error(f"Failed to send ICE candidate to {participant_id}: {e}")
                
    async def handle_voice_translation(self, call_id: str, user_id: int, message: dict):
        """Handle real-time voice translation"""
        if call_id not in self.active_calls:
            logger.warning(f"Call {call_id} not found for voice translation")
            return
            
        call_session = self.active_calls[call_id]
        if user_id not in call_session.participants:
            logger.warning(f"User {user_id} not in call {call_id}")
            return
            
        source_language = message.get("language", "en")
        audio_data = message.get("audio_data")
        
        if not audio_data:
            logger.warning("No audio data received for translation")
            return
            
        # Get other participants and their target languages
        other_participants = call_session.get_other_participants(user_id)
        
        for participant_id, participant in other_participants.items():
            target_language = participant.language
            
            # Skip if same language
            if source_language == target_language:
                continue
                
            # Only translate if participant has translation enabled
            if not participant.translation_enabled:
                continue
                
            try:
                # Process voice translation
                result = await whisper_translation_service.process_voice_chunk_realtime(
                    call_id=call_id,
                    user_id=user_id,
                    audio_data=audio_data,
                    source_language=source_language,
                    target_language=target_language
                )
                
                if result["success"] and "audio_output" in result:
                    # Send translated audio to participant
                    translation_message = {
                        "type": "voice_translation",
                        "from_user": user_id,
                        "audio_data": result["audio_output"],
                        "source_language": source_language,
                        "target_language": target_language,
                        "timestamp": result.get("timestamp", time.time())
                    }
                    
                    await participant.websocket.send_text(json.dumps(translation_message))
                    
                elif result["status"] == "buffering":
                    # Still buffering, no action needed
                    pass
                else:
                    logger.error(f"Translation failed for {user_id} -> {participant_id}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Voice translation error for {user_id} -> {participant_id}: {e}")
                
    async def update_translation_settings(self, call_id: str, user_id: int, settings: dict):
        """Update translation settings for a user"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        if user_id not in call_session.participants:
            return
            
        participant = call_session.participants[user_id]
        
        # Update language preference
        if "language" in settings:
            participant.language = settings["language"]
            
        # Update translation enabled/disabled
        if "translation_enabled" in settings:
            participant.translation_enabled = settings["translation_enabled"]
            
        logger.info(f"Updated translation settings for user {user_id} in call {call_id}: {settings}")
        
    async def _broadcast_participant_joined(self, call_id: str, user_id: int):
        """Broadcast participant joined message"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        other_participants = call_session.get_other_participants(user_id)
        
        message = {
            "type": "participant_joined",
            "user_id": user_id,
            "call_id": call_id,
            "participant_count": call_session.get_participant_count()
        }
        
        for participant_id, participant in other_participants.items():
            try:
                await participant.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast participant joined to {participant_id}: {e}")
                
    async def _broadcast_participant_left(self, call_id: str, user_id: int):
        """Broadcast participant left message"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        
        message = {
            "type": "participant_left",
            "user_id": user_id,
            "call_id": call_id,
            "participant_count": call_session.get_participant_count()
        }
        
        for participant_id, participant in call_session.participants.items():
            try:
                await participant.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast participant left to {participant_id}: {e}")
                
    async def end_call(self, call_id: str, user_id: int):
        """End a call (only initiator can end)"""
        if call_id not in self.active_calls:
            return
            
        call_session = self.active_calls[call_id]
        
        # Only initiator can end the call
        if user_id != call_session.initiator_id:
            logger.warning(f"User {user_id} tried to end call {call_id} but is not initiator")
            return
            
        # Notify all participants
        message = {
            "type": "call_ended",
            "call_id": call_id,
            "ended_by": user_id,
            "reason": "ended_by_initiator"
        }
        
        for participant_id, participant in call_session.participants.items():
            try:
                await participant.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to notify participant {participant_id} of call end: {e}")
                
        # Clean up call
        participant_ids = list(call_session.participants.keys())
        for participant_id in participant_ids:
            await self.leave_call(call_id, participant_id)
            
    def get_call_session(self, call_id: str) -> Optional[CallSession]:
        """Get call session by ID"""
        return self.active_calls.get(call_id)
        
    def get_user_call(self, user_id: int) -> Optional[str]:
        """Get active call ID for user"""
        return self.user_calls.get(user_id)
        
    def get_active_calls_count(self) -> int:
        """Get number of active calls"""
        return len(self.active_calls)
        
    def get_stats(self) -> Dict:
        """Get call manager statistics"""
        total_participants = sum(len(call.participants) for call in self.active_calls.values())
        
        return {
            "active_calls": len(self.active_calls),
            "total_participants": total_participants,
            "translation_service_available": whisper_translation_service.is_available,
            "supported_languages": whisper_translation_service.whisper_languages
        }

# Global call manager instance
voice_call_manager = VoiceCallManager()
