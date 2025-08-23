# filepath: app/services/call_manager.py
import asyncio
import json
from typing import Dict, Optional, Any, List
from datetime import datetime
from fastapi import WebSocket
import logging

from ..models.voice_call import CallStatus

logger = logging.getLogger(__name__)

class CallManager:
    """Manages active voice calls and WebSocket connections"""
    
    def __init__(self):
        # Active calls mapping: call_id -> call_info
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
        # WebSocket connections: call_id -> {user_id -> websocket}
        self.call_connections: Dict[str, Dict[int, WebSocket]] = {}
        
        # Notification connections: user_id -> websocket (for dashboard notifications)
        self.notification_connections: Dict[int, WebSocket] = {}
        
        # User online status: user_id -> {websocket, last_seen}
        self.online_users: Dict[int, Dict[str, Any]] = {}
        
        # Call participants: call_id -> {caller_id, callee_id}
        self.call_participants: Dict[str, Dict[str, int]] = {}
        
        # WebRTC peer connections metadata
        self.peer_connections: Dict[str, Dict[str, Any]] = {}
        
    async def register_call(self, db_call_id: int, call_id: str, caller_id: int, callee_id: int):
        """Register a new call in the manager"""
        
        self.active_calls[call_id] = {
            'db_id': db_call_id,
            'call_id': call_id,
            'caller_id': caller_id,
            'callee_id': callee_id,
            'status': CallStatus.INITIATED,
            'started_at': datetime.utcnow(),
            'participants': [caller_id, callee_id]
        }
        
        self.call_participants[call_id] = {
            'caller_id': caller_id,
            'callee_id': callee_id
        }
        
        self.call_connections[call_id] = {}
        
        logger.info(f"📞 Registered call {call_id} between users {caller_id} and {callee_id}")
    
    async def add_call_connection(self, call_id: str, user_id: int, websocket: WebSocket):
        """Add a WebSocket connection for a call"""
        
        if call_id not in self.call_connections:
            self.call_connections[call_id] = {}
        
        self.call_connections[call_id][user_id] = websocket
        
        # Mark user as online
        self.online_users[user_id] = {
            'websocket': websocket,
            'last_seen': datetime.utcnow(),
            'in_call': call_id
        }
        
        logger.info(f"📞 Added WebSocket connection for user {user_id} in call {call_id}")
    
    async def remove_call_connection(self, call_id: str, user_id: int):
        """Remove a WebSocket connection from a call"""
        
        if call_id in self.call_connections:
            self.call_connections[call_id].pop(user_id, None)
            
            # Remove from online users if they were in this call
            if user_id in self.online_users and self.online_users[user_id].get('in_call') == call_id:
                self.online_users.pop(user_id, None)
            
            logger.info(f"📞 Removed WebSocket connection for user {user_id} from call {call_id}")
    
    async def cleanup_call(self, call_id: str):
        """Clean up a finished call"""
        
        # Remove from active calls
        self.active_calls.pop(call_id, None)
        
        # Remove participants
        self.call_participants.pop(call_id, None)
        
        # Remove peer connections
        self.peer_connections.pop(call_id, None)
        
        # Remove WebSocket connections
        if call_id in self.call_connections:
            for user_id in list(self.call_connections[call_id].keys()):
                await self.remove_call_connection(call_id, user_id)
            self.call_connections.pop(call_id, None)
        
        logger.info(f"📞 Cleaned up call {call_id}")
    
    async def is_user_online(self, user_id: int) -> bool:
        """Check if a user is currently online using presence manager"""
        try:
            # Import here to avoid circular imports
            from .presence_manager import presence_manager
            
            # Check if user is online via presence manager (chat connections)
            online_users = presence_manager.get_online_users()
            is_online_via_chat = user_id in online_users
            
            # Check if user has voice call WebSocket connection
            is_online_via_voice = user_id in self.online_users
            if is_online_via_voice:
                user_info = self.online_users[user_id]
                last_seen = user_info.get('last_seen', datetime.utcnow())
                
                # Consider user offline if last seen more than 30 seconds ago
                if (datetime.utcnow() - last_seen).total_seconds() > 30:
                    self.online_users.pop(user_id, None)
                    is_online_via_voice = False
            
            # User is online if they have either chat or voice call connection
            return is_online_via_chat or is_online_via_voice
            
        except Exception as e:
            logger.error(f"Error checking user online status: {e}")
            # Fallback to voice call manager only
            return user_id in self.online_users
    
    async def is_user_in_call(self, user_id: int) -> Optional[str]:
        """Check if user is currently in a call, return call_id if yes"""
        
        if user_id in self.online_users:
            return self.online_users[user_id].get('in_call')
        return None
    
    async def send_call_notification(self, call_id: str, caller_id: int, callee_id: int, caller_name: str):
        """Send call notification to callee via available WebSocket connections"""
        
        notification = {
            'type': 'incoming_call',
            'call_id': call_id,
            'caller_id': caller_id,
            'caller_name': caller_name,
            'call_type': 'voice',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        success = False
        
        # Try notification WebSocket first
        if callee_id in self.notification_connections:
            try:
                websocket = self.notification_connections[callee_id]
                await websocket.send_text(json.dumps(notification))
                success = True
                logger.info(f"📞 Sent call notification to user {callee_id} via notification WebSocket")
            except Exception as e:
                logger.error(f"📞 Failed to send notification via notification WebSocket: {e}")
                # Remove dead connection
                self.notification_connections.pop(callee_id, None)
        
        # Try voice call WebSocket if available
        if not success and callee_id in self.online_users:
            try:
                websocket = self.online_users[callee_id]['websocket']
                await websocket.send_text(json.dumps(notification))
                success = True
                logger.info(f"📞 Sent call notification to user {callee_id} via voice call WebSocket")
            except Exception as e:
                logger.error(f"📞 Failed to send notification via voice call WebSocket: {e}")
                # Remove dead connection
                self.online_users.pop(callee_id, None)
        
        if success:
            # Update call status to ringing
            if call_id in self.active_calls:
                self.active_calls[call_id]['status'] = CallStatus.RINGING
        else:
            logger.warning(f"📞 Could not send call notification to user {callee_id} - no active WebSocket connections")
        
        return success
    
    async def send_call_answered(self, call_id: str, caller_id: int, sdp_answer: Optional[str] = None):
        """Send call answered notification to caller"""
        
        message = {
            'type': 'call_answered',
            'call_id': call_id,
            'sdp_answer': sdp_answer,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, caller_id, message)
        
        # Update call status
        if call_id in self.active_calls:
            self.active_calls[call_id]['status'] = CallStatus.ANSWERED
            self.active_calls[call_id]['answered_at'] = datetime.utcnow()
        
        logger.info(f"📞 Sent call answered notification for call {call_id}")
    
    async def send_call_initiated(self, call_id: str, caller_id: int):
        """Send call initiated notification to caller to open their call window"""
        
        message = {
            'type': 'call_initiated',
            'call_id': call_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to caller via notification WebSocket
        if caller_id in self.notification_connections:
            try:
                websocket = self.notification_connections[caller_id]
                await websocket.send_text(json.dumps(message))
                logger.info(f"📞 Sent call_initiated message to user {caller_id}")
            except Exception as e:
                logger.error(f"📞 Failed to send call_initiated message: {e}")
                # Remove dead connection
                self.notification_connections.pop(caller_id, None)
        else:
            logger.warning(f"📞 Could not send call_initiated to user {caller_id} - no notification connection")
    
    async def send_call_declined(self, call_id: str, caller_id: int):
        """Send call declined notification to caller"""
        
        message = {
            'type': 'call_declined',
            'call_id': call_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, caller_id, message)
        
        # Update call status
        if call_id in self.active_calls:
            self.active_calls[call_id]['status'] = CallStatus.DECLINED
        
        logger.info(f"📞 Sent call declined notification for call {call_id}")
    
    async def send_call_ended(self, call_id: str, user_id: int):
        """Send call ended notification to user"""
        
        message = {
            'type': 'call_ended',
            'call_id': call_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, user_id, message)
        
        # Update call status
        if call_id in self.active_calls:
            self.active_calls[call_id]['status'] = CallStatus.ENDED
        
        logger.info(f"📞 Sent call ended notification for call {call_id}")
    
    async def send_call_message(self, call_id: str, message_data: Dict[str, Any]):
        """Send a message to all participants in a call"""
        
        if call_id not in self.call_connections:
            logger.warning(f"📞 No connections found for call {call_id}")
            return
        
        message = {
            'type': 'call_message',
            'call_id': call_id,
            **message_data
        }
        
        # Send to all connected participants
        for user_id, websocket in self.call_connections[call_id].items():
            # Don't send message back to sender
            if user_id != message_data.get('sender_id'):
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"❌ Failed to send message to user {user_id} in call {call_id}: {e}")
                    # Remove stale connection
                    await self.remove_call_connection(call_id, user_id)
        
        logger.info(f"📞 Sent message in call {call_id}")
    
    async def forward_signaling_message(self, call_id: str, sender_id: int, signaling_data: Dict[str, Any]):
        """Forward WebRTC signaling messages between participants"""
        
        if call_id not in self.call_participants:
            logger.warning(f"📞 Call {call_id} not found for signaling")
            return
        
        participants = self.call_participants[call_id]
        receiver_id = participants['callee_id'] if sender_id == participants['caller_id'] else participants['caller_id']
        
        message = {
            'type': 'webrtc_signaling',
            'call_id': call_id,
            'sender_id': sender_id,
            'signaling_type': signaling_data.get('signaling_type'),
            'sdp': signaling_data.get('sdp'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, receiver_id, message)
        logger.info(f"📞 Forwarded signaling message in call {call_id}")
    
    async def forward_ice_candidate(self, call_id: str, sender_id: int, ice_data: Dict[str, Any]):
        """Forward ICE candidates between participants"""
        
        if call_id not in self.call_participants:
            logger.warning(f"📞 Call {call_id} not found for ICE candidate")
            return
        
        participants = self.call_participants[call_id]
        receiver_id = participants['callee_id'] if sender_id == participants['caller_id'] else participants['caller_id']
        
        message = {
            'type': 'ice_candidate',
            'call_id': call_id,
            'sender_id': sender_id,
            'candidate': ice_data.get('candidate'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, receiver_id, message)
        logger.info(f"📞 Forwarded ICE candidate in call {call_id}")
    
    async def forward_translation_message(self, call_id: str, sender_id: int, translation_data: Dict[str, Any]):
        """Forward translation-related messages between participants"""
        
        if call_id not in self.call_participants:
            logger.warning(f"📞 Call {call_id} not found for translation message")
            return
        
        participants = self.call_participants[call_id]
        receiver_id = participants['callee_id'] if sender_id == participants['caller_id'] else participants['caller_id']
        
        # Forward the translation message as-is but add metadata
        message = {
            **translation_data,
            'call_id': call_id,
            'sender_id': sender_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._send_to_user_in_call(call_id, receiver_id, message)
        logger.info(f"🌐 Forwarded translation message of type '{translation_data.get('type')}' in call {call_id}")
    
    async def broadcast_call_status(self, call_id: str, sender_id: int, status_data: Dict[str, Any]):
        """Broadcast call status changes (mute, unmute, etc.)"""
        
        if call_id not in self.call_connections:
            return
        
        message = {
            'type': 'call_status_update',
            'call_id': call_id,
            'sender_id': sender_id,
            'status': status_data.get('status'),
            'data': status_data.get('data', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all participants except sender
        for user_id, websocket in self.call_connections[call_id].items():
            if user_id != sender_id:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"❌ Failed to broadcast status to user {user_id}: {e}")
                    await self.remove_call_connection(call_id, user_id)
        
        logger.info(f"📞 Broadcasted call status in call {call_id}")
    
    async def _send_to_user_in_call(self, call_id: str, user_id: int, message: Dict[str, Any]):
        """Send a message to a specific user in a call"""
        
        # First try call-specific connection
        if call_id in self.call_connections and user_id in self.call_connections[call_id]:
            try:
                websocket = self.call_connections[call_id][user_id]
                await websocket.send_text(json.dumps(message))
                logger.info(f"📞 Sent message to user {user_id} via call connection")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to send to user {user_id} via call connection: {e}")
                await self.remove_call_connection(call_id, user_id)
        
        # Try notification connection  
        if user_id in self.notification_connections:
            try:
                websocket = self.notification_connections[user_id]
                await websocket.send_text(json.dumps(message))
                logger.info(f"📞 Sent message to user {user_id} via notification connection")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to send to user {user_id} via notification connection: {e}")
                self.notification_connections.pop(user_id, None)
        
        # Fallback to general user connection
        if user_id in self.online_users:
            try:
                websocket = self.online_users[user_id]['websocket']
                await websocket.send_text(json.dumps(message))
                logger.info(f"📞 Sent message to user {user_id} via general connection")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to send to user {user_id} via general connection: {e}")
                self.online_users.pop(user_id, None)
        
        logger.warning(f"📞 No connection found for user {user_id}")
        return False
    
    async def get_active_calls(self) -> List[Dict[str, Any]]:
        """Get list of all active calls"""
        return list(self.active_calls.values())
    
    async def get_call_info(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific call"""
        return self.active_calls.get(call_id)
    
    async def update_user_last_seen(self, user_id: int):
        """Update user's last seen timestamp"""
        if user_id in self.online_users:
            self.online_users[user_id]['last_seen'] = datetime.utcnow()
    
    async def get_online_users(self) -> List[int]:
        """Get list of currently online user IDs"""
        current_time = datetime.utcnow()
        online_users = []
        
        for user_id, user_info in list(self.online_users.items()):
            last_seen = user_info.get('last_seen', current_time)
            if (current_time - last_seen).total_seconds() <= 30:
                online_users.append(user_id)
            else:
                # Remove offline users
                self.online_users.pop(user_id, None)
        
        return online_users
    
    async def heartbeat_check(self):
        """Periodic heartbeat check to clean up stale connections"""
        current_time = datetime.utcnow()
        
        # Clean up offline users
        offline_users = []
        for user_id, user_info in self.online_users.items():
            last_seen = user_info.get('last_seen', current_time)
            if (current_time - last_seen).total_seconds() > 60:  # 1 minute timeout
                offline_users.append(user_id)
        
        for user_id in offline_users:
            self.online_users.pop(user_id, None)
            logger.info(f"📞 Removed offline user {user_id}")
        
        # Clean up stale calls (longer than 4 hours)
        stale_calls = []
        for call_id, call_info in self.active_calls.items():
            started_at = call_info.get('started_at', current_time)
            if (current_time - started_at).total_seconds() > 14400:  # 4 hours
                stale_calls.append(call_id)
        
        for call_id in stale_calls:
            await self.cleanup_call(call_id)
            logger.info(f"📞 Cleaned up stale call {call_id}")
    
    async def add_notification_connection(self, user_id: int, websocket: WebSocket):
        """Add a notification WebSocket connection for a user"""
        self.notification_connections[user_id] = websocket
        logger.info(f"📞 Added notification connection for user {user_id}")
    
    async def remove_notification_connection(self, user_id: int):
        """Remove a notification WebSocket connection for a user"""
        self.notification_connections.pop(user_id, None)
        logger.info(f"📞 Removed notification connection for user {user_id}")
    
    async def send_notification_to_user(self, user_id: int, notification: Dict[str, Any]):
        """Send a notification to a specific user via their notification WebSocket"""
        if user_id in self.notification_connections:
            try:
                websocket = self.notification_connections[user_id]
                await websocket.send_text(json.dumps(notification))
                return True
            except Exception as e:
                logger.error(f"📞 Failed to send notification to user {user_id}: {e}")
                # Remove dead connection
                self.notification_connections.pop(user_id, None)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get call manager statistics"""
        return {
            'active_calls': len(self.active_calls),
            'online_users': len(self.online_users),
            'total_connections': sum(len(connections) for connections in self.call_connections.values()),
            'call_connections': len(self.call_connections)
        }

# Global call manager instance
call_manager = CallManager()

# Background task to run heartbeat check
async def start_heartbeat_task():
    """Start the heartbeat cleanup task"""
    while True:
        try:
            await call_manager.heartbeat_check()
            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"❌ Heartbeat check error: {e}")
            await asyncio.sleep(60)  # Wait longer on error
