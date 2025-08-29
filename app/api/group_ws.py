# filepath: app/api/group_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import decode_access_token
from ..models.user import User
from ..models.group import group_members
from ..services.group_manager import group_manager
import logging
import json
import time
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/groups")
async def websocket_groups_endpoint(
    websocket: WebSocket, 
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for group communications"""
    await websocket.accept()
    user = None
    
    try:
        # Decode token and get user
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        user = db.query(User).filter(User.email == payload["sub"]).first()
        if not user:
            await websocket.close(code=1008, reason="User not found")
            return
        
        # Get user's groups
        user_groups = db.query(group_members.c.group_id).filter(
            group_members.c.user_id == user.id
        ).all()
        group_ids = [row[0] for row in user_groups]
        
        # Connect user to their groups with duplicate prevention
        connection_success = await group_manager.connect_user_to_groups(user.id, websocket, group_ids)
        
        if not connection_success:
            await websocket.close(code=1008, reason="Duplicate connection prevented")
            return
        
        logger.info(f"User {user.id} connected to groups WebSocket with groups: {group_ids}")
        
        # Send connection confirmation with enhanced status
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": user.id,
            "connected_groups": group_ids,
            "connection_id": f"conn_{user.id}_{int(time.time())}",
            "message": "Connected to groups successfully"
        }))
        
        # Send keep-alive ping every 30 seconds to maintain connection
        async def keep_alive():
            while True:
                try:
                    await asyncio.sleep(30)
                    await websocket.send_text(json.dumps({
                        "type": "keep_alive", 
                        "timestamp": int(time.time())
                    }))
                except Exception:
                    break
        
        # Start keep-alive task in background
        asyncio.create_task(keep_alive())
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                message_type = message.get("type")
                
                if message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
                elif message_type == "typing":
                    # Handle typing indicator
                    group_id = message.get("group_id")
                    is_typing = message.get("is_typing", True)
                    
                    if group_id and group_id in group_ids:
                        await group_manager.handle_typing(
                            user.id, 
                            group_id, 
                            is_typing, 
                            user.full_name or user.email
                        )
                
                elif message_type == "join_group":
                    # Handle dynamic group joining
                    group_id = message.get("group_id")
                    if group_id:
                        # Verify user is member of this group
                        membership = db.query(group_members).filter(
                            group_members.c.group_id == group_id,
                            group_members.c.user_id == user.id
                        ).first()
                        
                        if membership:
                            await group_manager.join_group(user.id, group_id)
                            group_ids.append(group_id)
                            
                            await websocket.send_text(json.dumps({
                                "type": "joined_group",
                                "group_id": group_id,
                                "message": "Successfully joined group"
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Not authorized to join this group"
                            }))
                
                elif message_type == "leave_group":
                    # Handle leaving a group
                    group_id = message.get("group_id")
                    if group_id and group_id in group_ids:
                        await group_manager.leave_group(user.id, group_id)
                        group_ids.remove(group_id)
                        
                        await websocket.send_text(json.dumps({
                            "type": "left_group",
                            "group_id": group_id,
                            "message": "Successfully left group"
                        }))
                
                elif message_type == "get_online_members":
                    # Get online members for a group
                    group_id = message.get("group_id")
                    if group_id and group_id in group_ids:
                        online_members = group_manager.get_online_members(group_id)
                        
                        # Get member details from database
                        if online_members:
                            members_data = db.query(User.id, User.full_name, User.email).filter(
                                User.id.in_(online_members)
                            ).all()
                            
                            members_list = [
                                {
                                    "id": member.id,
                                    "name": member.full_name or member.email,
                                    "email": member.email
                                }
                                for member in members_data
                            ]
                        else:
                            members_list = []
                        
                        await websocket.send_text(json.dumps({
                            "type": "online_members",
                            "group_id": group_id,
                            "members": members_list,
                            "count": len(members_list)
                        }))
                
                elif message_type == "request_group_info":
                    # Send group information
                    group_id = message.get("group_id")
                    if group_id and group_id in group_ids:
                        from ..models.group import Group
                        group = db.query(Group).filter(Group.id == group_id).first()
                        
                        if group:
                            await websocket.send_text(json.dumps({
                                "type": "group_info",
                                "group_id": group_id,
                                "name": group.name,
                                "description": group.description,
                                "member_count": group.get_member_count(db),
                                "online_count": len(group_manager.get_online_members(group_id))
                            }))
                
                else:
                    # Unknown message type
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }))
                
            except WebSocketDisconnect:
                logger.info(f"User {user.id if user else 'Unknown'} disconnected from groups WebSocket")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"User {user.id if user else 'Unknown'} disconnected from groups WebSocket")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if user:
            await group_manager.disconnect_user(user.id)
            logger.info(f"User {user.id} disconnected and cleaned up from groups WebSocket")
