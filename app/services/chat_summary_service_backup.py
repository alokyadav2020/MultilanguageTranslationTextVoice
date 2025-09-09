"""
Chat Summary Service using Llama 3.1 8B Quantized Model
Provides AI-powered chat summarization functionality for the translation app
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import platform

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class ChatSummaryService:
    """Service for generating AI-powered chat summaries using local Llama model"""
    
    def __init__(self):
        self.model_path = project_root / "artifacts" / "models" / "llama3.1-8b-instruct-q4" / "llama-3.1-8b-instruct-q4_k_m.gguf"
        self.model = None
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_model_loaded = False
        
        # GPU Configuration
        self.gpu_available = False
        self.gpu_layers = 0
        self.gpu_memory = 0
        self.gpu_info = {}
        
        # Detect GPU availability at initialization
        self._detect_gpu_capabilities()
        
        # Language mapping for summary generation
        self.language_codes = {
            'en': 'English',
            'ar': 'Arabic',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese'
        }
    
    def _detect_gpu_capabilities(self):
        """Detect GPU availability and capabilities"""
        try:
            # Check for CUDA availability
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device)
                    gpu_memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                    
                    self.gpu_available = True
                    self.gpu_info = {
                        'type': 'CUDA',
                        'count': gpu_count,
                        'current_device': current_device,
                        'name': gpu_name,
                        'memory_gb': round(gpu_memory_gb, 2),
                        'memory_bytes': torch.cuda.get_device_properties(current_device).total_memory
                    }
                    
                    # Calculate optimal GPU layers based on available memory
                    # Rule of thumb: ~100MB per layer for 8B model
                    available_memory_gb = gpu_memory_gb * 0.8  # Use 80% of GPU memory
                    estimated_layers = min(32, int(available_memory_gb * 10))  # Conservative estimate
                    self.gpu_layers = max(0, estimated_layers)
                    
                    logger.info(f"🎮 CUDA GPU detected: {gpu_name}")
                    logger.info(f"🎮 GPU Memory: {gpu_memory_gb:.2f} GB")
                    logger.info(f"🎮 GPU Layers to use: {self.gpu_layers}")
                    
                else:
                    logger.info("🖥️  CUDA not available, using CPU")
                    
            except ImportError:
                logger.info("🖥️  PyTorch not available, checking other GPU options")
            
            # Check for OpenCL (AMD/Intel GPUs)
            if not self.gpu_available:
                try:
                    import pyopencl as cl
                    platforms = cl.get_platforms()
                    if platforms:
                        for platform in platforms:
                            devices = platform.get_devices(device_type=cl.device_type.GPU)
                            if devices:
                                device = devices[0]
                                self.gpu_available = True
                                self.gpu_info = {
                                    'type': 'OpenCL',
                                    'platform': platform.name.strip(),
                                    'device': device.name.strip(),
                                    'memory_gb': round(device.global_mem_size / (1024**3), 2)
                                }
                                # OpenCL support in llama-cpp-python is limited, use fewer layers
                                self.gpu_layers = min(16, int(device.global_mem_size / (1024**3) * 4))
                                logger.info(f"🎮 OpenCL GPU detected: {device.name.strip()}")
                                break
                except ImportError:
                    logger.info("🖥️  OpenCL not available")
            
            # Check for Apple Metal (Mac M1/M2)
            if not self.gpu_available and platform.system() == 'Darwin':
                try:
                    import torch
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        self.gpu_available = True
                        self.gpu_info = {
                            'type': 'Metal',
                            'device': 'Apple Silicon GPU'
                        }
                        self.gpu_layers = 20  # Conservative for Apple Silicon
                        logger.info("🎮 Apple Metal GPU detected")
                except Exception:
                    pass
            
            if not self.gpu_available:
                logger.info("🖥️  No GPU acceleration available, using CPU only")
                self.gpu_layers = 0
                
        except Exception as e:
            logger.warning(f"⚠️  GPU detection failed: {e}")
            self.gpu_available = False
            self.gpu_layers = 0
    
    def _load_model(self) -> bool:
        """Load the Llama model (thread-safe)"""
        if self.is_model_loaded and self.model is not None:
            return True
            
        try:
            with self.model_lock:
                if self.is_model_loaded and self.model is not None:
                    return True
                
                # Check if model file exists
                if not self.model_path.exists():
                    logger.error(f"Model file not found: {self.model_path}")
                    logger.info("Please run: python download_llama_model.py")
                    return False
                
                # Import llama-cpp-python
                try:
                    from llama_cpp import Llama
                except ImportError:
                    logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
                    return False
                
                logger.info("🔄 Loading Llama 3.1 8B model...")
                
                # Determine optimal settings based on GPU availability
                if self.gpu_available and self.gpu_layers > 0:
                    logger.info(f"🎮 Loading model with GPU acceleration ({self.gpu_layers} layers)")
                    gpu_layers = self.gpu_layers
                    n_threads = 2  # Fewer CPU threads when using GPU
                else:
                    logger.info("🖥️  Loading model with CPU-only mode")
                    gpu_layers = 0
                    n_threads = min(8, max(4, threading.active_count()))  # Optimize CPU threads
                
                # Load model with optimized settings
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=4096,  # Context window
                    n_threads=n_threads,  # CPU threads (reduced if GPU is used)
                    n_gpu_layers=gpu_layers,  # GPU layers
                    verbose=False,
                    use_mmap=True,  # Memory mapping for efficiency
                    use_mlock=False,  # Don't lock memory
                    n_batch=512,  # Batch size
                    f16_kv=True,  # Use 16-bit for key-value cache (saves memory)
                    logits_all=False,  # Only compute logits for the last token
                    vocab_only=False,  # Load full model
                    offload_kqv=self.gpu_available,  # Offload KV cache to GPU if available
                )
                
                self.is_model_loaded = True
                logger.info("✅ Llama model loaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _generate_summary_sync(self, chat_data: Dict[str, Any]) -> str:
        """Generate summary synchronously (runs in thread pool)"""
        try:
            if not self._load_model():
                return "❌ Model not available. Please check model installation."
            
            messages = chat_data.get('messages', [])
            user_language = chat_data.get('language', 'en')
            chat_type = chat_data.get('chat_type', 'direct')
            participants = chat_data.get('participants', [])
            
            # Format chat messages for the model
            chat_text = self._format_chat_for_model(messages)
            
            # Create prompt based on language and chat type
            prompt = self._create_summary_prompt(
                chat_text, 
                user_language, 
                chat_type, 
                participants
            )
            
            # Generate summary
            response = self.model(
                prompt,
                max_tokens=800,  # Increased for detailed summary
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "\n\nUser:", "\n\nHuman:"],
                echo=False
            )
            
            summary_text = response['choices'][0]['text'].strip()
            
            # Clean up the response
            summary_text = self._clean_summary_response(summary_text)
            
            return summary_text
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return f"❌ Failed to generate summary: {str(e)}"
    
    def _format_chat_for_model(self, messages: List[Dict[str, Any]]) -> str:
        """Format chat messages for model input"""
        formatted_lines = []
        
        for msg in messages:
            timestamp = msg.get('timestamp', '')
            sender = msg.get('sender_name', msg.get('sender', 'Unknown'))
            message_type = msg.get('message_type', 'text')
            original_text = msg.get('original_text', msg.get('text', ''))
            translated_text = msg.get('translated_text', '')
            language = msg.get('original_language', 'unknown')
            
            # Format timestamp if available
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    time_str = f"[{dt.strftime('%H:%M')}] "
                except Exception:
                    time_str = f"[{timestamp}] "
            
            # Handle different message types
            if message_type == 'voice':
                duration = msg.get('audio_duration', 0)
                content = f"🎤 Voice message ({duration}s): {original_text}"
                if translated_text and translated_text != original_text:
                    content += f" [Translated: {translated_text}]"
            else:
                content = original_text
                if translated_text and translated_text != original_text:
                    content += f" [Translated: {translated_text}]"
            
            # Add language info if available
            if language and language != 'unknown':
                lang_info = f" ({language})"
            else:
                lang_info = ""
            
            formatted_lines.append(f"{time_str}{sender}{lang_info}: {content}")
        
        return "\n".join(formatted_lines)
    
    def _create_summary_prompt(
        self, 
        chat_text: str, 
        user_language: str, 
        chat_type: str,
        participants: List[str]
    ) -> str:
        """Create a comprehensive prompt for chat summarization"""
        
        language_name = self.language_codes.get(user_language, 'English')
        participant_list = ", ".join(participants) if participants else "Multiple users"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert chat conversation analyst. Your task is to create a comprehensive summary of the conversation in {language_name}. 

Key requirements:
1. Summarize in {language_name} language
2. Be objective and factual
3. Include key topics, decisions, and action items
4. Note language patterns and translation usage
5. Maintain professional tone
6. Keep summary concise but informative

Conversation details:
- Type: {chat_type.title()} chat
- Participants: {participant_list}
- Language for summary: {language_name}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please analyze and summarize this conversation:

{chat_text}

Create a structured summary that includes:

## 📋 Chat Summary

### 🎯 Main Topics
[List the key subjects discussed]

### 👥 Participation
[Analyze who participated and how actively]

### 💬 Communication Pattern  
[Note language usage, voice vs text messages, translation patterns]

### ✅ Key Points & Decisions
[Important conclusions, agreements, or decisions made]

### 📝 Action Items
[Any follow-up tasks or commitments mentioned]

### 🌍 Language Usage
[Languages used and any translation patterns observed]

### 📊 Summary Statistics
[Brief stats about message types, timing, engagement]

### 💡 Overall Summary
[2-3 sentence concise summary of the entire conversation]

Please provide the summary in {language_name}:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return prompt
    
    def _clean_summary_response(self, summary_text: str) -> str:
        """Clean and format the AI response"""
        # Remove any unwanted tokens or artifacts
        summary_text = re.sub(r'<\|.*?\|>', '', summary_text)
        summary_text = re.sub(r'\[INST\].*?\[/INST\]', '', summary_text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        summary_text = re.sub(r'\n{3,}', '\n\n', summary_text)
        summary_text = summary_text.strip()
        
        # Ensure we have content
        if len(summary_text) < 50:
            return "❌ Summary generation produced insufficient content. Please try again."
        
        return summary_text
    
    async def generate_chat_summary(
        self, 
        messages: List[Dict[str, Any]], 
        user_language: str = 'en',
        chat_type: str = 'direct',
        participants: List[str] = None,
        user_id: int = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive chat summary asynchronously
        
        Args:
            messages: List of chat messages
            user_language: User's preferred language for summary
            chat_type: Type of chat ('direct' or 'group')
            participants: List of participant names
            user_id: Current user's ID
            
        Returns:
            Dict containing summary data and statistics
        """
        try:
            if not messages:
                return {
                    "success": False,
                    "error": "No messages to summarize",
                    "summary": "No conversation history available.",
                    "statistics": {}
                }
            
            # Prepare data for model
            chat_data = {
                'messages': messages,
                'language': user_language,
                'chat_type': chat_type,
                'participants': participants or []
            }
            
            # Generate summary in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            summary_text = await loop.run_in_executor(
                self.executor, 
                self._generate_summary_sync, 
                chat_data
            )
            
            # Calculate statistics
            stats = self._calculate_statistics(messages, user_language)
            
            # Prepare response
            result = {
                "success": True,
                "summary": summary_text,
                "statistics": stats,
                "generated_at": datetime.now().isoformat(),
                "language": user_language,
                "chat_type": chat_type,
                "participants": participants or [],
                "message_count": len(messages),
                "user_id": user_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Chat summary generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"❌ Summary generation failed: {str(e)}",
                "statistics": self._calculate_statistics(messages, user_language),
                "generated_at": datetime.now().isoformat(),
                "language": user_language,
                "chat_type": chat_type,
                "participants": participants or [],
                "message_count": len(messages),
                "user_id": user_id
            }
    
    def _calculate_statistics(self, messages: List[Dict[str, Any]], user_language: str) -> Dict[str, Any]:
        """Calculate detailed conversation statistics"""
        if not messages:
            return {}
        
        total_messages = len(messages)
        voice_messages = 0
        text_messages = 0
        total_voice_duration = 0
        languages_used = set()
        participants = {}
        time_range = {"start": None, "end": None}
        
        for msg in messages:
            # Message type counting
            msg_type = msg.get('message_type', 'text')
            if msg_type == 'voice':
                voice_messages += 1
                duration = msg.get('audio_duration', 0)
                total_voice_duration += duration
            else:
                text_messages += 1
            
            # Participant counting
            sender = msg.get('sender_name', msg.get('sender', 'Unknown'))
            participants[sender] = participants.get(sender, 0) + 1
            
            # Language tracking
            lang = msg.get('original_language')
            if lang:
                languages_used.add(lang)
            
            # Time range
            timestamp = msg.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    
                    if time_range["start"] is None or dt < time_range["start"]:
                        time_range["start"] = dt
                    if time_range["end"] is None or dt > time_range["end"]:
                        time_range["end"] = dt
                except Exception:
                    pass
        
        # Format time range
        date_range = "Unknown"
        if time_range["start"] and time_range["end"]:
            if time_range["start"].date() == time_range["end"].date():
                date_range = time_range["start"].strftime('%Y-%m-%d')
            else:
                date_range = f"{time_range['start'].strftime('%Y-%m-%d')} to {time_range['end'].strftime('%Y-%m-%d')}"
        
        # Calculate percentages
        voice_percentage = round((voice_messages / total_messages) * 100, 1) if total_messages > 0 else 0
        
        # Format voice duration
        voice_duration_formatted = self._format_duration(total_voice_duration)
        
        # Find most active participant
        most_active = max(participants.items(), key=lambda x: x[1])[0] if participants else None
        
        return {
            "total_messages": total_messages,
            "voice_messages": voice_messages,
            "text_messages": text_messages,
            "voice_percentage": voice_percentage,
            "total_voice_duration_seconds": total_voice_duration,
            "total_voice_duration_formatted": voice_duration_formatted,
            "participants": participants,
            "participant_count": len(participants),
            "most_active_participant": most_active,
            "languages_used": list(languages_used),
            "language_count": len(languages_used),
            "date_range": date_range,
            "summary_language": user_language
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    def create_downloadable_summary(
        self, 
        summary_data: Dict[str, Any], 
        format_type: str = "markdown"
    ) -> str:
        """Create downloadable summary content"""
        
        if format_type.lower() == "markdown":
            return self._create_markdown_summary(summary_data)
        elif format_type.lower() == "txt":
            return self._create_text_summary(summary_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _create_markdown_summary(self, summary_data: Dict[str, Any]) -> str:
        """Create markdown formatted summary"""
        stats = summary_data.get('statistics', {})
        summary_text = summary_data.get('summary', 'No summary available')
        
        markdown_content = f"""# 💬 Chat Summary Report

**Generated:** {summary_data.get('generated_at', 'Unknown')}  
**Chat Type:** {summary_data.get('chat_type', 'Unknown').title()} Chat  
**Language:** {summary_data.get('language', 'en').upper()}  
**Participants:** {', '.join(summary_data.get('participants', []))}  

---

{summary_text}

---

## 📊 Conversation Statistics

| Metric | Value |
|--------|-------|
| Total Messages | {stats.get('total_messages', 0)} |
| Text Messages | {stats.get('text_messages', 0)} |
| Voice Messages | {stats.get('voice_messages', 0)} |
| Voice Percentage | {stats.get('voice_percentage', 0)}% |
| Total Voice Duration | {stats.get('total_voice_duration_formatted', '0s')} |
| Most Active Participant | {stats.get('most_active_participant', 'N/A')} |
| Languages Used | {', '.join(stats.get('languages_used', []))} |
| Date Range | {stats.get('date_range', 'Unknown')} |

---

## 👥 Participation Breakdown

"""
        
        # Add participant statistics
        participants = stats.get('participants', {})
        if participants:
            for participant, count in sorted(participants.items(), key=lambda x: x[1], reverse=True):
                percentage = round((count / stats.get('total_messages', 1)) * 100, 1)
                markdown_content += f"- **{participant}**: {count} messages ({percentage}%)\n"
        
        markdown_content += f"""

---

*Generated by AI Chat Summary Service using Llama 3.1 8B*  
*Summary Language: {summary_data.get('language', 'en').upper()}*
"""
        
        return markdown_content
    
    def _create_text_summary(self, summary_data: Dict[str, Any]) -> str:
        """Create plain text summary"""
        markdown_content = self._create_markdown_summary(summary_data)
        
        # Convert markdown to plain text
        text_content = re.sub(r'#+ ', '', markdown_content)  # Remove headers
        text_content = re.sub(r'\*\*(.*?)\*\*', r'\1', text_content)  # Remove bold
        text_content = re.sub(r'\*(.*?)\*', r'\1', text_content)  # Remove italic
        text_content = re.sub(r'\|.*?\|', '', text_content)  # Remove tables
        text_content = re.sub(r'^-+$', '', text_content, flags=re.MULTILINE)  # Remove horizontal rules
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)  # Clean extra newlines
        
        return text_content.strip()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status including GPU information"""
        return {
            "model_loaded": self.is_model_loaded,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "supported_languages": list(self.language_codes.keys()),
            "gpu_available": self.gpu_available,
            "gpu_layers": self.gpu_layers,
            "gpu_info": self.gpu_info,
            "acceleration_type": self.gpu_info.get('type', 'CPU') if self.gpu_available else 'CPU'
        }
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global service instance
chat_summary_service = ChatSummaryService()


def test_summary_service():
    """Test function for the chat summary service"""
    import asyncio
    
    async def run_test():
        # Test data
        test_messages = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "sender_name": "Alice",
                "message_type": "text",
                "original_text": "Hi everyone! How are you doing today?",
                "original_language": "en"
            },
            {
                "timestamp": "2024-01-01T10:01:00", 
                "sender_name": "Bob",
                "message_type": "text",
                "original_text": "I'm doing great! Working on the new project.",
                "original_language": "en"
            },
            {
                "timestamp": "2024-01-01T10:02:00",
                "sender_name": "Alice", 
                "message_type": "voice",
                "original_text": "That sounds interesting! Can you tell us more?",
                "original_language": "en",
                "audio_duration": 3.5
            }
        ]
        
        # Test summary generation
        result = await chat_summary_service.generate_chat_summary(
            messages=test_messages,
            user_language="en",
            chat_type="group",
            participants=["Alice", "Bob"],
            user_id=1
        )
        
        print("=== Test Results ===")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Summary: {result['summary'][:200]}...")
            print(f"Statistics: {result['statistics']}")
        else:
            print(f"Error: {result['error']}")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    print("🧪 Testing Chat Summary Service...")
    test_summary_service()
