"""
Enhanced Chat Summary Service using Hugging Face Transformers
Provides comprehensive AI-powered chat summarization with detailed conversation analysis
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor

# Try importing transformers with error handling
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"WARNING: Transformers not available: {e}")
    print("Install with: pip install transformers torch")

logger = logging.getLogger(__name__)

class ChatSummaryService:
    """
    Enhanced AI-powered chat summary service with comprehensive conversation analysis
    Uses BART model for intelligent summarization with detailed statistics
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.device = "cpu"
        self.model_name = "facebook/bart-large-cnn"  # Excellent summarization model
        self.is_loading = False
        self.load_lock = threading.Lock()
        
        # Enhanced configuration for better summaries
        self.max_length = 1024  # Increased for longer conversations
        self.summary_max_length = 200  # Longer summaries
        self.summary_min_length = 80
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary")
        
        # Local model path in artifacts folder
        self.local_model_path = Path("artifacts/models/bart-large-cnn")
        
        # Supported languages for summary generation (limited to 3 languages)
        self.supported_languages = {
            "en": "English",
            "fr": "French", 
            "ar": "Arabic"
        }
        
        logger.info("🤖 Enhanced Chat Summary Service initialized (Transformers-based)")
        logger.info(f"📁 Local model path: {self.local_model_path}")
        
        # Try to load model asynchronously if transformers available
        if TRANSFORMERS_AVAILABLE:
            asyncio.create_task(self._initialize_model_async())
    
    async def _initialize_model_async(self):
        """Asynchronously initialize the BART model for summarization"""
        try:
            await self._load_model()
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
    
    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect available GPU acceleration with enhanced detection"""
        gpu_info = {
            "cuda_available": False,
            "mps_available": False,
            "device": "cpu",
            "gpu_memory": 0,
            "gpu_name": "CPU Only",
            "gpu_count": 0,
            "compute_capability": None
        }
        
        try:
            # Check CUDA (NVIDIA) with detailed info
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info["cuda_available"] = True
                gpu_info["device"] = "cuda"
                gpu_info["gpu_count"] = gpu_count
                
                # Get info for the primary GPU
                properties = torch.cuda.get_device_properties(0)
                gpu_info["gpu_memory"] = properties.total_memory // (1024**3)
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["compute_capability"] = f"{properties.major}.{properties.minor}"
                
                # Log detailed GPU information
                logger.info(f"🚀 NVIDIA GPU detected: {gpu_info['gpu_name']}")
                logger.info(f"📊 GPU Memory: {gpu_info['gpu_memory']}GB")
                logger.info(f"🔢 GPU Count: {gpu_count}")
                logger.info(f"⚡ Compute Capability: {gpu_info['compute_capability']}")
            
            # Check MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["mps_available"] = True
                gpu_info["device"] = "mps"
                gpu_info["gpu_name"] = "Apple Silicon"
                logger.info("🍎 Apple Silicon MPS acceleration detected")
            
            else:
                logger.info("💻 Using CPU for inference")
                
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        return gpu_info
    
    async def _load_model(self) -> bool:
        """Load the summarization model with optimized GPU utilization"""
        if self.model is not None:
            return True
            
        with self.load_lock:
            if self.model is not None:  # Double-check after acquiring lock
                return True
                
            if self.is_loading:
                return False
                
            self.is_loading = True
            
        try:
            logger.info("🤖 Loading chat summarization model with GPU optimization...")
            
            # Detect GPU capabilities
            gpu_info = self._detect_gpu_capabilities()
            self.device = gpu_info["device"]
            
            # Determine model source (local artifacts or HuggingFace Hub)
            if self.local_model_path.exists() and (self.local_model_path / "config.json").exists():
                model_source = str(self.local_model_path)
                logger.info(f"📁 Loading model from local artifacts: {model_source}")
            else:
                model_source = self.model_name
                logger.info(f"🌐 Loading model from HuggingFace Hub: {model_source}")
                logger.info("💡 To use local model, run: python download_bart_model.py")
            
            # Load tokenizer
            logger.info(f"📥 Loading tokenizer from {model_source}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            # Configure model loading based on available hardware
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            # GPU-specific optimizations
            if self.device == "cuda":
                model_kwargs.update({
                    "device_map": "auto",  # Automatically distribute across GPUs
                    "load_in_8bit": gpu_info["gpu_memory"] < 8,  # Use 8-bit if limited GPU memory
                })
                logger.info("🚀 Configuring for CUDA acceleration")
            elif self.device == "mps":
                logger.info("🍎 Configuring for Apple Silicon MPS")
            
            # Load model with optimizations
            logger.info(f"📥 Loading model {model_source} with optimizations...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_source,
                **model_kwargs
            )
            
            # Move to appropriate device if not using device_map
            if self.device != "cpu" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
                logger.info(f"📱 Model moved to {self.device}")
            
            # Create optimized summarization pipeline
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_length": 150,
                "min_length": 30,
                "do_sample": False,
                "batch_size": 4 if self.device == "cuda" else 1,  # Larger batch on GPU
            }
            
            # Only set device if not using device_map (Accelerate handles device placement)
            if "device_map" not in model_kwargs:
                pipeline_kwargs["device"] = 0 if self.device == "cuda" else -1
                pipeline_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
            
            self.summarizer = pipeline("summarization", **pipeline_kwargs)
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
        finally:
            self.is_loading = False
    
    async def generate_chat_summary(
        self,
        messages: List[Dict[str, Any]],
        user_language: str = "en",
        chat_type: str = "direct",
        participants: List[str] = None,
        user_id: int = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered summary of chat messages
        """
        try:
            logger.info(f"🔄 Generating {chat_type} chat summary for {len(messages)} messages")
            
            # Check if model is available
            model_loaded = await self._load_model()
            if not model_loaded:
                return {
                    "success": False,
                    "error": "❌ Model not available. Please check model installation.",
                    "summary": "Model loading failed",
                    "statistics": self._calculate_statistics(messages),
                    "generated_at": datetime.now().isoformat()
                }
            
            # Prepare chat text for summarization
            chat_text = self._prepare_chat_text(messages, participants)
            
            if not chat_text.strip():
                return {
                    "success": False,
                    "error": "No valid chat content to summarize",
                    "summary": "No messages found",
                    "statistics": self._calculate_statistics(messages),
                    "generated_at": datetime.now().isoformat()
                }
            
            # Generate summary using the model
            summary = await self._generate_summary_with_model(chat_text, user_language, chat_type)
            
            # Calculate statistics
            statistics = self._calculate_statistics(messages)
            
            logger.info(f"✅ Summary generated successfully for user {user_id}")
            
            return {
                "success": True,
                "summary": summary,
                "statistics": statistics,
                "generated_at": datetime.now().isoformat(),
                "model_info": {
                    "name": self.model_name,
                    "device": self.device,
                    "language": user_language
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {
                "success": False,
                "error": f"Failed to generate summary: {str(e)}",
                "summary": "Summary generation error",
                "statistics": self._calculate_statistics(messages) if messages else {},
                "generated_at": datetime.now().isoformat()
            }
    
    async def _generate_summary_with_model(self, chat_text: str, language: str, chat_type: str) -> str:
        """Generate comprehensive summary using the transformer model with conversation content"""
        try:
            # Prepare text for summarization with enhanced conversation analysis
            prepared_text = self._prepare_text_for_summarization(chat_text, language, chat_type)
            
            # Generate summary using model in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self._generate_summary_sync, 
                prepared_text
            )
            
            if result and len(result) > 0:
                summary = result[0]['summary_text']
                return self._format_enhanced_summary(summary, language, chat_type, chat_text)
            else:
                return "Unable to generate detailed summary from conversation content."
                
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return f"Basic conversation analysis: {len(chat_text.split())} words exchanged."
    
    def _generate_summary_sync(self, text: str) -> List[Dict[str, str]]:
        """Synchronous summary generation for thread pool execution"""
        return self.summarizer(
            text, 
            max_length=self.summary_max_length, 
            min_length=self.summary_min_length, 
            do_sample=False,
            num_beams=4,  # Better quality summaries
            early_stopping=True
        )
    
    def _prepare_text_for_summarization(self, chat_text: str, language: str, chat_type: str) -> str:
        """Prepare chat text for BART summarization with enhanced conversation context"""
        
        # Enhanced truncation for longer conversations
        max_chars = self.max_length * 4  # Allow more content for better analysis
        if len(chat_text) > max_chars:
            # Smart truncation - take beginning and end of conversation
            start_part = chat_text[:max_chars//2]
            end_part = chat_text[-max_chars//2:]
            chat_text = f"{start_part}\n...[conversation continued]...\n{end_part}"
        
        # Enhanced context for different languages and chat types
        language_name = self.supported_languages.get(language, "English")
        
        # Create comprehensive prompt for better summarization
        if chat_type == "group":
            prompt_prefix = f"Summarize this group conversation in {language_name}. Include key topics discussed, main participants, and important decisions made:\n\n"
        else:
            prompt_prefix = f"Summarize this private conversation in {language_name}. Include main topics, key exchanges, and important points discussed:\n\n"
        
        return prompt_prefix + chat_text
    
    def _format_enhanced_summary(self, summary: str, language: str, chat_type: str, original_chat: str) -> str:
        """Format enhanced summary with conversation analysis and content highlights"""
        # Clean up the AI-generated summary
        summary = summary.strip()
        
        # Remove any prompt artifacts that might have leaked into output
        summary_lines = summary.split('\n')
        cleaned_lines = []
        for line in summary_lines:
            if not (line.startswith("Summarize") or line.startswith("Résumez") or line.startswith("لخص")):
                cleaned_lines.append(line)
        
        if cleaned_lines:
            summary = '\n'.join(cleaned_lines).strip()
        
        # Extract key conversation highlights
        conversation_highlights = self._extract_conversation_highlights(original_chat)
        
        # Create enhanced summary format
        enhanced_summary = []
        
        # Main AI summary
        if summary:
            enhanced_summary.append(f"📝 **Conversation Summary**: {summary}")
        else:
            # Fallback summary based on language
            fallback_summaries = {
                "en": "Detailed conversation analysis completed.",
                "fr": "Analyse détaillée de conversation terminée.",
                "ar": "تم إكمال تحليل مفصل للمحادثة."
            }
            enhanced_summary.append(f"📝 **Conversation Summary**: {fallback_summaries.get(language, fallback_summaries['en'])}")
        
        # Add conversation highlights if available
        if conversation_highlights:
            enhanced_summary.append("\n🔍 **Key Exchanges**:")
            for highlight in conversation_highlights[:3]:  # Top 3 highlights
                enhanced_summary.append(f"• {highlight}")
        
        return '\n'.join(enhanced_summary)
    
    def _extract_conversation_highlights(self, chat_text: str) -> List[str]:
        """Extract key conversation highlights and important exchanges"""
        highlights = []
        
        try:
            lines = chat_text.split('\n')
            
            # Look for longer messages (likely important)
            for line in lines:
                if ':' in line and len(line) > 50:
                    # Extract speaker and message
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        message = parts[1].strip()
                        
                        # Skip voice messages for highlights
                        if not message.startswith('[Voice message'):
                            # Truncate long messages for highlights
                            if len(message) > 100:
                                message = message[:100] + "..."
                            highlights.append(f"{speaker}: {message}")
            
            # Return most relevant highlights (longer ones first)
            highlights.sort(key=len, reverse=True)
            return highlights[:5]  # Top 5 highlights
            
        except Exception as e:
            logger.warning(f"Error extracting highlights: {e}")
            return []
    
    def _prepare_chat_text(self, messages: List[Dict[str, Any]], participants: List[str] = None) -> str:
        """Prepare chat messages for AI processing"""
        if not messages:
            return ""
        
        chat_lines = []
        for msg in messages:
            try:
                sender_name = msg.get('sender_name', 'Unknown')
                message_text = msg.get('translated_text') or msg.get('original_text', '')
                message_type = msg.get('message_type', 'text')
                
                if message_type == 'voice':
                    duration = msg.get('audio_duration', 0)
                    chat_lines.append(f"{sender_name}: [Voice message - {duration}s]")
                elif message_text.strip():
                    # Limit message length to prevent token overflow
                    if len(message_text) > 200:
                        message_text = message_text[:200] + "..."
                    chat_lines.append(f"{sender_name}: {message_text}")
                    
            except Exception as e:
                logger.warning(f"Error processing message: {e}")
                continue
        
        return "\n".join(chat_lines)
    
    def _calculate_statistics(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate conversation statistics"""
        if not messages:
            return {
                "total_messages": 0,
                "voice_messages": 0,
                "voice_percentage": 0,
                "total_voice_duration": 0,
                "most_active_user": "None",
                "languages_used": [],
                "date_range": "No messages",
                "participants": [],
                "participant_count": 0,
                "conversation_length": 0,
                "average_message_length": 0,
                "longest_message": "",
                "busiest_hour": "Unknown"
            }
        
        # Enhanced analysis
        total_messages = len(messages)
        voice_messages = 0
        total_voice_duration = 0
        user_counts = {}
        languages = set()
        message_lengths = []
        longest_message = ""
        max_length = 0
        hour_counts = {}
        
        for msg in messages:
            # User activity tracking
            sender = msg.get('sender_name', 'Unknown')
            user_counts[sender] = user_counts.get(sender, 0) + 1
            
            # Language tracking
            lang = msg.get('original_language', 'en')
            if lang:
                languages.add(lang)
            
            # Voice message analysis
            if msg.get('message_type') == 'voice':
                voice_messages += 1
                duration = msg.get('audio_duration', 0)
                total_voice_duration += duration
            
            # Message length analysis
            message_text = msg.get('translated_text') or msg.get('original_text', '')
            if message_text:
                length = len(message_text)
                message_lengths.append(length)
                if length > max_length:
                    max_length = length
                    longest_message = message_text[:100] + "..." if length > 100 else message_text
            
            # Time analysis for busiest hour
            try:
                timestamp = msg.get('timestamp', '')
                if timestamp and 'T' in timestamp:
                    hour = int(timestamp.split('T')[1][:2])
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except (ValueError, IndexError):
                pass
        
        # Calculate percentages and averages
        voice_percentage = (voice_messages / total_messages * 100) if total_messages > 0 else 0
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        # Find most active user and busiest hour
        most_active = max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else "Unknown"
        busiest_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else "Unknown"
        if busiest_hour != "Unknown":
            busiest_hour = f"{busiest_hour:02d}:00"
        
        # Date range calculation
        dates = []
        for msg in messages:
            if msg.get('timestamp'):
                try:
                    date_str = msg['timestamp'][:10]  # Get YYYY-MM-DD part
                    dates.append(date_str)
                except Exception:
                    continue
        
        date_range = f"{min(dates)} to {max(dates)}" if len(set(dates)) > 1 else (dates[0] if dates else "Unknown")
        
        return {
            "total_messages": total_messages,
            "voice_messages": voice_messages,
            "voice_percentage": round(voice_percentage, 1),
            "total_voice_duration": round(total_voice_duration, 1),
            "most_active_user": most_active,
            "languages_used": list(languages),
            "date_range": date_range,
            "participants": list(user_counts.keys()),
            "participant_count": len(user_counts),
            "average_message_length": round(avg_message_length, 1),
            "longest_message": longest_message,
            "busiest_hour": busiest_hour
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "device": self.device,
            "supported_languages": list(self.supported_languages.keys()),
            "is_loading": self.is_loading
        }
    
    def create_downloadable_summary(self, summary_data: Dict[str, Any], format: str = "markdown") -> str:
        """Create enhanced downloadable summary content with conversation highlights"""
        
        summary = summary_data.get("summary", "No summary available")
        stats = summary_data.get("statistics", {})
        generated_at = summary_data.get("generated_at", "")
        language = summary_data.get("language", "en")
        
        if format == "markdown":
            content = f"""# 📊 Enhanced Chat Summary

**Generated:** {generated_at}  
**Language:** {self.supported_languages.get(language, language)}  
**Participants:** {stats.get('participant_count', 0)} users

## 📝 AI Summary

{summary}

## 📈 Detailed Statistics

### 💬 Message Analysis
- **Total Messages:** {stats.get('total_messages', 0)}
- **Average Message Length:** {stats.get('average_message_length', 0)} characters
- **Most Active User:** {stats.get('most_active_user', 'Unknown')}
- **Participants:** {', '.join(stats.get('participants', []))}

### 🎙️ Voice Messages
- **Voice Messages:** {stats.get('voice_messages', 0)} ({stats.get('voice_percentage', 0)}%)
- **Total Voice Duration:** {stats.get('total_voice_duration', 0)}s

### 🌍 Languages & Time
- **Languages Used:** {', '.join(stats.get('languages_used', []))}
- **Date Range:** {stats.get('date_range', 'Unknown')}
- **Busiest Hour:** {stats.get('busiest_hour', 'Unknown')}

### 📋 Key Information
- **Longest Message Preview:** {stats.get('longest_message', 'N/A')}

---
*📱 Generated by Enhanced AI Chat Summarization Service*  
*🤖 Model: {self.model_name} | Device: {self.device}*
"""
        else:  # txt format
            content = f"""ENHANCED CHAT SUMMARY
=====================

Generated: {generated_at}
Language: {self.supported_languages.get(language, language)}
Participants: {stats.get('participant_count', 0)} users

AI SUMMARY
----------
{summary}

DETAILED STATISTICS
-------------------

Message Analysis:
- Total Messages: {stats.get('total_messages', 0)}
- Average Message Length: {stats.get('average_message_length', 0)} characters
- Most Active User: {stats.get('most_active_user', 'Unknown')}
- Participants: {', '.join(stats.get('participants', []))}

Voice Messages:
- Voice Messages: {stats.get('voice_messages', 0)} ({stats.get('voice_percentage', 0)}%)
- Total Voice Duration: {stats.get('total_voice_duration', 0)}s

Languages & Time:
- Languages Used: {', '.join(stats.get('languages_used', []))}
- Date Range: {stats.get('date_range', 'Unknown')}
- Busiest Hour: {stats.get('busiest_hour', 'Unknown')}

Key Information:
- Longest Message Preview: {stats.get('longest_message', 'N/A')}

Generated by Enhanced AI Chat Summarization Service
Model: {self.model_name} | Device: {self.device}
"""
        
        return content


# Global service instance
chat_summary_service = ChatSummaryService()
