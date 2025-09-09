"""
Chat Summary Service using Hugging Face Transformers
Provides AI-powered chat summarization without llama-cpp dependency
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

class ChatSummaryService:
    """
    AI-powered chat summary service using Hugging Face Transformers
    Uses lightweight models that don't require llama-cpp
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.device = "cpu"
        self.model_name = "facebook/bart-large-cnn"  # Excellent summarization model
        self.is_loading = False
        self.load_lock = threading.Lock()
        
        # Local model path in artifacts folder
        self.local_model_path = Path("artifacts/models/bart-large-cnn")
        
        # Supported languages for summary generation (limited to 3 languages)
        self.supported_languages = {
            "en": "English",
            "fr": "French", 
            "ar": "Arabic"
        }
        
        logger.info("🤖 Chat Summary Service initialized (Transformers-based)")
        logger.info(f"📁 Local model path: {self.local_model_path}")
    
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
                "device": 0 if self.device == "cuda" else -1,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "max_length": 150,
                "min_length": 30,
                "do_sample": False,
                "batch_size": 4 if self.device == "cuda" else 1,  # Larger batch on GPU
            }
            
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
        """Generate summary using the transformer model"""
        try:
            # Prepare text for summarization
            prepared_text = self._prepare_text_for_summarization(chat_text, language, chat_type)
            
            # Generate summary
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.summarizer(prepared_text, max_length=150, min_length=30, do_sample=False)
            )
            
            if result and len(result) > 0:
                summary = result[0]['summary_text']
                return self._format_summary(summary, language, chat_type)
            else:
                return "Unable to generate detailed summary."
                
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return f"Brief conversation summary: {len(chat_text.split())} words exchanged."
    
    def _prepare_text_for_summarization(self, chat_text: str, language: str, chat_type: str) -> str:
        """Prepare chat text for the BART summarization model with language-specific optimization"""
        
        # Truncate if too long (BART has input limits)
        max_chars = 1000
        if len(chat_text) > max_chars:
            chat_text = chat_text[:max_chars] + "..."
        
        # Language-specific context prefixes for better summarization
        language_contexts = {
            "en": f"Summarize this {chat_type} conversation:",
            "fr": f"Résumez cette conversation {chat_type}:",
            "ar": f"لخص هذه المحادثة {chat_type}:"
        }
        
        context_prefix = language_contexts.get(language, language_contexts["en"])
        return f"{context_prefix}\n\n{chat_text}"
    
    def _format_summary(self, summary: str, language: str, chat_type: str) -> str:
        """Format the generated summary with language-specific enhancements"""
        # Clean up the summary
        summary = summary.strip()
        
        # Remove any context prefix that might have been included in the output
        if summary.startswith("Summarize") or summary.startswith("Résumez") or summary.startswith("لخص"):
            lines = summary.split('\n')
            summary = '\n'.join(lines[1:]).strip() if len(lines) > 1 else summary
        
        # Add language indicator if not English
        if language == "fr":
            if not summary:
                summary = "Résumé de conversation généré automatiquement."
        elif language == "ar":
            if not summary:
                summary = "ملخص محادثة تم إنشاؤه تلقائياً."
        elif language == "en":
            if not summary:
                summary = "Automatically generated conversation summary."
        
        return summary
    
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
                "date_range": "No messages"
            }
        
        # Basic counts
        total_messages = len(messages)
        voice_messages = sum(1 for msg in messages if msg.get('message_type') == 'voice')
        voice_percentage = (voice_messages / total_messages * 100) if total_messages > 0 else 0
        
        # Voice duration
        total_voice_duration = sum(msg.get('audio_duration', 0) for msg in messages)
        
        # Most active user
        user_counts = {}
        languages = set()
        
        for msg in messages:
            sender = msg.get('sender_name', 'Unknown')
            user_counts[sender] = user_counts.get(sender, 0) + 1
            
            lang = msg.get('original_language', 'en')
            if lang:
                languages.add(lang)
        
        most_active = max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else "Unknown"
        
        # Date range
        dates = []
        for msg in messages:
            if msg.get('timestamp'):
                try:
                    date_str = msg['timestamp'][:10]  # Get YYYY-MM-DD part
                    dates.append(date_str)
                except Exception:
                    continue
        
        date_range = f"{min(dates)} to {max(dates)}" if dates else "Unknown"
        if len(set(dates)) == 1:
            date_range = dates[0]
        
        return {
            "total_messages": total_messages,
            "voice_messages": voice_messages,
            "voice_percentage": round(voice_percentage, 1),
            "total_voice_duration": round(total_voice_duration, 1),
            "most_active_user": most_active,
            "languages_used": list(languages),
            "date_range": date_range
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
        """Create downloadable summary content"""
        
        summary = summary_data.get("summary", "No summary available")
        stats = summary_data.get("statistics", {})
        participants = summary_data.get("participants", [])
        generated_at = summary_data.get("generated_at", "")
        language = summary_data.get("language", "en")
        
        if format == "markdown":
            content = f"""# Chat Summary

**Generated:** {generated_at}  
**Language:** {self.supported_languages.get(language, language)}  
**Participants:** {', '.join(participants)}

## Summary

{summary}

## Statistics

- **Total Messages:** {stats.get('total_messages', 0)}
- **Voice Messages:** {stats.get('voice_messages', 0)} ({stats.get('voice_percentage', 0)}%)
- **Voice Duration:** {stats.get('total_voice_duration', 0)}s
- **Most Active:** {stats.get('most_active_user', 'Unknown')}
- **Languages:** {', '.join(stats.get('languages_used', []))}
- **Date Range:** {stats.get('date_range', 'Unknown')}

---
*Generated by AI Chat Summarization Service*
"""
        else:  # txt format
            content = f"""CHAT SUMMARY
============

Generated: {generated_at}
Language: {self.supported_languages.get(language, language)}
Participants: {', '.join(participants)}

SUMMARY
-------
{summary}

STATISTICS
----------
Total Messages: {stats.get('total_messages', 0)}
Voice Messages: {stats.get('voice_messages', 0)} ({stats.get('voice_percentage', 0)}%)
Voice Duration: {stats.get('total_voice_duration', 0)}s
Most Active: {stats.get('most_active_user', 'Unknown')}
Languages: {', '.join(stats.get('languages_used', []))}
Date Range: {stats.get('date_range', 'Unknown')}

Generated by AI Chat Summarization Service
"""
        
        return content


# Global service instance
chat_summary_service = ChatSummaryService()
