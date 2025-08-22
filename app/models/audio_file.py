# filepath: app/models/audio_file.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import relationship
from ..core.database import Base

class AudioFile(Base):
    """
    Detailed tracking of audio files for voice messages
    Each voice message can have multiple audio files (one per language)
    """
    __tablename__ = "audio_files"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Audio file details
    language = Column(String(10), nullable=False, index=True)  # Language code (en, fr, ar, etc.)
    file_path = Column(String(500), nullable=False)  # Local file system path
    file_url = Column(String(500), nullable=False)  # Public URL for accessing the file
    file_size = Column(Integer, nullable=False)  # File size in bytes
    duration = Column(Float, nullable=True)  # Duration in seconds
    mime_type = Column(String(50), default="audio/mp3")  # MIME type
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    message = relationship("Message", back_populates="audio_files")
    
    def __repr__(self):
        return f"<AudioFile(id={self.id}, message_id={self.message_id}, language={self.language})>"
    
    @property
    def size_mb(self) -> float:
        """Get file size in MB"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0.0
    
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration (mm:ss)"""
        if not self.duration:
            return "0:00"
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes}:{seconds:02d}"
