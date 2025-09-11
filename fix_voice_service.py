#!/usr/bin/env python3
"""
Fix Voice Service - Remove duplicate Whisper model loading
"""

import os
import shutil

def fix_voice_service():
    """Fix the voice service to use lazy loading and prevent duplicate Whisper loading"""
    
    backup_file = "app/services/voice_service_backup.py"
    target_file = "app/services/voice_service.py"
    
    print("🔧 Fixing voice service to prevent duplicate Whisper loading...")
    
    # Read the backup file
    with open(backup_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Make the necessary changes for lazy loading
    modified_content = content.replace(
        '''        # Initialize Whisper model with better configuration
        self.whisper_model = None
        self.whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if WHISPER_AVAILABLE:
            try:
                print("🔄 Loading Whisper model...")
                # Use 'small' model for better accuracy than 'base'
                self.whisper_model = whisper.load_model("small", device=self.whisper_device)
                print(f"✅ Whisper model loaded successfully on {self.whisper_device}")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None''',
        '''        # Initialize Whisper model (lazy loading for better startup performance)
        self.whisper_model = None
        self.whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._whisper_loading = False  # Prevent multiple loading attempts'''
    )
    
    # Add the lazy loading method after the __init__ method
    init_end = modified_content.find('        self.target_dbfs = -16  # Target volume level for optimal recognition') + len('        self.target_dbfs = -16  # Target volume level for optimal recognition')
    
    lazy_loading_method = '''
    
    def _load_whisper_model(self) -> bool:
        """Lazy load Whisper model to avoid startup delays"""
        if self.whisper_model is None and not self._whisper_loading and WHISPER_AVAILABLE:
            self._whisper_loading = True
            try:
                print("🔄 Loading Whisper model...")
                # Use 'turbo' model for better speed
                self.whisper_model = whisper.load_model("turbo", device=self.whisper_device)
                print(f"✅ Whisper model loaded successfully on {self.whisper_device}")
                return True
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None
                return False
            finally:
                self._whisper_loading = False
        return self.whisper_model is not None'''
    
    modified_content = modified_content[:init_end] + lazy_loading_method + modified_content[init_end:]
    
    # Replace all whisper model checks with lazy loading calls
    modified_content = modified_content.replace(
        'if not self.whisper_model:',
        'if not self._load_whisper_model():'
    )
    
    modified_content = modified_content.replace(
        'if WHISPER_AVAILABLE and self.whisper_model:',
        'if WHISPER_AVAILABLE and self._load_whisper_model():'
    )
    
    # Write the modified content to the target file
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("✅ Voice service fixed!")
    print("🔧 Changes made:")
    print("  • Whisper model now uses lazy loading")
    print("  • No model loading during service initialization")
    print("  • Model loads only when first needed")
    print("  • Prevents duplicate loading at startup")

if __name__ == "__main__":
    fix_voice_service()
