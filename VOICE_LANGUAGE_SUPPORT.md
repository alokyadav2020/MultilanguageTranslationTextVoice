# Voice Functionality Language Support

## Supported Languages

The voice functionality now supports only **three languages**:

- **English** (`en`)
- **French** (`fr`) 
- **Arabic** (`ar`)

## Changes Made

### 1. Language Validation
- Added validation in the voice service to only accept the three supported languages
- Added validation in the API endpoint to reject unsupported languages
- Added validation in Pydantic schemas

### 2. Speech Recognition
- Simplified language mapping for Google Speech Recognition
- Only supports:
  - English: `en-US`
  - French: `fr-FR`
  - Arabic: `ar-SA`

### 3. Text-to-Speech (TTS)
- Simplified language mapping for Google TTS
- Only supports: `en`, `fr`, `ar`

### 4. Translation
- Translation service now only translates between the three supported languages
- If the source language is one of the three, it translates to the other two

### 5. File Path Fix
- Fixed the temporary file path issue for Windows compatibility
- Uses `tempfile.gettempdir()` instead of hardcoded `/tmp/` path

## API Usage

When uploading a voice message, the `language` parameter must be one of:
- `en` (English)
- `fr` (French) 
- `ar` (Arabic)

Any other language code will result in a 400 Bad Request error.

## Error Handling

The system now provides clear error messages:
- "Unsupported language '{language}'. Supported languages: ['en', 'fr', 'ar']"
- Language validation happens at both the API level and service level

## User Selection

Users must explicitly select their language from the three available options. The system no longer attempts to automatically detect the language from the audio.
