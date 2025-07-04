#!/usr/bin/env python3
"""
Simple test script for transcription functionality.
Usage: python test_transcribe.py <path_to_mp3_file>
"""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from fastapi import UploadFile
from eve.api.handlers import handle_transcribe
from eve.api.api_requests import TranscribeRequest

async def main():
    """Main function to run the transcription test."""
    if len(sys.argv) != 2:
        print("Usage: python test_transcribe.py <path_to_mp3_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    try:
        # Check if file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing audio file: {audio_path}")
        
        # Read file content
        with open(audio_path, "rb") as f:
            file_content = f.read()
        
        # Create mock UploadFile
        import io
        
        class MockUploadFile:
            def __init__(self, filename, content, content_type):
                self.filename = filename
                self.content_type = content_type
                self._content = content
                
            async def read(self):
                return self._content
                
        file = MockUploadFile(
            filename=Path(audio_path).name,
            content=file_content,
            content_type="audio/mpeg"
        )
        
        # Create transcription request
        request = TranscribeRequest(
            user_id="test_user",
            model="gpt-4o-mini-transcribe"
        )
        
        # Call the existing handler
        result = await handle_transcribe(file, request)
        
        # Extract transcription text from response
        if hasattr(result, 'body'):
            import json
            response_data = json.loads(result.body)
            transcription_text = response_data.get('transcription', '')
        else:
            transcription_text = result.get('transcription', '')
        
        print("\nTranscription:")
        print(transcription_text)
        
    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())