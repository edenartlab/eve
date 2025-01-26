import anthropic
import json
import shutil
import subprocess
import os
import uuid
from pathlib import Path
import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from instructor.function_calls import openai_schema
from .... import eden_utils

class FFmpegError(Exception):
    """Custom exception for FFmpeg-related errors"""
    def __init__(self, message: str, command: str, stderr: Optional[str] = None):
        self.message = message
        self.command = command
        self.stderr = stderr
        super().__init__(self.message)

class FFmpegResponse(BaseModel):
    """Response model for FFmpeg command generation"""
    command: str = Field(..., description="The complete FFmpeg command to execute")
    output_path: str = Field(..., description="The output filepath for the resulting file")


def probe_media(filepath: str) -> dict:
    """Get media file information using ffprobe"""
    try:
        process = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filepath
        ], capture_output=True, text=True)
        
        if process.returncode != 0:
            return {}
            
        probe_data = json.loads(process.stdout)
        
        info = {}
        # Get video/image stream info
        video_stream = next(
            (s for s in probe_data.get('streams', []) 
             if s['codec_type'] in ['video', 'image']),
            {}
        )
        if video_stream:
            info['width'] = int(video_stream.get('width', 0))
            info['height'] = int(video_stream.get('height', 0))
            # Calculate fps for videos
            if 'r_frame_rate' in video_stream:
                num, den = map(int, video_stream['r_frame_rate'].split('/'))
                info['fps'] = round(num / den, 2)
        
        # Get audio stream info
        audio_stream = next(
            (s for s in probe_data.get('streams', [])
             if s['codec_type'] == 'audio'),
            {}
        )
        if audio_stream:
            info['samplerate'] = int(audio_stream.get('sample_rate', 0))
        
        # Get duration from format info
        if 'format' in probe_data:
            info['duration'] = round(float(probe_data['format'].get('duration', 0)), 2)
            
        return info
    except Exception:
        return {}

class MediaFiles(BaseModel):
    """Model for organizing and validating media inputs"""
    images: List[str] = Field(default_factory=list)
    video1: Optional[str] = None
    video2: Optional[str] = None
    audio1: Optional[str] = None
    audio2: Optional[str] = None

    def has_media(self) -> bool:
        """Check if at least one media file is provided"""
        return any([
            self.images,
            self.video1,
            self.video2,
            self.audio1,
            self.audio2
        ])

    def to_context_string(self) -> str:
        """Convert media files to readable format for prompt including technical details"""
        media_items = []
        
        for i, img_path in enumerate(self.images, 1):
            img_info = probe_media(img_path)
            width = img_info.get('width', 'unknown')
            height = img_info.get('height', 'unknown')
            media_items.append(f"- Image {i}: {img_path} ({width}x{height})")
        
        video_info = {
            "Video 1": self.video1,
            "Video 2": self.video2
        }
        
        for video_type, path in video_info.items():
            if path:
                v_info = probe_media(path)
                width = v_info.get('width', 'unknown')
                height = v_info.get('height', 'unknown')
                fps = v_info.get('fps', 'unknown')
                duration = v_info.get('duration', 'unknown')
                media_items.append(
                    f"- {video_type}: {path} ({width}x{height}, {fps}fps, {duration}s)"
                )
        
        audio_info = {
            "Audio 1": self.audio1,
            "Audio 2": self.audio2
        }
        
        for audio_type, path in audio_info.items():
            if path:
                a_info = probe_media(path)
                duration = a_info.get('duration', 'unknown')
                samplerate = a_info.get('samplerate', 'unknown')
                media_items.append(
                    f"- {audio_type}: {path} ({duration}s, {samplerate}Hz)"
                )
                
        return "Available media files:\n" + "\n".join(media_items)

async def generate_ffmpeg_command(
    task_instruction: str, 
    media: MediaFiles, 
    previous_attempt: Optional[Dict[str, str]] = None
) -> FFmpegResponse:
    """Generate FFmpeg command using Anthropic API with JSON mode"""
    if not task_instruction:
        raise ValueError("Task instruction cannot be empty")
        
    if not media.has_media():
        raise ValueError("No media files provided")
        
    try:
        client = anthropic.AsyncAnthropic()
        
        prompt_parts = [
            "You are a professional media editing assistant working in a Linux terminal. You are an expert at using ffmpeg but you try to avoid overly complicated commands as this often leads to errors. Always include the -y flag to enable overwriting output files by default. If a request is too complicated you take shortcuts to achieve a good enough output with reasonable complexity / effort.",
            media.to_context_string(),
            f"Generate a single, executable (typically ffmpeg) command to perform the following task:",
            task_instruction
        ]
        
        if previous_attempt:
            error_details = [
                "Your previous attempt to do this failed with the following details:",
                f"Command: {previous_attempt['command']}",
                f"Error: {previous_attempt['error']}"
            ]
            
            if stderr := previous_attempt.get('stderr'):
                error_details.append(f"stderr: {stderr}")
                
            error_details.append("Please analyze the error and generate a new command that addresses the issue and completes the requested task.")
            prompt_parts.extend(error_details)

        prompt_parts.append(
            "Provide the command and output path in a JSON object with the following schema:"
            "{\n"
            '    "command": "string",  // The complete FFmpeg command\n'
            '    "output_path": "string"  // The output filepath for the resulting file\n'
            "}"
        )

        messages = [{"role": "user", "content": "\n\n".join(prompt_parts)}]

        print("---------------------------------------------------------------------")
        print("LLM Prompt:")
        print("\n\n".join(prompt_parts))
        print("---------------------------------------------------------------------")
        
        prompt = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2048,
            "messages": messages,
            "system": "You are an expert at generating FFmpeg commands. Respond with a JSON object containing the command and output path.",
            "tools": [openai_schema(FFmpegResponse).anthropic_schema],
            "tool_choice": {"type": "tool", "name": "FFmpegResponse"}
        }
        
        response = await client.messages.create(**prompt)
        
        if not response.content:
            raise ValueError("Empty response from Anthropic API")
            
        return FFmpegResponse(**response.content[0].input)

    except (json.JSONDecodeError, KeyError) as e:
        raise FFmpegError(f"Invalid response format: {str(e)}", "")
    except Exception as e:
        raise FFmpegError(f"Failed to generate FFmpeg command: {str(e)}", "")

async def execute_ffmpeg_command(command: str, timeout: int = 300) -> None:
    """Execute FFmpeg command with timeout"""
    if not command:
        raise ValueError("FFmpeg command cannot be empty")
        
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        if process.returncode != 0:
            raise FFmpegError(
                f"FFmpeg command failed with exit code {process.returncode}",
                command,
                stderr.decode() if stderr else None
            )
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()  # Ensure process is fully terminated
        except ProcessLookupError:
            pass  # Process already terminated
        raise FFmpegError(f"FFmpeg command timed out after {timeout} seconds", command)
    finally:
        # Ensure process resources are cleaned up
        if process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

def validate_and_prepare_media(args: Dict[str, Any], tmp_dir: Optional[str] = None) -> MediaFiles:
    """Validate and prepare media files from input arguments"""
    if tmp_dir is None:
        tmp_dir = f"tmp_{uuid.uuid4().hex[:8]}"
        
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        image_paths = []
        for idx, image_url in enumerate(args.get("images", []), start=1):
            if not image_url:
                continue
                
            try:
                image_filename = Path(image_url).name
                original_path = eden_utils.download_file(image_url, image_filename)
                
                extension = Path(original_path).suffix.lstrip('.')
                new_path = os.path.join(tmp_dir, f"img{idx}.{extension}")
                shutil.move(original_path, new_path)
                image_paths.append(new_path)
            except Exception as e:
                raise FFmpegError(f"Failed to download image {image_url}: {str(e)}", "")

        media_handlers = {}
        for media_type in ["video1", "video2", "audio1", "audio2"]:
            if url := args.get(media_type):
                try:
                    extension = Path(url).suffix.lstrip('.')
                    original_handler = eden_utils.get_file_handler(extension, url)
                    
                    new_path = os.path.join(tmp_dir, f"{media_type}.{extension}")
                    shutil.copy2(original_handler, new_path)
                    media_handlers[media_type] = new_path
                except Exception as e:
                    raise FFmpegError(f"Failed to handle {media_type} file: {str(e)}", "")

        return MediaFiles(
            images=image_paths,
            **media_handlers
        )
        
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

async def handler(args: Dict[str, Any]) -> Dict[str, str]:
    """Main handler function for processing media files and generating FFmpeg commands"""
    if not isinstance(args, dict):
        raise TypeError("Args must be a dictionary")
        
    n_retries = max(1, int(args.get("n_retries", 3)))
    timeout = max(1, int(args.get("timeout", 30)))
    
    tmp_dir = None
    output_path = None
    preserved_output = None
    
    try:
        if not args.get("task_instruction"):
            raise ValueError("Task instruction is required")

        tmp_dir = f"tmp_{uuid.uuid4().hex[:8]}"
        media = validate_and_prepare_media(args, tmp_dir)
        
        if not media.has_media():
            raise ValueError("At least one media file (image, video, or audio) must be provided")
        
        previous_attempt = None
        last_error = None
        
        for attempt in range(n_retries):
            try:
                ffmpeg_response = await generate_ffmpeg_command(
                    args["task_instruction"], 
                    media,
                    previous_attempt
                )
                
                print(f"Attempt {attempt + 1}/{n_retries}: Executing command: {ffmpeg_response.command}")
                
                # Store the output path
                output_path = ffmpeg_response.output_path
                
                await execute_ffmpeg_command(ffmpeg_response.command, timeout)
                
                # If the output file exists, move it to a preserved location
                if os.path.exists(output_path):
                    preserved_output = f"output_{uuid.uuid4().hex[:8]}_{Path(output_path).name}"
                    shutil.move(output_path, preserved_output)
                    return {"output": preserved_output}
                else:
                    raise FFmpegError("Output file was not generated", ffmpeg_response.command)
                
            except FFmpegError as e:
                last_error = e
                previous_attempt = {
                    "command": e.command,
                    "error": e.message,
                    **({"stderr": e.stderr} if e.stderr else {})
                }
                
                print(f"Attempt {attempt + 1}/{n_retries} failed: {str(e)}")
                continue
                
        if last_error:
            raise ValueError({
                "error": last_error.message,
                "command": last_error.command,
                **({"stderr": last_error.stderr} if last_error.stderr else {})
            })
            
    except Exception as e:
        if isinstance(e, FFmpegError):
            error_details = {
                "error": e.message,
                "command": e.command,
                **({"stderr": e.stderr} if e.stderr else {})
            }
            raise ValueError(error_details)
        raise ValueError(str(e))
        
    finally:
        if tmp_dir:
            # Make sure we don't delete the preserved output file if it exists
            if preserved_output and os.path.exists(preserved_output):
                try:
                    # If the output file is somehow still in the temp directory, remove it from there
                    if output_path and os.path.exists(output_path):
                        os.remove(output_path)
                except:
                    pass  # Ignore any errors during cleanup
            
            # Clean up the temporary directory
            shutil.rmtree(tmp_dir, ignore_errors=True)