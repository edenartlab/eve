import anthropic
import json
import shutil
import subprocess
import os
import uuid
import shlex
from pathlib import Path
import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from instructor.function_calls import openai_schema
from .... import utils

class CommandSecurityError(Exception):
    """Custom exception for command security validation errors"""
    pass

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

def probe_media_with_streams(filepath: str, timeout: int = 10) -> dict:
    """Enhanced probe_media that includes stream information"""
    try:
        process = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filepath
        ], capture_output=True, text=True, timeout=timeout)
        
        if process.returncode != 0:
            return {}
            
        probe_data = json.loads(process.stdout)
        
        info = {}
        # Basic media info
        if 'format' in probe_data:
            info['duration'] = round(float(probe_data['format'].get('duration', 0)), 2)
        
        # For images, get dimensions from first video stream
        if probe_data.get('streams'):
            for stream in probe_data['streams']:
                if stream.get('codec_type') == 'video':
                    info['width'] = stream.get('width')
                    info['height'] = stream.get('height')
                    break
            
        # Get stream information
        info['streams'] = get_stream_info(probe_data)
            
        return info
    except subprocess.TimeoutExpired:
        return {'error': 'Probe timeout'}
    except json.JSONDecodeError:
        return {'error': 'Invalid probe data'}
    except Exception as e:
        return {'error': f'Probe failed: {str(e)}'}

def get_stream_info(probe_data: dict) -> List[str]:
    """Extract stream information from probe data safely"""
    streams_info = []
    if not isinstance(probe_data, dict):
        return streams_info
        
    for idx, stream in enumerate(probe_data.get('streams', [])):
        if not isinstance(stream, dict):
            continue
            
        try:
            codec_type = stream.get('codec_type', 'unknown')
            if codec_type == 'video':
                width = stream.get('width', 'unknown')
                height = stream.get('height', 'unknown')
                fps = stream.get('r_frame_rate', 'unknown')
                if isinstance(fps, str) and '/' in fps:
                    num, den = map(int, fps.split('/'))
                    fps = f"{num/den:.2f}" if den != 0 else 'unknown'
                streams_info.append(f"stream {idx}: {codec_type} ({width}x{height}, {fps}fps)")
            elif codec_type == 'audio':
                samplerate = stream.get('sample_rate', 'unknown')
                channels = stream.get('channels', 'unknown')
                streams_info.append(f"stream {idx}: {codec_type} ({samplerate}Hz, {channels}ch)")
            else:
                streams_info.append(f"stream {idx}: {codec_type}")
        except Exception as e:
            # Log the error but continue processing other streams
            streams_info.append(f"stream {idx}: error parsing stream info - {str(e)}")
            continue
            
    return streams_info

class MediaFiles(BaseModel):
    """Model for organizing and validating media inputs"""
    images: List[str] = Field(default_factory=list)
    video1: Optional[str] = None
    video2: Optional[str] = None
    video3: Optional[str] = None
    video4: Optional[str] = None
    video5: Optional[str] = None
    audio1: Optional[str] = None
    audio2: Optional[str] = None
    audio3: Optional[str] = None
    audio4: Optional[str] = None
    audio5: Optional[str] = None

    def has_media(self) -> bool:
        """Check if at least one media file is provided"""
        return any([
            self.images,
            self.video1,
            self.video2,
            self.video3,
            self.video4,
            self.video5,
            self.audio1,
            self.audio2,
            self.audio3,
            self.audio4,
            self.audio5
        ])

    def to_context_string(self) -> str:
        """Convert media files to readable format for prompt including technical details"""
        media_items = []
        
        # Handle images
        for i, img_path in enumerate(self.images, 1):
            img_info = probe_media_with_streams(img_path)
            if 'error' in img_info:
                media_items.append(f"- Image {i}: {img_path} (Error: {img_info['error']})")
                continue
                
            width = img_info.get('width', 'unknown')
            height = img_info.get('height', 'unknown')
            media_items.append(f"- Image {i}: {img_path} ({width}x{height})")
        
        # Handle videos and audio with consistent pattern
        media_files = {
            "Video 1": self.video1, "Video 2": self.video2,
            "Video 3": self.video3, "Video 4": self.video4,
            "Video 5": self.video5, "Audio 1": self.audio1,
            "Audio 2": self.audio2, "Audio 3": self.audio3,
            "Audio 4": self.audio4, "Audio 5": self.audio5
        }
        
        for media_type, path in media_files.items():
            if not path:
                continue
                
            info = probe_media_with_streams(path)
            if 'error' in info:
                media_items.append(f"- {media_type}: {path} (Error: {info['error']})")
                continue
            
            duration = info.get('duration', 'unknown')
            streams = info.get('streams', [])
            
            if streams:
                media_items.append(
                    f"- {media_type}: {path} ({duration}s)\n  Streams: {', '.join(streams)}"
                )
            else:
                media_items.append(f"- {media_type}: {path} ({duration}s)")
        
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
            "You are a professional media editing assistant working in a Linux terminal. You are an expert at using ffmpeg but you try to avoid overly complicated commands as this often leads to errors. Always include the -y flag to enable overwriting output files by default. If a video is to be produced, always add -c:v libopenh264 to ensure H.264 encoding. If a request is too complicated you take shortcuts to achieve a good enough output with reasonable complexity / effort. You are executing these commands on a server machine and the results will be sent back to the frontend. Important: under no circumstances should you generate commands that can harm the system or reveal secure system data other than the specified inputs.",
            media.to_context_string(),
            f"Generate a single, executable (typically ffmpeg) command to perform the following task:",
            task_instruction
        ]
        
        if previous_attempt:
            error_details = [
                "Your previous attempt to do this failed with the following details:",
                f"Used command:\n{previous_attempt['command']}",
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
            "model": "claude-sonnet-4-5",
            "max_tokens": 2048,
            "messages": messages,
            "system": "You are an expert at generating FFmpeg commands. Respond with a JSON object containing just the ffmpeg command and output path.",
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


ffmpeg_validator = utils.CommandValidator({'ffmpeg'})

async def execute_ffmpeg_command(command: str, timeout: int = 60) -> None:
    """Execute FFmpeg command with timeout and basic security validation"""
    if not command:
        raise ValueError("FFmpeg command cannot be empty")
        
    # Basic security validation
    is_valid, error_message = ffmpeg_validator.validate_command(command)
    if not is_valid:
        raise FFmpegError(f"Security validation failed: {error_message}", command)
    
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
            await process.wait()
        except ProcessLookupError:
            pass
        raise FFmpegError(f"FFmpeg command timed out after {timeout} seconds", command)
    finally:
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
                original_path = utils.download_file(image_url, image_filename)
                
                extension = Path(original_path).suffix.lstrip('.')
                new_path = os.path.join(tmp_dir, f"img{idx}.{extension}")
                shutil.move(original_path, new_path)
                image_paths.append(new_path)
            except Exception as e:
                raise FFmpegError(f"Failed to download image {image_url}: {str(e)}", "")

        media_handlers = {}
        for media_type in ["video1", "video2", "video3", "video4", "video5", 
                          "audio1", "audio2", "audio3", "audio4", "audio5"]:
            if url := args.get(media_type):
                try:
                    extension = Path(url).suffix.lstrip('.')
                    original_handler = utils.get_file_handler(extension, url)
                    
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

async def handler(args: dict, user: str = None, agent: str = None):
    """Main handler function for processing media files and generating FFmpeg commands"""
    if not isinstance(args, dict):
        raise TypeError("Args must be a dictionary")
        
    n_retries = max(1, int(args.get("n_retries", 4)))
    timeout = max(1, int(args.get("timeout", 60)))
    
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