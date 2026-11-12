import asyncio
import json
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import anthropic
from instructor.function_calls import openai_schema
from pydantic import BaseModel, Field

from eve.tool import ToolContext

from .... import utils
from .sandbox import (
    SANDBOX_FONT_DIR,
    SANDBOX_FONT_FAMILIES,
    TIMEOUT_EXIT_CODE,
    FFmpegSandbox,
    OutputTooLargeError,
)


class CommandSecurityError(Exception):
    """Custom exception for command security validation errors"""


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
    output_path: str = Field(
        ..., description="The output filepath for the resulting file"
    )


async def probe_media_with_streams(filepath: str, timeout: int = 10) -> dict:
    """Enhanced probe_media that includes stream information (async to avoid blocking)"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            filepath,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"error": "Probe timeout"}

        if process.returncode != 0:
            return {}

        probe_data = json.loads(stdout.decode())

        info = {}
        # Basic media info
        if "format" in probe_data:
            info["duration"] = round(float(probe_data["format"].get("duration", 0)), 2)

        # For images, get dimensions from first video stream
        if probe_data.get("streams"):
            for stream in probe_data["streams"]:
                if stream.get("codec_type") == "video":
                    info["width"] = stream.get("width")
                    info["height"] = stream.get("height")
                    break

        # Get stream information
        info["streams"] = get_stream_info(probe_data)

        return info
    except json.JSONDecodeError:
        return {"error": "Invalid probe data"}
    except Exception as e:
        return {"error": f"Probe failed: {str(e)}"}


def get_stream_info(probe_data: dict) -> List[str]:
    """Extract stream information from probe data safely"""
    streams_info = []
    if not isinstance(probe_data, dict):
        return streams_info

    for idx, stream in enumerate(probe_data.get("streams", [])):
        if not isinstance(stream, dict):
            continue

        try:
            codec_type = stream.get("codec_type", "unknown")
            if codec_type == "video":
                width = stream.get("width", "unknown")
                height = stream.get("height", "unknown")
                fps = stream.get("r_frame_rate", "unknown")
                if isinstance(fps, str) and "/" in fps:
                    num, den = map(int, fps.split("/"))
                    fps = f"{num / den:.2f}" if den != 0 else "unknown"
                streams_info.append(
                    f"stream {idx}: {codec_type} ({width}x{height}, {fps}fps)"
                )
            elif codec_type == "audio":
                samplerate = stream.get("sample_rate", "unknown")
                channels = stream.get("channels", "unknown")
                streams_info.append(
                    f"stream {idx}: {codec_type} ({samplerate}Hz, {channels}ch)"
                )
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
    videos: List[str] = Field(default_factory=list)
    audios: List[str] = Field(default_factory=list)

    def has_media(self) -> bool:
        """Check if at least one media file is provided"""
        return any([self.images, self.videos, self.audios])

    async def to_context_string(self) -> str:
        """Convert media files to readable format for prompt including technical details"""
        media_items = []

        # Handle images
        for i, img_path in enumerate(self.images, 1):
            img_info = await probe_media_with_streams(img_path)
            if "error" in img_info:
                media_items.append(
                    f"- Image {i}: {img_path} (Error: {img_info['error']})"
                )
                continue

            width = img_info.get("width", "unknown")
            height = img_info.get("height", "unknown")
            media_items.append(f"- Image {i}: {img_path} ({width}x{height})")

        # Handle videos
        for i, video_path in enumerate(self.videos, 1):
            info = await probe_media_with_streams(video_path)
            if "error" in info:
                media_items.append(
                    f"- Video {i}: {video_path} (Error: {info['error']})"
                )
                continue

            duration = info.get("duration", "unknown")
            streams = info.get("streams", [])

            if streams:
                media_items.append(
                    f"- Video {i}: {video_path} ({duration}s)\n  Streams: {', '.join(streams)}"
                )
            else:
                media_items.append(f"- Video {i}: {video_path} ({duration}s)")

        # Handle audios
        for i, audio_path in enumerate(self.audios, 1):
            info = await probe_media_with_streams(audio_path)
            if "error" in info:
                media_items.append(
                    f"- Audio {i}: {audio_path} (Error: {info['error']})"
                )
                continue

            duration = info.get("duration", "unknown")
            streams = info.get("streams", [])

            if streams:
                media_items.append(
                    f"- Audio {i}: {audio_path} ({duration}s)\n  Streams: {', '.join(streams)}"
                )
            else:
                media_items.append(f"- Audio {i}: {audio_path} ({duration}s)")

        return "Available media files:\n" + "\n".join(media_items)


SYSTEM_CONTEXT = (
    "You are a professional media editing assistant working in a Linux terminal. "
    "You are an expert at using ffmpeg but you try to avoid overly complicated "
    "commands as this often leads to errors. Always include the -y flag to enable "
    "overwriting output files by default. If a video is to be produced, always add "
    "-c:v libx264 -pix_fmt yuv420p to ensure widely playable H.264 encoding (and "
    "-c:a aac when the output has audio). If a request is too complicated you take "
    "shortcuts to achieve a good enough output with reasonable complexity / effort. "
    "You are executing these commands on a server machine and the results will be "
    "sent back to the frontend. Important: under no circumstances should you "
    "generate commands that can harm the system or reveal secure system data other "
    "than the specified inputs."
)

# What the execution environment (see sandbox.py and the argument checks
# below) accepts, so the first attempt is not spent discovering it.
EXECUTION_RULES = (
    "How the command is executed:\n"
    "- It is one ffmpeg (or ffprobe) invocation, run directly without a shell: "
    "pipes, redirects, &&, ;, variables, globbing and command substitution do not "
    "work. Do everything in a single ffmpeg command.\n"
    "- It runs in an isolated working directory that contains only the media "
    "files listed above. Refer to them by exactly the relative paths shown, and "
    "write the result to a new relative filename (for example output.mp4).\n"
    "- Absolute paths, '..', '~', URLs and protocol prefixes (file:, concat:, "
    "pipe:) are rejected. There is no network access.\n"
    "- No other files exist and none can be created before the command runs, so "
    "do not use the concat demuxer with a list file (use the concat filter in "
    "-filter_complex) and do not reference subtitle, text or script files.\n"
    "- Exactly one output file is returned: the one you name in output_path.\n"
    f"- For text overlays use drawtext with font='{SANDBOX_FONT_FAMILIES[0]}' "
    f"(also installed: {', '.join(SANDBOX_FONT_FAMILIES[1:])}; bold via "
    f"fontfile={SANDBOX_FONT_DIR}/DejaVuSans-Bold.ttf). Fonts under "
    "/usr/share/fonts/ are the only absolute paths allowed."
)


async def generate_ffmpeg_command(
    task_instruction: str,
    media: MediaFiles,
    previous_attempt: Optional[Dict[str, str]] = None,
) -> FFmpegResponse:
    """Generate FFmpeg command using Anthropic API with JSON mode"""
    if not task_instruction:
        raise ValueError("Task instruction cannot be empty")

    if not media.has_media():
        raise ValueError("No media files provided")

    try:
        client = anthropic.AsyncAnthropic()

        media_context = await media.to_context_string()
        prompt_parts = [
            SYSTEM_CONTEXT,
            media_context,
            EXECUTION_RULES,
            "Generate a single, executable (typically ffmpeg) command to perform the following task:",
            task_instruction,
        ]

        if previous_attempt:
            error_details = [
                "Your previous attempt to do this failed with the following details:",
                f"Used command:\n{previous_attempt['command']}",
                f"Error: {previous_attempt['error']}",
            ]

            if stderr := previous_attempt.get("stderr"):
                error_details.append(f"stderr: {stderr}")

            error_details.append(
                "Please analyze the error and generate a new command that addresses the issue and completes the requested task."
            )
            prompt_parts.extend(error_details)

        prompt_parts.append(
            "Provide the command and output path in a JSON object with the following schema:"
            "{\n"
            '    "command": "string",  // The complete FFmpeg command\n'
            '    "output_path": "string"  // The output filepath for the resulting file\n'
            "}"
        )

        messages = [{"role": "user", "content": "\n\n".join(prompt_parts)}]

        prompt = {
            "model": "claude-sonnet-5",
            "max_tokens": 2048,
            "messages": messages,
            "system": "You are an expert at generating FFmpeg commands. Respond with a JSON object containing just the ffmpeg command and output path.",
            "tools": [openai_schema(FFmpegResponse).anthropic_schema],
            "tool_choice": {"type": "tool", "name": "FFmpegResponse"},
        }

        response = await client.messages.create(**prompt)

        if not response.content:
            raise ValueError("Empty response from Anthropic API")

        return FFmpegResponse(**response.content[0].input)

    except (json.JSONDecodeError, KeyError) as e:
        raise FFmpegError(f"Invalid response format: {str(e)}", "")
    except Exception as e:
        raise FFmpegError(f"Failed to generate FFmpeg command: {str(e)}", "")


ALLOWED_BINARIES = {"ffmpeg", "ffprobe"}

# The only absolute location a command may reference (fonts for drawtext).
ALLOWED_ABSOLUTE_PREFIXES = ("/usr/share/fonts/",)

# Characters that can precede a path or protocol inside an option value or a
# filtergraph (e.g. "movie=/x", "drawtext=textfile='/x'", "[in]/x").
_DELIM = r"(?:^|[\s=:,;'\"\[\]|])"
_ABSOLUTE_PATH = re.compile(_DELIM + r"(/[A-Za-z0-9_.~\-][^\s:,;'\"\[\]|]*)")
_HOME_PATH = re.compile(_DELIM + r"~")
_TRAVERSAL = re.compile(r"(?:^|[/\s=:,;'\"\[\]|])\.\.(?:$|[/\s:,;'\"\[\]|])")
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://")
# ffmpeg protocols that need no "//" (local file access and indirection).
_LOCAL_PROTOCOL = re.compile(
    _DELIM
    + r"(?:(?:file|concat|concatf|pipe|fd|data|subfile|crypto|cache|async|tee|unix):"
    # "subfile,,start,0,end,100,,:a.mp4" puts its options before the colon. Only
    # subfile takes a comma: "concat," is the concat filter with its defaults.
    + r"|subfile,)"
)


def check_ffmpeg_argument(arg: str) -> Optional[str]:
    """Return why an argument is refused, or None if it is acceptable.

    Inputs are downloaded into the job's working directory before the command
    runs, so a legitimate command only needs relative paths inside it. This
    is a second layer; the sandbox the command runs in is the real boundary.
    """
    if "\x00" in arg:
        return "invalid character"
    if arg.startswith("-/"):
        return "options that load their value from a file are not allowed"
    # Filtergraph escaping ("\/etc") must not hide a path from the checks.
    plain = arg.replace("\\", "")
    if _URL_SCHEME.search(plain):
        return "URLs are not allowed; use the provided local input files"
    if _LOCAL_PROTOCOL.search(plain):
        return "protocol prefixes (file:, concat:, pipe:, ...) are not allowed"
    if _TRAVERSAL.search(plain):
        return "'..' path segments are not allowed"
    if _HOME_PATH.search(plain):
        return "paths must be relative to the working directory"
    for match in _ABSOLUTE_PATH.finditer(plain):
        if not match.group(1).startswith(ALLOWED_ABSOLUTE_PREFIXES):
            return "absolute paths are not allowed; use relative paths"
    return None


def check_relative_path(path: str) -> str:
    """Validate a path that must stay inside the job's working directory."""
    if not path or not isinstance(path, str):
        raise FFmpegError("Output path is missing", "")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path.startswith("~"):
        raise FFmpegError(
            "Security validation failed: output path must be a relative path "
            "inside the working directory",
            "",
        )
    if "\x00" in path or "\\" in path or ":" in path:
        raise FFmpegError("Security validation failed: invalid output path", "")
    return str(pure)


def parse_ffmpeg_command(command: str) -> List[str]:
    """Split a generated command into an argument list for direct execution.

    The command is never handed to a shell: it is tokenised, the program must
    be exactly one of the allow-listed binaries (no path), and the tokens are
    passed to exec as-is, so shell syntax (pipes, redirects, substitution,
    chaining) has no effect. Arguments that reach outside the job's working
    directory (absolute paths, "..", URLs, protocol prefixes) are refused.
    """
    if not command or not isinstance(command, str):
        raise FFmpegError("FFmpeg command cannot be empty", command or "")
    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise FFmpegError(f"Invalid command syntax: {e}", command)
    if not argv or argv[0] not in ALLOWED_BINARIES:
        raise FFmpegError(
            "Security validation failed: only ffmpeg/ffprobe may be executed", command
        )
    for arg in argv[1:]:
        reason = check_ffmpeg_argument(arg)
        if reason:
            raise FFmpegError(f"Security validation failed: {reason}", command)
    return argv


async def execute_ffmpeg_command(sandbox: FFmpegSandbox, command: str) -> None:
    """Run a generated FFmpeg command in the isolated sandbox, without a shell.

    The command never runs in this process: this container holds service
    credentials, the sandbox holds none and has no network.
    """
    argv = parse_ffmpeg_command(command)
    if argv[0] == "ffmpeg":
        argv.insert(1, "-nostdin")

    try:
        returncode, stderr = await sandbox.run(argv)
    except FFmpegError:
        raise
    except Exception as e:
        raise FFmpegError(f"FFmpeg sandbox execution failed: {e}", command)

    if returncode == TIMEOUT_EXIT_CODE:
        raise FFmpegError(
            f"FFmpeg command was stopped at the time limit of "
            f"{sandbox.command_timeout} seconds",
            command,
            stderr,
        )
    if returncode != 0:
        raise FFmpegError(
            f"FFmpeg command failed with exit code {returncode}", command, stderr
        )


DEFAULT_EXTENSIONS = {"img": "jpg", "video": "mp4", "audio": "mp3"}


def _input_extension(source: str, kind: str) -> str:
    """A plain file extension for an input, from its URL path (no query string)."""
    suffix = Path(urlsplit(source).path).suffix.lstrip(".").lower()
    if re.fullmatch(r"[a-z0-9]{1,5}", suffix):
        return suffix
    return DEFAULT_EXTENSIONS[kind]


def _fetch_input(source: str, destination: str) -> None:
    """Place one input file at `destination`, inside the job's directory."""
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        utils.download_file(source, destination, overwrite=True)
    else:
        # Anything else must be a file in the temp directory; get_file_handler
        # refuses other local paths.
        shutil.copy2(utils.get_file_handler("", source), destination)


def validate_and_prepare_media(
    args: Dict[str, Any], tmp_dir: Optional[str] = None
) -> MediaFiles:
    """Download the inputs into the job's directory as img1.jpg, video1.mp4, ..."""
    if tmp_dir is None:
        tmp_dir = f"tmp_{uuid.uuid4().hex[:8]}"

    os.makedirs(tmp_dir, exist_ok=True)

    try:
        paths: Dict[str, List[str]] = {}
        for kind, arg_name, label in (
            ("img", "images", "image"),
            ("video", "videos", "video"),
            ("audio", "audios", "audio"),
        ):
            paths[kind] = []
            for idx, source in enumerate(args.get(arg_name) or [], start=1):
                if not source:
                    continue
                try:
                    extension = _input_extension(source, kind)
                    new_path = os.path.join(tmp_dir, f"{kind}{idx}.{extension}")
                    _fetch_input(source, new_path)
                    paths[kind].append(new_path)
                except Exception as e:
                    raise FFmpegError(f"Failed to download {label} {source}: {e}", "")

        return MediaFiles(
            images=paths["img"], videos=paths["video"], audios=paths["audio"]
        )

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


async def handler(context: ToolContext):
    """Main handler function for processing media files and generating FFmpeg commands"""
    if not isinstance(context.args, dict):
        raise TypeError("Args must be a dictionary")

    n_retries = max(1, int(context.args.get("n_retries", 4)))
    timeout = max(1, int(context.args.get("timeout", 600)))

    tmp_dir = None
    sandbox = None

    try:
        if not context.args.get("task_instruction"):
            raise ValueError("Task instruction is required")

        tmp_dir = f"tmp_{uuid.uuid4().hex[:8]}"
        media = await asyncio.to_thread(
            validate_and_prepare_media, context.args, tmp_dir
        )

        if not media.has_media():
            raise ValueError(
                "At least one media file (image, video, or audio) must be provided"
            )

        # Execution happens in a sandbox with no secrets and no network; only
        # the downloaded inputs go in and only the output file comes back.
        sandbox = FFmpegSandbox(command_timeout=timeout, attempts=n_retries)
        await sandbox.start(media.images + media.videos + media.audios)

        previous_attempt = None
        last_error = None

        for attempt in range(n_retries):
            try:
                ffmpeg_response = await generate_ffmpeg_command(
                    context.args["task_instruction"], media, previous_attempt
                )

                try:
                    output_path = check_relative_path(ffmpeg_response.output_path)
                except FFmpegError as e:
                    raise FFmpegError(e.message, ffmpeg_response.command)

                await execute_ffmpeg_command(sandbox, ffmpeg_response.command)

                # Copy the result out of the sandbox; the caller uploads it.
                preserved_output = (
                    f"output_{uuid.uuid4().hex[:8]}_{Path(output_path).name}"
                )
                try:
                    fetched = await sandbox.fetch(output_path, preserved_output)
                except OutputTooLargeError as e:
                    raise FFmpegError(str(e), ffmpeg_response.command)
                if fetched:
                    return {"output": preserved_output}
                raise FFmpegError(
                    f"The command finished but the output file {output_path} "
                    "does not exist or is empty",
                    ffmpeg_response.command,
                )

            except FFmpegError as e:
                last_error = e
                previous_attempt = {
                    "command": e.command,
                    "error": e.message,
                    **({"stderr": e.stderr} if e.stderr else {}),
                }

                continue

        if last_error:
            raise ValueError(
                {
                    "error": last_error.message,
                    "command": last_error.command,
                    **({"stderr": last_error.stderr} if last_error.stderr else {}),
                }
            )

    except Exception as e:
        if isinstance(e, FFmpegError):
            error_details = {
                "error": e.message,
                "command": e.command,
                **({"stderr": e.stderr} if e.stderr else {}),
            }
            raise ValueError(error_details)
        raise ValueError(str(e))

    finally:
        if sandbox:
            await sandbox.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
