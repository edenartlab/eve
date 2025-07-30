from __future__ import annotations
import os
import re
import json
import time
import math
import magic
import httpx
import random
import base64
import asyncio
import pathlib
import textwrap
import requests
import tempfile
import blurhash
import subprocess
import replicate
import boto3
import shlex
import subprocess
import numpy as np
from jinja2 import Template
from bson import ObjectId
from datetime import datetime, date
from pprint import pformat
from typing import Union, Tuple, Set, List, Optional, Dict

try:
    # MoviePy 2.x
    from moviepy import VideoFileClip, ImageClip, AudioClip
except ImportError:
    # MoviePy 1.x
    from moviepy.editor import VideoFileClip, ImageClip, AudioClip

from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import s3

USE_MEDIA_CACHE = True


class CommandValidator:
    """Simple validator to ensure basic command security"""

    # Most dangerous shell operations that shouldn't appear in legitimate commands
    DANGEROUS_OPERATIONS = [
        "&&",  # Command chaining
        "||",  # Command chaining
        " ; ",  # Command chaining (with spaces to avoid ffmpeg filter syntax)
        ";\\n",  # Command chaining (newline variant)
        "$(",
        "`",  # Command substitution
        "> /",  # Writing to root
        ">>/",  # Appending to root
        "sudo ",  # Privilege escalation (with space to avoid false positives)
        "| rm",  # Pipe to remove
        "| sh",  # Pipe to shell
        "| bash",  # Pipe to shell
        "eval ",  # Command evaluation (with space)
        "exec ",  # Command execution (with space)
    ]

    def __init__(self, allowed_commands: Set[str]):
        """
        Initialize the command validator.

        Args:
            allowed_commands: Set of base commands that are allowed to be executed
        """
        self.allowed_commands = {cmd.lower() for cmd in allowed_commands}

    def validate_command(self, command: str) -> Tuple[bool, Union[str, None]]:
        """
        Validates that a command is safe to execute.
        Only checks for base command and the most dangerous shell operations.

        Args:
            command: The command string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not isinstance(command, str):
            return False, "Command must be a non-empty string"

        # Try to parse command into tokens and get base command
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "Empty command"
            base_cmd = os.path.basename(tokens[0]).lower()
        except ValueError as e:
            return False, f"Invalid command syntax: {str(e)}"

        # Verify base command is allowed
        if base_cmd not in self.allowed_commands:
            return False, f"Command '{base_cmd}' is not in the allowed list"

        # Check for dangerous operations
        for pattern in self.DANGEROUS_OPERATIONS:
            if pattern in command:
                return False, f"Command contains dangerous operation: {pattern}"

        return True, None


def log_memory_info():
    """
    Log basic GPU, RAM, and disk usage percentages using nvidia-smi for GPU metrics.
    """
    import psutil
    import shutil
    import subprocess

    print("\n=== Memory Usage ===")

    # GPU VRAM using nvidia-smi
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ]
        )
        total_mem, used_mem = map(int, result.decode("utf-8").strip().split(","))
        gpu_percent = (used_mem / total_mem) * 100
        print(f"GPU Memory: {gpu_percent:.1f}% of {total_mem / 1024:.1f}GB")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GPU info not available")

    # System RAM
    ram = psutil.virtual_memory()
    print(f"RAM Usage: {ram.percent}% of {ram.total / (1024**3):.1f}GB")

    # Disk usage (root directory)
    usage = shutil.disk_usage("/root")
    disk_percent = (usage.used / usage.total) * 100
    print(f"Disk Usage: {disk_percent:.1f}% of {usage.total / (1024**3):.1f}GB")
    print("==================\n")


def prepare_result(result, summarize=False):
    if isinstance(result, dict):
        if "error" in result and result["error"] is not None:
            return result
        if "mediaAttributes" in result:
            result["mediaAttributes"].pop("blurhash", None)
        if "filename" in result:
            filename = result.pop("filename")
            url = s3.get_full_url(filename)
            if summarize:
                return url
            else:
                result["url"] = url
        return {k: prepare_result(v, summarize) for k, v in result.items()}
    elif isinstance(result, list):
        return [prepare_result(item, summarize) for item in result]
    else:
        return result


def upload_result(result, save_thumbnails=False, save_blurhash=False):
    if isinstance(result, dict):
        exlude_result_processing_keys = ["subtool_calls"]
        return {
            k: upload_result(
                v, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
            )
            if k not in exlude_result_processing_keys
            else v
            for k, v in result.items()
        }
    elif isinstance(result, list):
        return [
            upload_result(
                item, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
            )
            for item in result
        ]
    elif is_downloadable_file(result):
        return upload_media(
            result, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
        )
    else:
        return result


def upload_media(output, save_thumbnails=True, save_blurhash=True):
    file_url, sha = s3.upload_file(output)
    filename = file_url.split("/")[-1]

    media_attributes, thumbnail = get_media_attributes(output)

    if save_thumbnails and thumbnail:
        for width in [384, 768, 1024, 2560]:
            img = thumbnail.copy()
            img.thumbnail(
                (width, 2560), Image.Resampling.LANCZOS
            ) if width < thumbnail.width else thumbnail
            img_bytes = PIL_to_bytes(img)
            s3.upload_buffer(img_bytes, name=f"{sha}_{width}", file_type=".webp")
            # s3.upload_buffer(img_bytes, name=f"{sha}_{width}", file_type=".jpg")
    if save_blurhash and thumbnail:
        try:
            img = thumbnail.copy()
            img.thumbnail((100, 100), Image.LANCZOS)
            media_attributes["blurhash"] = blurhash.encode(np.array(img), 4, 4)
        except Exception as e:
            print(f"Error encoding blurhash: {e}")

    return {"filename": filename, "mediaAttributes": media_attributes}


def get_media_attributes(file):
    if isinstance(file, replicate.helpers.FileOutput):
        is_url = False
        file_content = file.read()
        mime_type = magic.from_buffer(file_content, mime=True)
        file = BytesIO(file_content)
    else:
        is_url = file.startswith("http://") or file.startswith("https://")
        if is_url:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file = download_file(file, temp_file.name, overwrite=False)
        mime_type = magic.from_file(file, mime=True)

    thumbnail = None
    media_attributes = {
        "mimeType": mime_type,
    }

    if "image" in mime_type:
        image = Image.open(file)
        thumbnail = image.copy()
        width, height = thumbnail.size
        media_attributes.update(
            {"width": width, "height": height, "aspectRatio": width / height}
        )

    elif "video" in mime_type:
        video = VideoFileClip(file)
        thumbnail = Image.fromarray(video.get_frame(0).astype("uint8"), "RGB")
        width, height = thumbnail.size
        media_attributes.update(
            {
                "width": width,
                "height": height,
                "aspectRatio": width / height,
                "duration": video.duration,
            }
        )
        video.close()

    elif "audio" in mime_type:
        media_attributes.update({"duration": get_media_duration(file)})

    if is_url:
        os.remove(file)

    return media_attributes, thumbnail


def _check_volume_cache(url, local_filepath, overwrite=False):
    """
    Check if file exists in Modal Volume cache and copy it if found.

    Args:
        url: Original URL being downloaded
        local_filepath: Target local path
        overwrite: Whether to overwrite existing files

    Returns:
        str: Path to file if found in cache, None otherwise
    """
    try:
        # Check if we're in Modal environment with volume access
        volume_cache_dir = pathlib.Path("/data/media-cache")
        if not volume_cache_dir.exists():
            return None

        # Generate cache key from URL filename (filenames are designed to be unique)
        cache_filename = pathlib.Path(url).name
        # Remove query parameters from filename
        cache_filename = re.sub(r"\?.*$", "", cache_filename)

        cache_filepath = volume_cache_dir / cache_filename

        if cache_filepath.exists():
            print(
                f"<**> Found {cache_filename} in volume cache, copying to {local_filepath}"
            )
            # Copy from cache to target location
            import shutil

            shutil.copy2(str(cache_filepath), str(local_filepath))
            return str(local_filepath)

    except Exception as e:
        # If volume access fails, silently continue with normal download
        print(f"Volume cache check failed: {e}")

    return None


def _save_to_volume_cache(url, local_filepath):
    """
    Save downloaded file to Modal Volume cache for future use.

    Args:
        url: Original URL that was downloaded
        local_filepath: Path to the downloaded file
    """
    try:
        # Check if we're in Modal environment with volume access
        volume_cache_dir = pathlib.Path("/data/media-cache")
        if not volume_cache_dir.exists():
            return

        volume_cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache key from URL filename
        cache_filename = pathlib.Path(url).name
        # Remove query parameters from filename
        cache_filename = re.sub(r"\?.*$", "", cache_filename)

        cache_filepath = volume_cache_dir / cache_filename

        if not cache_filepath.exists():
            print(f"<**> Saving {cache_filename} to volume cache")
            import shutil

            shutil.copy2(str(local_filepath), str(cache_filepath))

            # Commit volume changes if we have access to modal
            try:
                import modal
                # Try to get the volume and commit changes
                # This will only work if we're in a Modal function with volume access
                # We can't directly access the volume object, so we rely on Modal's auto-commit
            except:
                pass

    except Exception as e:
        # If volume access fails, silently continue
        print(f"Volume cache save failed: {e}")


def download_file(url, local_filepath, overwrite=False):
    """
    Download a file from a URL to a local filepath, with special handling for AWS S3 URLs.
    Uses Modal Volume caching when available to avoid re-downloading files.

    Args:
        url: URL to download from
        local_filepath: Local path to save the file to
        overwrite: Whether to overwrite existing files

    Returns:
        str: Path to the downloaded file
    """
    local_filepath = pathlib.Path(local_filepath)
    local_filepath.parent.mkdir(parents=True, exist_ok=True)

    if local_filepath.exists() and not overwrite:
        print(f"File {local_filepath} already exists. Skipping download.")
        return str(local_filepath)

    # Check for Modal Volume cache
    if USE_MEDIA_CACHE:
        cache_path = _check_volume_cache(url, local_filepath, overwrite)
        if cache_path:
            return cache_path

    print(f"Downloading file from {url} to {local_filepath}")

    try:
        # Parse S3 URL to extract bucket and key
        s3_pattern = r"https://([^.]+)\.s3(?:\.([^.]+))?\.amazonaws\.com/(.+)"
        s3_match = re.match(s3_pattern, url)

        if s3_match:
            # This is an S3 URL
            bucket_name = s3_match.group(1)
            region = s3_match.group(2) or os.getenv("AWS_REGION_NAME", "us-east-1")
            key = s3_match.group(3)

            print(
                f"Detected S3 URL - Bucket: {bucket_name}, Region: {region}, Key: {key}"
            )

            # Use boto3 to download with credentials
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=region,
            )

            try:
                print(f"Downloading {key} from S3 bucket {bucket_name}")
                s3_client.download_file(bucket_name, key, str(local_filepath))
                if USE_MEDIA_CACHE:
                    # Save to volume cache after successful download
                    _save_to_volume_cache(url, local_filepath)
                return str(local_filepath)
            except Exception as s3_error:
                print(f"S3 download error: {s3_error}")
                # Fall back to standard HTTP request below

        # For CloudFront or standard HTTP requests
        with httpx.stream("GET", url, follow_redirects=True) as response:
            if response.status_code == 404:
                raise FileNotFoundError(f"No file found at {url}")
            if response.status_code != 200:
                raise Exception(
                    f"Failed to download from {url}. Status code: {response.status_code}"
                )

            # Get content length if available
            total = int(response.headers.get("Content-Length", "0"))

            if total == 0:
                # If Content-Length not provided, read all at once
                content = response.read()
                with open(local_filepath, "wb") as f:
                    f.write(content)
            else:
                # Stream with progress bar if Content-Length available
                with (
                    open(local_filepath, "wb") as f,
                    tqdm(
                        total=total, unit_scale=True, unit_divisor=1024, unit="B"
                    ) as progress,
                ):
                    num_bytes_downloaded = response.num_bytes_downloaded
                    for data in response.iter_bytes():
                        f.write(data)
                        progress.update(
                            response.num_bytes_downloaded - num_bytes_downloaded
                        )
                        num_bytes_downloaded = response.num_bytes_downloaded

        # Save to volume cache after successful download
        _save_to_volume_cache(url, local_filepath)
        return str(local_filepath)

    except httpx.HTTPStatusError as e:
        raise Exception(f"HTTP error: {e}")
    except Exception as e:
        raise Exception(f"Error downloading file: {e}")


def exponential_backoff(
    func,
    max_attempts=5,
    initial_delay=1,
    max_jitter=1,
):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(-max_jitter, max_jitter)
            print(
                f"Attempt {attempt} failed because: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay + jitter)
            delay = delay * 2


async def async_exponential_backoff(
    func,
    max_attempts=5,
    initial_delay=1,
    max_jitter=1,
):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(-max_jitter, max_jitter)
            print(
                f"Attempt {attempt} failed because: {e}. Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay + jitter)
            delay = delay * 2


def mock_image(args):
    image = Image.new("RGB", (300, 300), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped_text = textwrap.fill(str(args), width=50)
    draw.text((5, 5), wrapped_text, fill="black", font=font)
    image = image.resize((512, 512), Image.LANCZOS)
    buffer = PIL_to_bytes(image)
    url, _ = s3.upload_buffer(buffer)
    return url


def get_media_duration(media_file):
    # If it's a BytesIO object, we need to save it to a temporary file first
    if isinstance(media_file, BytesIO):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(media_file.getvalue())
        temp_file.close()
        media_file_path = temp_file.name
    else:
        media_file_path = media_file

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_file_path,
    ]

    try:
        duration = subprocess.check_output(cmd).decode().strip()
        result = float(duration)
    finally:
        # Clean up temporary file if we created one
        if isinstance(media_file, BytesIO) and os.path.exists(media_file_path):
            os.unlink(media_file_path)

    return result


def get_font(font_name, font_size):
    font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
    font = ImageFont.truetype(font_path, font_size)
    return font


def text_to_lines(text):
    pattern = r"^\d+[\.:]\s*\"?"
    lines = [line for line in text.split("\n") if line]
    lines = [re.sub(pattern, "", line, flags=re.MULTILINE) for line in lines]
    return lines


def download_image_to_PIL(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def PIL_to_bytes(image, ext="JPEG", quality=95):
    if image.mode == "RGBA" and ext.upper() not in ["PNG", "WEBP"]:
        image = image.convert("RGB")
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=ext, quality=quality)
    return img_byte_arr.getvalue()


def image_to_base64(file_path, max_size, quality=95, truncate=False):
    mime_type = magic.from_file(file_path, mime=True)
    if "video" in mime_type:
        # Extract the first frame image as thumbnail
        video = VideoFileClip(file_path)
        img = Image.fromarray(video.get_frame(0).astype("uint8"), "RGB")
        video.close()
    else:
        img = Image.open(file_path)
    if isinstance(max_size, (int, float)):
        w, h = img.size
        ratio = min(1.0, ((max_size**2) / (w * h)) ** 0.5)
        max_size = int(w * ratio), int(h * ratio)
    img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img_bytes = PIL_to_bytes(img, ext="JPEG", quality=quality)
    data = base64.b64encode(img_bytes).decode("utf-8")
    if truncate:
        data = data[:64] + data[-16:] + "..."
    return data


from safetensors.torch import load_file, save_file


def convert_pti_to_safetensors(input_path: str, output_path: str):
    try:
        data = load_file(input_path)
    except Exception as e:
        print(f"❌ Failed to load {input_path} with safetensors: {e}")
        return False

    # Expected mapping from .pti → .safetensors SDXL format
    key_map = {
        "text_encoders_0": "clip_l",  # 768-dim
        "text_encoders_1": "clip_g",  # 1280-dim
    }

    remapped = {}
    for k, v in data.items():
        if k not in key_map:
            print(
                f"⚠️ Unexpected key '{k}' in {input_path}. Expected only {list(key_map.keys())}. Skipping this key."
            )
            continue

        new_key = key_map[k]
        remapped[new_key] = v

    if not remapped:
        print(
            f"❌ No valid keys found for conversion in {input_path}. Output file not saved."
        )
        return False

    try:
        save_file(remapped, output_path)
        print(f"✅ Converted {input_path} → {output_path}")
        for k, v in remapped.items():
            print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
        return True
    except Exception as e:
        print(f"❌ Failed to save converted file to {output_path}: {e}")
        return False


def deep_filter(current, changes):
    if not isinstance(current, dict) or not isinstance(changes, dict):
        return changes if changes != current else None
    result = {}
    for key, value in changes.items():
        if key in current:
            if isinstance(current[key], dict) and isinstance(value, dict):
                filtered = deep_filter(current[key], value)
                if filtered:
                    result[key] = filtered
            elif current[key] != value:
                result[key] = value
        else:
            result[key] = value
    return result if result else None


def deep_update(data, changes):
    if not isinstance(data, dict) or not isinstance(changes, dict):
        return changes
    for key, value in changes.items():
        if key in data:
            if isinstance(data[key], dict) and isinstance(value, dict):
                deep_update(data[key], value)
            elif data[key] != value:
                data[key] = value
        else:
            data[key] = value
    return data


def calculate_target_dimensions(images, max_pixels):
    min_w = float("inf")
    min_h = float("inf")

    total_aspect_ratio = 0.0

    for image_url in images:
        image = download_image_to_PIL(image_url)
        width, height = image.size
        min_w = min(min_w, width)
        min_h = min(min_h, height)
        total_aspect_ratio += width / height

    avg_aspect_ratio = total_aspect_ratio / len(images)

    if min_w / min_h > avg_aspect_ratio:
        target_height = min_h
        target_width = round(target_height * avg_aspect_ratio)
    else:
        target_width = min_w
        target_height = round(target_width / avg_aspect_ratio)

    if target_width * target_height > max_pixels:
        ratio = (target_width * target_height) / max_pixels
        ratio = math.sqrt((target_width * target_height) / max_pixels)
        target_width = round(target_width / ratio)
        target_height = round(target_height / ratio)

    target_width -= target_width % 2
    target_height -= target_height % 2

    return target_width, target_height


def resize_and_crop(image, width, height):
    target_ratio = width / height
    orig_width, orig_height = image.size
    orig_ratio = orig_width / orig_height

    if orig_ratio > target_ratio:
        new_width = int(target_ratio * orig_height)
        left = (orig_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = orig_height
    else:
        new_height = int(orig_width / target_ratio)
        top = (orig_height - new_height) // 2
        left = 0
        bottom = top + new_height
        right = orig_width

    image = image.crop((left, top, right, bottom))
    image = image.resize((width, height), Image.LANCZOS)

    return image


def create_dialogue_thumbnail(image1_url, image2_url, width, height, ext="WEBP"):
    image1 = download_image_to_PIL(image1_url)
    image2 = download_image_to_PIL(image2_url)

    half_width = width // 2

    image1 = resize_and_crop(image1, half_width, height)
    image2 = resize_and_crop(image2, half_width, height)

    combined_image = Image.new("RGB", (width, height))

    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (half_width, 0))

    img_byte_arr = BytesIO()
    combined_image.save(img_byte_arr, format=ext)

    return img_byte_arr.getvalue()


def concatenate_videos(video_files, output_file, fps=30):
    converted_videos = []
    for video in video_files:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            output_video = temp.name
            convert_command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "panic",
                "-i",
                video,
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                output_video,
            ]
            subprocess.run(convert_command)
            converted_videos.append(output_video)
    filter_complex = "".join(
        [f"[{i}:v] [{i}:a] " for i in range(len(converted_videos))],
    )
    filter_complex += f"concat=n={len(converted_videos)}:v=1:a=1 [v] [a]"
    concat_command = ["ffmpeg"]
    for video in converted_videos:
        concat_command.extend(["-i", video])
    concat_command.extend(
        [
            "-y",
            "-loglevel",
            "panic",
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            output_file,
        ],
    )
    subprocess.run(concat_command)
    for video in converted_videos:
        os.remove(video)


def get_file_handler(suffix, input_data):
    if isinstance(input_data, str) and os.path.exists(input_data):
        return input_data
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    if isinstance(input_data, str) and input_data.startswith("http"):
        download_file(input_data, temp_file.name, overwrite=True)
    elif isinstance(input_data, bytes):
        temp_file.write(input_data)
    elif isinstance(input_data, BytesIO):
        temp_file.write(input_data.getvalue())
    else:
        raise ValueError("input_data must be either a URL string or a BytesIO object")
    temp_file.close()
    return temp_file.name


def make_audiovideo_clip(video_input, audio_input):
    video_file = get_file_handler(".mp4", video_input)
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    if audio_input:
        audio_file = get_file_handler(".mp3", audio_input)
        audio_duration = get_media_duration(audio_file)

        # loop the video to match the audio duration
        looped_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "panic",
            "-stream_loop",
            "-1",
            "-i",
            video_file,
            "-c",
            "copy",
            "-t",
            str(audio_duration),
            looped_video.name,
        ]
        subprocess.run(cmd)

        # merge the audio and the looped video
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "panic",
            "-i",
            looped_video.name,
            "-i",
            audio_file,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-shortest",
            output_file.name,
        ]

    else:
        # if no audio, create a silent audio track with same duration as video
        video_duration = get_media_duration(video_file)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "panic",
            "-i",
            video_file,
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={video_duration}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_file.name,
        ]

    subprocess.run(cmd)

    return output_file.name


def add_audio_to_audiovideo(video_input, audio_input, output_path):
    video_file = get_file_handler(".mp4", video_input)
    audio_file = get_file_handler(".mp3", audio_input)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_file,
        "-i",
        audio_file,
        "-filter_complex",
        "[1:a]volume=1.0[a1];[0:a][a1]amerge=inputs=2[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-ac",
        "2",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def stitch_image_video(image_file: str, video_file: str, image_left: bool = False):
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    if image_left:
        filter_complex = '"[1:v][0:v]scale2ref[img][vid];[img]setpts=PTS-STARTPTS[imgp];[vid]setpts=PTS-STARTPTS[vidp];[imgp][vidp]hstack"'
    else:
        filter_complex = '"[0:v][1:v]scale2ref[vid][img];[vid]setpts=PTS-STARTPTS[vidp];[img]setpts=PTS-STARTPTS[imgp];[vidp][imgp]hstack"'

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-i",
        video_file,
        "-i",
        image_file,
        "-filter_complex",
        filter_complex,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_file.name,
    ]
    subprocess.run(cmd)

    return output_file.name


def process_in_parallel(array, func, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(func, item, index): index
            for index, item in enumerate(array)
        }
        results = [None] * len(array)
        for future in as_completed(futures):
            try:
                index = futures[future]
                results[index] = future.result()
            except Exception as e:
                print(f"Task error: {e}")
                for f in futures:
                    f.cancel()
                raise e
    return results


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        if draw.textlength(" ".join(current_line + [word]), font=font) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def video_textbox(
    paragraphs: list[str],
    width: int,
    height: int,
    duration: float,
    fade_in: float,
    font_size: int = 36,
    font_ttf: str = "Arial.ttf",
    margin_left: int = 25,
    margin_right: int = 25,
    line_spacing: float = 1.25,
):
    font = get_font(font_ttf, font_size)

    canvas = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([(0, 0), (width, height)], fill="black")

    y = 100
    for text in paragraphs:
        wrapped_text = wrap_text(draw, text, font, width - margin_left - margin_right)
        for line in wrapped_text:
            draw.text((margin_left, y), line, fill="white", font=font)
            y += int(line_spacing * font.size)
        y += int(line_spacing * font.size)

    image_np = np.array(canvas)
    clip = ImageClip(image_np, duration=duration)
    clip = clip.fadein(fade_in).fadeout(fade_in)

    # Create a silent audio clip and set it as the audio of the video clip
    silent_audio = AudioClip(lambda t: [0, 0], duration=duration, fps=44100)
    clip = clip.set_audio(silent_audio)

    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    clip.write_videofile(output_file.name, fps=30, codec="libx264", audio_codec="aac")

    return output_file.name


def concat_sentences(*sentences):
    return " ".join([s.strip().rstrip(".") + "." for s in sentences if s and s.strip()])


def is_downloadable_file(value):
    return isinstance(value, replicate.helpers.FileOutput) or (
        isinstance(value, str)
        and (
            os.path.isfile(value)  # is a file
            or (  # is a url but not from twitter
                value.startswith(("http://", "https://"))
                and "x.com" not in value
                and "pbs.twimg.com" not in value
            )
        )
    )


def get_human_readable_error(error_list):
    errors = [f"{error['loc'][0]}: {error['msg']}" for error in error_list]
    error_str = "\n\t".join(errors)
    error_str = f"Invalid args\n\t{error_str}"
    return error_str


def pprint(*args, color=None, indent=4):
    colors = {
        "red": "\033[38;2;255;100;100m",
        "green": "\033[38;2;100;255;100m",
        "blue": "\033[38;2;100;100;255m",
        "yellow": "\033[38;2;255;255;100m",
        "magenta": "\033[38;2;255;100;255m",
        "cyan": "\033[38;2;100;255;255m",
    }
    if not color:
        color = random.choice(list(colors.keys()))
    if color not in colors:
        raise ValueError(f"Invalid color: {color}")
    for arg in args:
        string = pformat(arg, indent=indent)
        colored_output = f"{colors[color]}{string}\033[0m"
        print(colored_output)


def random_string(length=28):
    # modeled after Replicate id
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))


def save_test_results(tools, results):
    if not results:
        return

    results_dir = os.path.join(
        "tests", "out", f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(results_dir, exist_ok=True)

    for tool, tool_result in zip(tools.keys(), results):
        if isinstance(tool_result, dict) and tool_result.get("error"):
            file_path = os.path.join(results_dir, f"{tool}_ERROR.txt")
            with open(file_path, "w") as f:
                f.write(tool_result["error"])
        else:
            outputs = tool_result.get("output", [])
            outputs = outputs if isinstance(outputs, list) else [outputs]
            intermediate_outputs = tool_result.get("intermediate_outputs", {})

            for o, output in enumerate(outputs):
                if "url" in output:
                    ext = output.get("url").split(".")[-1]
                    filename = (
                        f"{tool}_{o}.{ext}" if len(outputs) > 1 else f"{tool}.{ext}"
                    )
                    file_path = os.path.join(results_dir, filename)
                    response = requests.get(output.get("url"))
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                else:
                    filename = f"{tool}_{o}.txt" if len(outputs) > 1 else f"{tool}.txt"
                    file_path = os.path.join(results_dir, filename)
                    with open(file_path, "w") as f:
                        f.write(output)

            for k, v in intermediate_outputs.items():
                if "url" in v:
                    ext = v.get("url").split(".")[-1]
                    filename = f"{tool}_{k}.{ext}"
                    file_path = os.path.join(results_dir, filename)
                    response = requests.get(v.get("url"))
                    with open(file_path, "wb") as f:
                        f.write(response.content)
    print(f"Test results saved to {results_dir}")


def serialize_json(obj, *, indent=None, exclude=None):
    """Return *obj* as a JSON string."""

    def scrub(value):
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            pruned = {k: v for k, v in value.items() if not exclude or k not in exclude}
            return {k: scrub(v) for k, v in pruned.items()}
        if isinstance(value, (list, tuple, set)):
            return [scrub(item) for item in value]
        return value

    if obj is None:
        return ""

    cleaned = scrub(obj)
    return cleaned


def dumps_json(obj, *, indent=None, exclude=None):
    cleaned = serialize_json(obj, indent=indent, exclude=exclude)
    return json.dumps(cleaned, indent=indent)


def overwrite_dict(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            overwrite_dict(base[key], value)
        else:
            base[key] = value


def load_template(filename: str) -> Template:
    """Load and compile a template from the templates directory"""
    TEMPLATE_DIR = pathlib.Path(__file__).parent / "prompt_templates"
    template_path = TEMPLATE_DIR / f"{filename}.txt"
    with open(template_path) as f:
        return Template(f.read())


CLICK_COLORS = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
]


from typing import Any, Dict
from pyparsing import (
    Forward,
    Keyword,
    Literal,
    ParserElement,
    QuotedString,
    Suppress,
    Word,
    alphanums,
    alphas,
    infixNotation,
    oneOf,
    opAssoc,
    pyparsing_common,
)

# Enable memoisation to speed up recursive parsing.  This has a global effect but does not interfere with other parsers in normal usage.
ParserElement.enablePackrat()


def _build_expression_parser(variables: Dict[str, Any]) -> ParserElement:
    """Construct a pyparsing parser for evaluating JS‑style expressions.

    The returned parser will substitute values from ``variables`` when
    encountering identifiers.  Evaluation is performed by parse actions
    attached to the grammar elements; no further processing is necessary
    once parsing succeeds.

    Parameters
    ----------
    variables : Dict[str, Any]
        A mapping of variable names to Python values used when
        evaluating the expression.

    Returns
    -------
    ParserElement
        A parser configured to evaluate an expression and return the
        resulting Python value.
    """
    # Define number literals.  Use copy() because the default instances
    # attach parse actions that return numbers in lists; copy() preserves
    # these semantics while allowing us to attach our own parse action.
    integer = pyparsing_common.signed_integer.copy().setParseAction(
        lambda t: [int(t[0])]
    )
    real = pyparsing_common.fnumber.copy().setParseAction(lambda t: [float(t[0])])
    number = real | integer

    # Define string literals (single or double quoted).  These return
    # Python str values when parsed.  The escChar parameter ensures
    # backslash escapes are processed correctly.
    single_quoted = QuotedString("'", escChar="\\").setParseAction(
        lambda t: [str(t[0])]
    )
    double_quoted = QuotedString('"', escChar="\\").setParseAction(
        lambda t: [str(t[0])]
    )
    string = single_quoted | double_quoted

    # Define booleans and null/None.  Keywords are case‑insensitive.
    true_literal = (Keyword("true", caseless=True) | Keyword("True")).setParseAction(
        lambda: [True]
    )
    false_literal = (Keyword("false", caseless=True) | Keyword("False")).setParseAction(
        lambda: [False]
    )
    null_literal = (Keyword("null", caseless=True) | Keyword("None")).setParseAction(
        lambda: [None]
    )

    # Identifiers: variable names consisting of letters, digits and
    # underscores.  When encountered, look up the value in ``variables``.
    # The parse action returns a single‑element list containing the
    # variable's value.  It is important to return a list rather than
    # ``None`` directly; returning ``None`` from a parse action tells
    # pyparsing to delete that token from the result, which leads to
    # incorrect evaluation when a variable's value is ``None``.
    ident = Word(alphas + "_", alphanums + "_").setParseAction(
        lambda t: [variables.get(t[0], None)]
    )

    # Forward declarations for recursive grammar elements.
    expr: Forward = Forward()
    operand: Forward = Forward()

    # Atomic operands: numbers, strings, booleans, null/None or identifiers.
    atom = number | string | true_literal | false_literal | null_literal | ident

    # Parenthesised expressions.  Suppress the parentheses so they do not
    # clutter the parse result.  The enclosed expression is parsed
    # recursively by referencing ``expr``.
    operand <<= atom | (Suppress("(") + expr + Suppress(")"))

    # Define evaluation functions for unary and binary operators.  The
    # parse actions receive a nested list structure representing the
    # operator and its operand(s) and must return a single evaluated
    # Python value.
    def unary_eval(tokens):
        op = tokens[0][0]
        val = tokens[0][1]
        if op == "-":
            return -val
        if op == "+":
            return +val
        if op in ("!", "not"):
            return not val
        raise ValueError(f"Unknown unary operator: {op}")

    def binary_eval(tokens):
        values = tokens[0]
        result = values[0]
        for i in range(1, len(values), 2):
            op = values[i]
            right = values[i + 1]
            if op == "+":
                result = result + right
            elif op == "-":
                result = result - right
            elif op == "*":
                result = result * right
            elif op == "/":
                result = result / right
            elif op == "%":
                result = result % right
            elif op == "<":
                result = result < right
            elif op == "<=":
                result = result <= right
            elif op == ">":
                result = result > right
            elif op == ">=":
                result = result >= right
            elif op in ("==", "==="):
                # JavaScript's == and === are both treated as Python ==.
                result = result == right
            elif op in ("!=", "!=="):
                result = result != right
            elif op in ("and", "&&"):
                result = result and right
            elif op in ("or", "||"):
                result = result or right
            else:
                raise ValueError(f"Unknown binary operator: {op}")
        return result

    # Use infixNotation (also known as operatorPrecedence) to declare
    # operator precedence and associativity.  Operators are listed from
    # highest precedence to lowest.  For each operator level we provide
    # the parse action that combines the operands.
    cond_expr = infixNotation(
        operand,
        [
            (oneOf("! not"), 1, opAssoc.RIGHT, unary_eval),
            (oneOf("* / %"), 2, opAssoc.LEFT, binary_eval),
            (oneOf("+ -"), 2, opAssoc.LEFT, binary_eval),
            (oneOf("< <= > >="), 2, opAssoc.LEFT, binary_eval),
            (oneOf("== != === !=="), 2, opAssoc.LEFT, binary_eval),
            ((Literal("and") | Literal("&&")), 2, opAssoc.LEFT, binary_eval),
            ((Literal("or") | Literal("||")), 2, opAssoc.LEFT, binary_eval),
        ],
    )

    # Ternary expression: cond_expr ? expr : expr
    ternary = cond_expr + Suppress("?") + expr + Suppress(":") + expr

    def ternary_action(tokens):
        # ``tokens`` is a list: [cond_val, true_val, false_val]
        cond_val, true_val, false_val = tokens
        return true_val if cond_val else false_val

    ternary.setParseAction(ternary_action)

    # The full expression can be a ternary or a conditional expression.
    expr <<= ternary | cond_expr
    return expr


def eval_cost(expression: str, **variables: Any) -> Any:
    """Evaluate a JavaScript‑style expression in Python.

    The expression may contain nested ternary operators (``a ? b : c``),
    comparison operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``),
    logical operators (``&&``, ``||``, ``!``/``not``), arithmetic
    operations (``+``, ``-``, ``*``, ``/``, ``%``) and parentheses for
    grouping.  JavaScript boolean literals (``true``/``false``), the
    ``null`` literal (interpreted as Python ``None``) and string
    literals (single or double quoted) are recognised.  Variable names
    consisting of letters, digits and underscores are looked up in the
    keyword arguments provided to this function.

    Examples
    --------
    >>> expr = '(output == "video" ? (((quality == "pro" ? 40 : 20) + (sound_effects ? 5 : 0)) * duration) : (5 * n_samples))'
    >>> eval_cost(expr, duration=10, quality="pro", sound_effects=None, n_samples=5, output="video")
    400

    The same expression evaluated with a different ``output`` value:

    >>> eval_cost(expr, duration=10, quality="pro", sound_effects=True, n_samples=5, output="audio")
    25

    Parameters
    ----------
    expression : str
        A string containing the expression written using JavaScript
        syntax.  The expression must be a valid expression (no
        statements) and may include nested ternaries.
    **variables : Any
        Keyword arguments mapping variable names used in ``expression`` to
        their Python values.

    Returns
    -------
    Any
        The value of the evaluated expression.  Numeric results that
        happen to be integers will be returned as Python ``int`` values;
        other numeric results are returned as ``float``.  Strings,
        booleans and ``None`` values are returned unchanged.
    """
    # Build a parser configured with the provided variables.  This
    # parser encapsulates the evaluation logic via parse actions.
    parser = _build_expression_parser(variables)
    try:
        result = parser.parseString(expression, parseAll=True)[0]
    except Exception as exc:
        # Re‑raise with additional context for easier debugging.
        raise ValueError(
            f"Failed to evaluate expression '{expression}': {exc}"
        ) from exc
    # Coerce floats that are mathematically integers back to int for
    # convenience.  Many arithmetic operations produce floats via the
    # numeric grammar; this step normalises results such as 25.0 to 25.
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
