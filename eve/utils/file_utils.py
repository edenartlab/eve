import io
import os
import pathlib
import re
import subprocess
import tempfile
from io import BytesIO

import boto3
import httpx
import requests
from loguru import logger
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from .constants import USE_MEDIA_CACHE


def get_filename_from_url(url: str) -> str:
    """
    Extract a clean filename from a URL, stripping query parameters.

    Args:
        url: URL to extract filename from

    Returns:
        str: Filename without query parameters
    """
    filename = pathlib.Path(url).name
    # Remove query parameters and URL fragments
    filename = re.sub(r"[?#].*$", "", filename)
    return filename


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
            logger.info(
                f"<**> Found {cache_filename} in volume cache, copying to {local_filepath}"
            )
            # Copy from cache to target location
            import shutil

            shutil.copy2(str(cache_filepath), str(local_filepath))
            return str(local_filepath)

    except Exception as e:
        # If volume access fails, silently continue with normal download
        logger.error(f"Volume cache check failed: {e}")

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
            logger.info(f"<**> Saving {cache_filename} to volume cache")
            import shutil

            shutil.copy2(str(local_filepath), str(cache_filepath))

            # Commit volume changes if we have access to modal
            try:
                import modal

                logger.debug(f"Modal module available: {modal}")
                # Try to get the volume and commit changes
                # This will only work if we're in a Modal function with volume access
                # We can't directly access the volume object, so we rely on Modal's auto-commit
            except Exception:
                pass

    except Exception as e:
        # If volume access fails, silently continue
        logger.error(f"Volume cache save failed: {e}")


def url_exists(url: str, timeout: int = 5) -> bool:
    """Check if a URL exists by making a HEAD request."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False


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
        return str(local_filepath)

    # Check for Modal Volume cache
    if USE_MEDIA_CACHE:
        cache_path = _check_volume_cache(url, local_filepath, overwrite)
        if cache_path:
            return cache_path

    try:
        # Parse S3 URL to extract bucket and key
        s3_pattern = r"https://([^.]+)\.s3(?:\.([^.]+))?\.amazonaws\.com/(.+)"
        s3_match = re.match(s3_pattern, url)

        if s3_match:
            # This is an S3 URL
            bucket_name = s3_match.group(1)
            region = s3_match.group(2) or os.getenv("AWS_REGION_NAME", "us-east-1")
            key = s3_match.group(3)

            # Use boto3 to download with credentials
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=region,
            )

            try:
                s3_client.download_file(bucket_name, key, str(local_filepath))
                if USE_MEDIA_CACHE:
                    # Save to volume cache after successful download
                    _save_to_volume_cache(url, local_filepath)
                return str(local_filepath)
            except Exception as s3_error:
                logger.error(f"S3 download error: {s3_error}")
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


def convert_pti_to_safetensors(input_path: str, output_path: str):
    try:
        data = load_file(input_path)
    except Exception as e:
        logger.error(f"❌ Failed to load {input_path} with safetensors: {e}")
        return False

    # Expected mapping from .pti → .safetensors SDXL format
    key_map = {
        "text_encoders_0": "clip_l",  # 768-dim
        "text_encoders_1": "clip_g",  # 1280-dim
    }

    remapped = {}
    for k, v in data.items():
        if k not in key_map:
            logger.warning(
                f"⚠️ Unexpected key '{k}' in {input_path}. Expected only {list(key_map.keys())}. Skipping this key."
            )
            continue

        new_key = key_map[k]
        remapped[new_key] = v

    if not remapped:
        logger.error(
            f"❌ No valid keys found for conversion in {input_path}. Output file not saved."
        )
        return False

    try:
        save_file(remapped, output_path)
        logger.info(f"✅ Converted {input_path} → {output_path}")
        for k, v in remapped.items():
            logger.info(f"  {k}: shape {v.shape}, dtype {v.dtype}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save converted file to {output_path}: {e}")
        return False


def validate_image_bytes(image_bytes: bytes) -> tuple[bool, dict]:
    """
    Validate that bytes represent a valid image using Pillow.

    Returns:
        tuple: (ok: bool, info: dict)
            - If valid: (True, {"format": str, "size": (width, height)})
            - If invalid: (False, {"reason": str})
    """
    try:
        # Load and verify image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Check consistency

        # Reopen to get details (verify() invalidates the image)
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Fully decode to catch any issues

        width, height = img.size
        if width <= 0 or height <= 0:
            return False, {"reason": f"Invalid dimensions: {width}x{height}"}

        if not img.format:
            return False, {"reason": "Unable to determine image format"}

        return True, {"format": img.format, "size": (width, height)}
    except UnidentifiedImageError:
        return False, {"reason": "Unrecognized image format"}
    except Exception as e:
        return False, {"reason": f"Image validation error: {e.__class__.__name__}"}


def validate_video_bytes(video_bytes: bytes) -> tuple[bool, dict]:
    """
    Validate that bytes represent a valid video file using ffprobe.

    Returns:
        tuple: (ok: bool, info: dict)
            - If valid: (True, {"format": str, "duration": float, "size": (width, height)})
            - If invalid: (False, {"reason": str})
    """
    try:
        # Write bytes to temporary file for ffprobe to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # Use ffprobe to validate and get info
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,duration",
                    "-show_entries",
                    "format=format_name,duration",
                    "-of",
                    "json",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return False, {"reason": f"ffprobe failed: {result.stderr.strip()}"}

            import json

            probe_data = json.loads(result.stdout)

            # Extract video stream info
            if "streams" not in probe_data or not probe_data["streams"]:
                return False, {"reason": "No video stream found"}

            stream = probe_data["streams"][0]
            format_info = probe_data.get("format", {})

            codec = stream.get("codec_name", "unknown")
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            duration = float(stream.get("duration") or format_info.get("duration") or 0)
            format_name = format_info.get("format_name", "unknown")

            if width <= 0 or height <= 0:
                return False, {"reason": f"Invalid video dimensions: {width}x{height}"}

            return True, {
                "format": codec,
                "container": format_name,
                "duration": duration,
                "size": (width, height),
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        return False, {"reason": "Video validation timeout"}
    except FileNotFoundError:
        return False, {"reason": "ffprobe not found (install ffmpeg)"}
    except Exception as e:
        return False, {"reason": f"Video validation error: {e.__class__.__name__}"}


def validate_audio_bytes(audio_bytes: bytes) -> tuple[bool, dict]:
    """
    Validate that bytes represent a valid audio file using ffprobe.

    Returns:
        tuple: (ok: bool, info: dict)
            - If valid: (True, {"format": str, "duration": float, "sample_rate": int})
            - If invalid: (False, {"reason": str})
    """
    try:
        # Write bytes to temporary file for ffprobe to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Use ffprobe to validate and get info
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate,duration,channels",
                    "-show_entries",
                    "format=format_name,duration",
                    "-of",
                    "json",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return False, {"reason": f"ffprobe failed: {result.stderr.strip()}"}

            import json

            probe_data = json.loads(result.stdout)

            # Extract audio stream info
            if "streams" not in probe_data or not probe_data["streams"]:
                return False, {"reason": "No audio stream found"}

            stream = probe_data["streams"][0]
            format_info = probe_data.get("format", {})

            codec = stream.get("codec_name", "unknown")
            sample_rate = int(stream.get("sample_rate", 0))
            channels = int(stream.get("channels", 0))
            duration = float(stream.get("duration") or format_info.get("duration") or 0)
            format_name = format_info.get("format_name", "unknown")

            if sample_rate <= 0:
                return False, {"reason": f"Invalid sample rate: {sample_rate}"}

            return True, {
                "format": codec,
                "container": format_name,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        return False, {"reason": "Audio validation timeout"}
    except FileNotFoundError:
        return False, {"reason": "ffprobe not found (install ffmpeg)"}
    except Exception as e:
        return False, {"reason": f"Audio validation error: {e.__class__.__name__}"}


def is_valid_image_url(url: str, *, timeout=(5, 15), max_bytes=8 * 1024 * 1024):
    """
    Validate an image URL by downloading and verifying it.

    Returns:
        tuple: (ok: bool, info: dict)
            - Checks HTTP status
            - Ensures Content-Type starts with image/
            - Downloads with a size cap
            - Validates image with Pillow
    """
    try:
        # Quick HEAD check (best effort)
        try:
            h = requests.head(url, timeout=timeout, allow_redirects=True)
            if h.status_code >= 400:
                return False, {"reason": f"HTTP {h.status_code} on HEAD"}
            ct = (h.headers.get("Content-Type") or "").lower().split(";")[0].strip()
            if ct and not ct.startswith("image/"):
                return False, {"reason": f"Not an image Content-Type: {ct}"}
            cl = h.headers.get("Content-Length")
            if cl and int(cl) > max_bytes:
                return False, {"reason": "Too large by Content-Length"}
        except requests.RequestException:
            # If HEAD fails, we'll still try GET
            pass

        # Streamed GET with byte cap
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        buf = io.BytesIO()
        total = 0
        for chunk in r.iter_content(8192):
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return False, {"reason": "Downloaded bytes exceed limit"}
            buf.write(chunk)
        data = buf.getvalue()

        # Validate the downloaded bytes
        return validate_image_bytes(data)
    except requests.RequestException as e:
        return False, {"reason": f"Network error: {e.__class__.__name__}"}
    except Exception as e:
        return False, {"reason": f"Other error: {e.__class__.__name__}"}
