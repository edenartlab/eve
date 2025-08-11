from __future__ import annotations
import os
import re
import pathlib
import tempfile
import requests
import httpx
import boto3
from io import BytesIO
from tqdm import tqdm
from safetensors.torch import load_file, save_file

from .constants import USE_MEDIA_CACHE


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