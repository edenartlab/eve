import os
import io
import hashlib
import mimetypes
import magic
import requests
import tempfile
import replicate
from pydub import AudioSegment
from typing import Iterator
from PIL import Image
from typing import Union
from pathlib import Path

file_extensions = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".mp4",
    "audio/flac": ".flac",
    "audio/wav": ".wav",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/png": ".png",
    "video/mp4": ".mp4",
    "application/x-tar": ".tar",
    "application/zip": ".zip",
}

def get_root_url(filename):
    """Returns the root URL"""
    base_folder = os.getenv("LOCAL_STORAGE", "storage")
    mime_type, _ = mimetypes.guess_type(filename) or "application/octet-stream"
    subfolder = subfolder = get_folder_by_mime_type(mime_type)
    return os.path.join(base_folder, subfolder)


def get_full_url(filename):
    return f"{get_root_url(filename)}/{filename}"


def upload_file_from_url(url, name=None, file_type=None):
    """Saves a file into local storage by downloading it to a temporary file and saving it into local."""

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile() as tmp_file:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_file.seek(0)
            return upload_file(tmp_file.name, name, file_type)


def upload_file(file, name=None, file_type=None):
    """Saves a file into local storage and returns the file URL."""

    if isinstance(file, replicate.helpers.FileOutput):
        file = file.read()
        file_bytes = io.BytesIO(file)
        return upload_buffer(file_bytes, name, file_type)
    
    elif isinstance(file, str):
        if file.endswith(".safetensors"):
            file_type = ".safetensors"

        if file.startswith("http://") or file.startswith("https://"):
            return upload_file_from_url(file, name, file_type)

        with open(file, "rb") as file:
            buffer = file.read()

    return upload_buffer(buffer, name, file_type)


def upload_buffer(buffer, name=None, file_type=None):
    """Saves a buffer into local storage and returns the file URL."""

    assert (
        file_type
        in [
            None,
            ".jpg",
            ".webp",
            ".png",
            ".mp3",
            ".mp4",
            ".flac",
            ".wav",
            ".tar",
            ".zip",
            ".safetensors",
        ]
    ), "file_type must be one of ['.jpg', '.webp', '.png', '.mp3', '.mp4', '.flac', '.wav', '.tar', '.zip', '.safetensors']"

    if isinstance(buffer, Iterator):
        buffer = b"".join(buffer)

    # Get file extension from mimetype
    mime_type = magic.from_buffer(buffer, mime=True)
    originial_file_type = (
        file_extensions.get(mime_type)
        or mimetypes.guess_extension(mime_type)
        or f".{mime_type.split('/')[-1]}"
    )
    if not file_type:
        file_type = originial_file_type

    # if it's an image of the wrong type, convert it
    if file_type != originial_file_type and mime_type.startswith("image/"):
        image = Image.open(io.BytesIO(buffer))
        output = io.BytesIO()
        if file_type == ".jpg":
            image.save(output, "JPEG", quality=95)
            mime_type = "image/jpeg"
        elif file_type == ".webp":
            image.save(output, "WEBP", quality=95)
            mime_type = "image/webp"
        elif file_type == ".png":
            image.save(output, "PNG", quality=95)
            mime_type = "image/png"
        buffer = output.getvalue()

    # if no name is provided, use sha256 of content
    if not name:
        hasher = hashlib.sha256()
        hasher.update(buffer)
        name = hasher.hexdigest()

    # Save file to local storage
    filename = f"{name}{file_type}"
    file_bytes = io.BytesIO(buffer)

    return save_file(file_bytes, filename, mime_type)


def get_folder_by_mime_type(mime_type: str) -> str:
    mime_type = mime_type.lower()

    if mime_type.startswith('video/'):
        return 'videos'
    
    elif mime_type.startswith('image/'):
        return 'images'
    
    elif mime_type.startswith('audio/'):
        return 'audios'
    
    else:
        return 'others'


def save_file(buffer: Union[bytes, io.BytesIO], file_name: str, mime_type: str):
    base_folder = os.getenv("LOCAL_STORAGE", "storage")
    subfolder = get_folder_by_mime_type(mime_type)
    storage_path = os.path.join(base_folder, subfolder)

    Path(storage_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(storage_path, file_name)

    if os.path.exists(file_path):
        return file_path, file_name
    
    try:
        if isinstance(buffer, io.BytesIO):
            buffer = buffer.getvalue()
        with open(file_path, 'wb') as f:
            f.write(buffer)
    except Exception as e:
        raise Exception(f"Failed to save file: {str(e)}")

    return file_path, file_name
    

def upload_PIL_image(image: Image.Image, name=None, file_type=None):
    format = file_type.split(".")[-1] or "webp"
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return upload_buffer(buffer, name, file_type)


def upload_audio_segment(audio: AudioSegment):
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3")
    output = upload_buffer(buffer)
    return output


def upload(data: any, name=None, file_type=None):
    if isinstance(data, Image.Image):
        return upload_PIL_image(data, name, file_type)
    elif isinstance(data, AudioSegment):
        return upload_audio_segment(data)
    elif isinstance(data, bytes):
        return upload_buffer(data, name, file_type)
    else:
        return upload_file(data, name, file_type)
