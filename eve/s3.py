import os
import io
import boto3
import hashlib
import mimetypes
import magic
import requests
import tempfile
import replicate
from pydub import AudioSegment
from typing import Iterator
from PIL import Image


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
)


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


def get_root_url():
    """Returns the root URL for the specified bucket."""
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
    url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION_NAME}.amazonaws.com"
    return url


def get_full_url(filename):
    return f"{get_root_url()}/{filename}"


def upload_file_from_url(url, name=None, file_type=None):
    """Uploads a file to an S3 bucket by downloading it to a temporary file and uploading it to S3."""

    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    if f"{AWS_BUCKET_NAME}.s3." in url and ".amazonaws.com" in url:
        # file is already uploaded
        filename = url.split("/")[-1].split(".")[0]
        return url, filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile() as tmp_file:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_file.seek(0)
            return upload_file(tmp_file.name, name, file_type)


def upload_file(file, name=None, file_type=None):
    """Uploads a file to an S3 bucket and returns the file URL."""

    if isinstance(file, replicate.helpers.FileOutput):
        file = file.read()
        file_bytes = io.BytesIO(file)
        return upload_buffer(file_bytes, name, file_type)
    
    if file.endswith(".safetensors"):
        file_type = ".safetensors"


    if file.startswith("http://") or file.startswith("https://"):
        return upload_file_from_url(file, name, file_type)

    with open(file, "rb") as file:
        buffer = file.read()

    return upload_buffer(buffer, name, file_type)


def upload_buffer(buffer, name=None, file_type=None):
    """Uploads a buffer to an S3 bucket and returns the file URL."""

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

    # print(f"Uploading file to S3: {name}{file_type}")

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

    # Upload file to S3
    filename = f"{name}{file_type}"
    file_bytes = io.BytesIO(buffer)
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    file_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"

    # if file doesn't exist, upload it
    try:
        s3.head_object(Bucket=bucket_name, Key=filename)
        return file_url, name
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.upload_fileobj(
                file_bytes,
                bucket_name,
                filename,
                ExtraArgs={"ContentType": mime_type, "ContentDisposition": "inline"},
            )
        else:
            raise e

    return file_url, name


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


def copy_file_to_bucket(source_bucket, dest_bucket, source_key, dest_key=None):
    """
    S3 server-side copy of a file from one bucket to another.
    """
    if dest_key is None:
        dest_key = source_key

    copy_source = {"Bucket": source_bucket, "Key": source_key}

    file_url = f"https://{dest_bucket}.s3.amazonaws.com/{dest_key}"

    try:
        s3.head_object(Bucket=dest_bucket, Key=dest_key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
        else:
            raise e

    return file_url
