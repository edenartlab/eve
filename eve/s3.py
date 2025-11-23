import datetime
import gzip
import hashlib
import io
import json
import mimetypes
import os
import tarfile
import tempfile
from typing import Iterator

import boto3
import magic
import replicate
import requests
from PIL import Image
from pydub import AudioSegment

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


def get_root_url(s3=False):
    """Returns the root URL, CloudFront by default, or S3"""
    if s3:
        db = os.getenv("DB", "STAGE").upper()
        AWS_BUCKET_NAME = os.getenv(f"AWS_BUCKET_NAME_{db}")
        AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
        url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION_NAME}.amazonaws.com"
        return url
    else:
        return os.getenv("CLOUDFRONT_URL")


def get_full_url(filename):
    return f"{get_root_url()}/{filename}"


def upload_file_from_url(url, name=None, file_type=None):
    """Uploads a file to an S3 bucket by downloading it to a temporary file and uploading it to S3."""

    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    if f"{AWS_BUCKET_NAME}.s3." in url and ".amazonaws.com" in url:
        # file is already uploaded
        filename = url.split("/")[-1].split(".")[0]
        return url, filename

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    with requests.get(url, stream=True, headers=headers) as r:
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

    elif isinstance(file, str):
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


########################################################
########################################################


# https://chatgpt.com/c/68dbf846-933c-832a-91be-0dcb6be3647b


# big_export.py


try:
    import zstandard as zstd  # optional, for .zst

    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False


class MultipartUploadWriter:
    """File-like object: writes bytes into S3 Multipart Upload."""

    def __init__(
        self,
        bucket,
        key,
        part_size=128 * 1024 * 1024,  # 128MB parts (~400 parts for 50GB)
        content_type="application/x-tar",
        content_disposition=None,
        kms_key_id=None,
        storage_class="STANDARD",
        tags=None,
        metadata=None,
    ):
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        self.part_size = max(part_size, 5 * 1024 * 1024)  # S3 minimum is 5MB
        self.buf = bytearray()
        self.part_no = 1
        self.parts = []
        self.closed = False
        self.sha256 = hashlib.sha256()

        create_args = {
            "Bucket": bucket,
            "Key": key,
            "ContentType": content_type,
            "StorageClass": storage_class,
        }
        if content_disposition:
            create_args["ContentDisposition"] = content_disposition
        if kms_key_id:
            create_args["ServerSideEncryption"] = "aws:kms"
            create_args["SSEKMSKeyId"] = kms_key_id
        if tags:
            # "k1=v1&k2=v2" format
            create_args["Tagging"] = "&".join(f"{k}={v}" for k, v in tags.items())
        if metadata:
            create_args["Metadata"] = metadata

        resp = self.s3.create_multipart_upload(**create_args)
        self.upload_id = resp["UploadId"]

    def write(self, data: bytes):
        if self.closed:
            raise ValueError("Writer already closed")
        if not data:
            return 0
        self.sha256.update(data)
        self.buf.extend(data)
        if len(self.buf) >= self.part_size:
            self._flush_part()
        return len(data)

    def _flush_part(self):
        body = bytes(self.buf)
        self.buf.clear()
        resp = self.s3.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            PartNumber=self.part_no,
            UploadId=self.upload_id,
            Body=body,
        )
        self.parts.append({"ETag": resp["ETag"], "PartNumber": self.part_no})
        self.part_no += 1

    def close(self):
        if self.closed:
            return
        try:
            if self.buf:
                self._flush_part()
            self.s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self.upload_id,
                MultipartUpload={"Parts": self.parts},
            )
            self.closed = True
        except Exception:
            # Ensure we don't leave pending MPU junk in the bucket
            try:
                self.s3.abort_multipart_upload(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
                )
            except Exception:
                pass
            raise

    def digest_hex(self):
        return self.sha256.hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.s3.abort_multipart_upload(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
                )
            except Exception:
                pass


def _open_tar_sink(muw: MultipartUploadWriter, compression: str):
    """
    Returns (tarfile_obj, sink_to_close, content_ext, content_type)
    compression: "none" | "gz" | "zst"
    """
    if compression == "gz":
        gz = gzip.GzipFile(fileobj=muw, mode="wb", mtime=0)
        tf = tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT)
        return tf, gz, ".tar.gz", "application/gzip"
    elif compression == "zst":
        if not HAS_ZSTD:
            raise RuntimeError(
                "zstandard not installed; pip install zstandard or use 'gz'/'none'"
            )
        cctx = zstd.ZstdCompressor(level=10)
        zfh = cctx.stream_writer(muw)  # file-like
        tf = tarfile.open(fileobj=zfh, mode="w|", format=tarfile.PAX_FORMAT)
        return tf, zfh, ".tar.zst", "application/zstd"
    else:
        # plain tar (fastest for already-compressed media)
        tf = tarfile.open(fileobj=muw, mode="w|", format=tarfile.PAX_FORMAT)
        return tf, None, ".tar", "application/x-tar"


def stream_export_to_s3(
    *,
    user_id: str,
    items: list[dict],
    compression: str = "none",  # "none" | "gz" | "zst"
    exports_bucket: str | None = None,
    kms_key_id: str | None = None,
) -> dict:
    """
    items: list of {"s3_bucket": "...", "s3_key": "...", "arcname": "path/in/archive.ext"}
    Returns dict with {bucket, key, filename, sha256}
    """
    exports_bucket = (
        exports_bucket
        or os.getenv("AWS_EXPORTS_BUCKET_NAME")
        or os.getenv("AWS_BUCKET_NAME")
    )
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_filename = f"user-{user_id}-export-{ts}"
    s3_key_prefix = f"exports/{user_id}/{ts}/"
    # We’ll decide extension/content-type after we pick compression

    # open sink (figures out extension + content-type)
    # Create the writer AFTER knowing extension/content-type (we need to set ContentType/Disposition on MPU)
    # So, we first create a dummy writer, detect, then recreate? Instead: call _open_tar_sink after creating writer with chosen types.
    # We'll determine these before instantiating writer:
    ext_map = {
        "none": (".tar", "application/x-tar"),
        "gz": (".tar.gz", "application/gzip"),
        "zst": (".tar.zst", "application/zstd"),
    }
    content_ext, content_type = ext_map[compression]

    filename = base_filename + content_ext
    key = s3_key_prefix + filename
    content_disp = f'attachment; filename="{filename}"'

    with MultipartUploadWriter(
        exports_bucket,
        key,
        part_size=128 * 1024 * 1024,
        content_type=content_type,
        content_disposition=content_disp,
        kms_key_id=kms_key_id,
        tags={"export": "1", "user_id": user_id},
        metadata={"creator": "modal-export"},
    ) as muw:
        tf, secondary_sink, _, _ = _open_tar_sink(muw, compression)

        manifest = {"user_id": user_id, "created_at": ts, "count": 0, "items": []}

        for it in items:
            src_bucket = it["s3_bucket"]
            src_key = it["s3_key"]
            arcname = it["arcname"]

            head = s3.head_object(Bucket=src_bucket, Key=src_key)
            size = head["ContentLength"]
            mtime = int(head["LastModified"].timestamp())

            ti = tarfile.TarInfo(name=arcname)
            ti.size = size
            ti.mtime = mtime
            # Optional: ti.mode, ti.uid, ti.gid, ti.uname/gname

            # Stream the source object into the tar (no temp files)
            obj = s3.get_object(Bucket=src_bucket, Key=src_key)
            body = obj["Body"]  # StreamingBody, provides .read()
            tf.addfile(ti, fileobj=body)

            manifest["count"] += 1
            manifest["items"].append(
                {
                    "name": arcname,
                    "size": size,
                    "src": f"s3://{src_bucket}/{src_key}",
                    "last_modified": head["LastModified"].isoformat(),
                }
            )

        tf.close()
        if secondary_sink is not None:
            secondary_sink.close()

        sha256_hex = muw.digest_hex()

    # Upload checksum + manifest (small single PUTs)
    s3.put_object(
        Bucket=exports_bucket,
        Key=f"{key}.sha256",
        Body=f"{sha256_hex}  {filename}\n".encode("utf-8"),
        ContentType="text/plain",
        ServerSideEncryption="aws:kms" if kms_key_id else None,
        SSEKMSKeyId=kms_key_id if kms_key_id else None,
    )
    s3.put_object(
        Bucket=exports_bucket,
        Key=f"{key}.manifest.json",
        Body=json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="aws:kms" if kms_key_id else None,
        SSEKMSKeyId=kms_key_id if kms_key_id else None,
    )

    return {
        "bucket": exports_bucket,
        "key": key,
        "filename": filename,
        "sha256": sha256_hex,
        "manifest_key": f"{key}.manifest.json",
        "sha256_key": f"{key}.sha256",
    }
