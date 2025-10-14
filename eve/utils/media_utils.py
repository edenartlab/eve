from __future__ import annotations
import os
import math
import magic
import base64
import tempfile
import textwrap
import subprocess
import blurhash
import numpy as np
import requests
from typing import List
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO
from loguru import logger

try:
    # MoviePy 2.x
    from moviepy import VideoFileClip, ImageClip, AudioClip
except ImportError:
    # MoviePy 1.x
    from moviepy.editor import VideoFileClip, ImageClip, AudioClip

import replicate

from .. import s3
from .file_utils import get_file_handler, download_file
from .text_utils import get_font, wrap_text


def get_media_attributes(file):
    if isinstance(file, replicate.helpers.FileOutput):
        is_url = False
        file_content = file.read()
        mime_type = magic.from_buffer(file_content, mime=True)
        file = BytesIO(file_content)
    else:
        is_url = file.startswith("http://") or file.startswith("https://")
        if is_url:
            from .file_utils import get_filename_from_url

            temp_file_path = "/tmp/eden_file_cache/" + get_filename_from_url(file)
            file = download_file(file, temp_file_path, overwrite=False)
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
            logger.error(f"Error encoding blurhash: {e}")

    return {"filename": filename, "mediaAttributes": media_attributes}


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


def center_crop_resize(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    """
    Resize and center crop an image to exact dimensions while preserving aspect ratio.
    """
    original_width, original_height = image.size
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        # Image is wider than target - crop width
        new_height = original_height
        new_width = int(new_height * target_ratio)
        left = (original_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = original_height
    else:
        # Image is taller than target - crop height
        new_width = original_width
        new_height = int(new_width / target_ratio)
        left = 0
        top = (original_height - new_height) // 2
        right = original_width
        bottom = top + new_height

    # Crop to the calculated dimensions
    cropped = image.crop((left, top, right, bottom))

    # Resize to exact target dimensions
    return cropped.resize((target_width, target_height), Image.LANCZOS)


def create_thumbnail(images: List[str]) -> str:
    """
    Create a 1024x1024px webp thumbnail from a list of image URLs.

    - 1 image: copy that image
    - 2 images: split (1024x512 + 1024x512 or 512x1024 + 512x1024)
    - 3 images: one split + subdivide one side
    - 4+ images: 2x2 grid (first 4 images only)
    """
    if not images:
        return None

    # Take first 4 images max
    images_to_use = images[:4]
    num_images = len(images_to_use)

    # Create 1024x1024 canvas
    canvas = Image.new("RGB", (1024, 1024), "white")

    if num_images == 1:
        # Single image - center crop and resize to 1024x1024
        img = download_image_to_PIL(images_to_use[0])
        canvas = center_crop_resize(img, 1024, 1024)

    elif num_images == 2:
        # Two images - determine layout based on aspect ratios
        img1 = download_image_to_PIL(images_to_use[0])
        img2 = download_image_to_PIL(images_to_use[1])

        # Calculate average aspect ratio
        avg_aspect = (img1.width / img1.height + img2.width / img2.height) / 2

        if avg_aspect > 1.0:  # Wider images - use horizontal split (two rows)
            img1_resized = center_crop_resize(img1, 1024, 512)
            img2_resized = center_crop_resize(img2, 1024, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (0, 512))
        else:  # Taller images - use vertical split (two columns)
            img1_resized = center_crop_resize(img1, 512, 1024)
            img2_resized = center_crop_resize(img2, 512, 1024)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (512, 0))

    elif num_images == 3:
        # Three images - one main + two subdivided
        img1 = download_image_to_PIL(images_to_use[0])
        img2 = download_image_to_PIL(images_to_use[1])
        img3 = download_image_to_PIL(images_to_use[2])

        # Calculate average aspect ratio
        avg_aspect = (
            img1.width / img1.height
            + img2.width / img2.height
            + img3.width / img3.height
        ) / 3

        if (
            avg_aspect > 1.0
        ):  # Wider images - horizontal main split, then vertical subdivision
            img1_resized = center_crop_resize(img1, 1024, 512)
            img2_resized = center_crop_resize(img2, 512, 512)
            img3_resized = center_crop_resize(img3, 512, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (0, 512))
            canvas.paste(img3_resized, (512, 512))
        else:  # Taller images - vertical main split, then horizontal subdivision
            img1_resized = center_crop_resize(img1, 512, 1024)
            img2_resized = center_crop_resize(img2, 512, 512)
            img3_resized = center_crop_resize(img3, 512, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (512, 0))
            canvas.paste(img3_resized, (512, 512))

    else:  # 4+ images - 2x2 grid
        positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
        for i in range(4):
            img = download_image_to_PIL(images_to_use[i])
            img_resized = center_crop_resize(img, 512, 512)
            canvas.paste(img_resized, positions[i])

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
    canvas.save(temp_file.name, "WEBP", quality=95)

    # Upload the thumbnail
    result = upload_media(temp_file.name, save_thumbnails=False)

    # Clean up temp file
    os.unlink(temp_file.name)

    return result["filename"]
