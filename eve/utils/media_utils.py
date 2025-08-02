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
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO

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