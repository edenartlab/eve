import asyncio
import os
import subprocess
import tempfile

from eve.tool import ToolContext

# from ... import utils


SILENT_AUDIO = "anullsrc=channel_layout=stereo:sample_rate=44100"


async def _has_audio(video: str) -> bool:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video,
    ]
    probe_result = await asyncio.to_thread(
        subprocess.run, probe_command, capture_output=True, text=True
    )
    return probe_result.stdout.strip() != ""


async def handler(context: ToolContext):
    from .... import utils

    video_urls = context.args.get("videos")
    fps = context.args.get("fps", 30)

    # Inputs are URLs, downloaded to temp files (local paths are only accepted
    # from the temp directory).
    video_files = [
        await asyncio.to_thread(utils.get_file_handler, ".mp4", video_url)
        for video_url in video_urls
    ]

    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # The concat step needs the same streams in every part. If only some of
    # the videos have sound, the silent ones get a silent track; if none do,
    # the result stays silent.
    audio_flags = [await _has_audio(video) for video in video_files]
    any_audio = any(audio_flags)

    converted_videos = []
    for video, has_audio in zip(video_files, audio_flags):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            output_video = temp.name

            convert_command = ["ffmpeg", "-y", "-loglevel", "info", "-i", video]
            if any_audio and not has_audio:
                convert_command.extend(["-f", "lavfi", "-i", SILENT_AUDIO, "-shortest"])
            convert_command.extend(
                ["-r", str(fps), "-c:v", "libx264", "-crf", "19", "-preset", "fast"]
            )

            if any_audio:
                # One layout for every part, so the parts can be joined.
                convert_command.extend(
                    ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
                )
            else:
                convert_command.append("-an")

            convert_command.append(output_video)

            # Run conversion and log outputs
            result = await asyncio.to_thread(
                subprocess.run, convert_command, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise Exception(f"Error in converting video {video}: {result.stderr}")

            # Check if output video exists and has content
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                converted_videos.append(output_video)
            else:
                raise Exception(
                    f"Converted video {output_video} is empty or was not created."
                )

    if not converted_videos:
        raise Exception("No videos to concatenate.")

    # Create a text file listing all converted videos for concatenation
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as concat_list_file:
        for video in converted_videos:
            concat_list_file.write(f"file '{video}'\n")
        concat_list_filename = concat_list_file.name

    # Concatenate all videos using the list file
    concat_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "info",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_filename,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        *(["-c:a", "aac", "-b:a", "128k"] if any_audio else ["-an"]),
        "-movflags",
        "+faststart",
        output_file.name,
    ]

    result = await asyncio.to_thread(
        subprocess.run, concat_command, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Error in concatenating videos: {result.stderr}")

    # Check if the output video file is created and has content
    if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
        pass
    else:
        raise Exception("Final output video is empty or was not created.")

    # Clean up intermediate video files and concatenation list
    for video in converted_videos:
        os.remove(video)
    os.remove(concat_list_filename)

    if (
        not os.path.exists(output_file.name)
        or not os.path.getsize(output_file.name) > 0
    ):
        raise Exception("Final output video is empty. Upload skipped.")

    return {"output": output_file.name}
