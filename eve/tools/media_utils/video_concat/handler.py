from eve.tool import ToolContext
import os
import tempfile
import subprocess
# from ... import utils


# bug: if some videos are silent but others have sound, the concatenated video will have no sound


async def handler(context: ToolContext):
    from .... import utils

    video_urls = context.args.get("videos")
    fps = context.args.get("fps", 30)

    video_files = [
        utils.download_file(video_url, video_url.split("/")[-1])
        for video_url in video_urls
    ]

    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    converted_videos = []
    for video in video_files:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            output_video = temp.name

            # Check if the video has audio
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
            probe_result = subprocess.run(probe_command, capture_output=True, text=True)
            has_audio = probe_result.stdout.strip() != ""

            # Prepare ffmpeg command
            convert_command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "info",
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
            ]

            if has_audio:
                convert_command.extend(["-c:a", "aac", "-b:a", "128k"])
            else:
                # Add a silent audio track if the video has no audio
                convert_command.extend(
                    [
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-af",
                        "anullsrc=channel_layout=stereo:sample_rate=44100",
                    ]
                )

            convert_command.append(output_video)

            # Run conversion and log outputs
            result = subprocess.run(convert_command, capture_output=True, text=True)
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
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_file.name,
    ]

    result = subprocess.run(concat_command, capture_output=True, text=True)
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
