import subprocess
import tempfile

from eve.tool import ToolContext


async def handler(context: ToolContext):
    from .... import utils

    video_url = context.args.get("video")
    audio_urls = context.args.get("audio", [])

    video_file = utils.get_file_handler(".mp4", video_url)
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # Get video duration - this will be the output duration
    video_duration = utils.get_media_duration(video_file)

    # Download all audio files
    audio_files = [utils.get_file_handler(".mp3", url) for url in audio_urls]

    # Check if video has existing audio
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_file,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    video_has_audio = result.stdout.strip() == "audio"

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-i",
        video_file,
    ]

    # Add all audio file inputs
    for audio_file in audio_files:
        cmd.extend(["-i", audio_file])

    # Build filter complex for mixing all audio tracks
    # First, process each new audio track (trim/pad to video duration)
    filter_parts = []
    for i, _ in enumerate(audio_files, start=1):
        filter_parts.append(
            f"[{i}:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,apad[a{i}]"
        )

    # Then mix all tracks together
    if video_has_audio:
        # Include video's existing audio in the mix
        num_inputs = len(audio_files) + 1
        mix_inputs = "[0:a]" + "".join(
            f"[a{i}]" for i in range(1, len(audio_files) + 1)
        )
        filter_parts.append(
            f"{mix_inputs}amix=inputs={num_inputs}:duration=first:dropout_transition=0[aout]"
        )
    else:
        # Mix only the new audio tracks
        num_inputs = len(audio_files)
        if num_inputs == 1:
            # Single track, no mixing needed
            filter_parts.append("[a1]acopy[aout]")
        else:
            # Multiple tracks, mix them
            mix_inputs = "".join(f"[a{i}]" for i in range(1, len(audio_files) + 1))
            filter_parts.append(
                f"{mix_inputs}amix=inputs={num_inputs}:duration=first:dropout_transition=0[aout]"
            )

    filter_complex = ";".join(filter_parts)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-t",
            str(video_duration),
            "-movflags",
            "+faststart",
            output_file.name,
        ]
    )

    subprocess.run(cmd)

    return {"output": output_file.name}
