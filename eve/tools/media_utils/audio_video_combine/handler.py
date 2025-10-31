from eve.tool import ToolContext
import tempfile
import subprocess


async def handler(context: ToolContext):
    from .... import utils

    video_url = context.args.get("video")
    audio_url = context.args.get("audio")

    video_file = utils.get_file_handler(".mp4", video_url)
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # Get video duration - this will be the output duration
    video_duration = utils.get_media_duration(video_file)

    if audio_url:
        audio_file = utils.get_file_handler(".mp3", audio_url)

        # Check if video has existing audio
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_has_audio = result.stdout.strip() == "audio"

        if video_has_audio:
            # Mix the new audio with existing video audio
            # Both audio tracks will be trimmed/padded to video duration
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "panic",
                "-i",
                video_file,
                "-i",
                audio_file,
                "-filter_complex",
                f"[1:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,apad[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-t", str(video_duration),
                "-movflags",
                "+faststart",
                output_file.name,
            ]
        else:
            # Video has no audio, just add the new audio (trimmed/padded to video duration)
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "panic",
                "-i",
                video_file,
                "-i",
                audio_file,
                "-filter_complex",
                f"[1:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,apad[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-t", str(video_duration),
                "-movflags",
                "+faststart",
                output_file.name,
            ]

    else:
        # if no audio, create a silent audio track with same duration as video
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
            "-movflags",
            "+faststart",
            output_file.name,
        ]

    subprocess.run(cmd)

    return {"output": output_file.name}
