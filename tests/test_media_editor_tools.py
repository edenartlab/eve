"""media_editor and reel hand their work to a sub-session; the tools they
name must exist, and the editing tools must work on the files they are given.
No model is called here."""

import asyncio
import os
import re
import shutil
import subprocess

import pytest
import yaml

from eve.tool import ToolContext
from eve.tools.media_utils.media_editor import handler as media_editor
from eve.tools.media_utils.reel import handler as reel
from eve.tools.tool_handlers import HANDLER_PATHS, load_handler

TOOLS_DIR = os.path.join(os.path.dirname(media_editor.__file__), "..")
EDITING_TOOLS = ["video_concat", "audio_video_combine", "ffmpeg_multitool", "audio_mix"]


def _capture_session_post(monkeypatch, module):
    posted = {}

    async def fake_session_post(context):
        posted["args"] = context.args
        return {"output": [{"session": "child"}]}

    monkeypatch.setattr(module, "session_post_handler", fake_session_post)
    return posted


def test_media_editor_offers_exactly_the_tools_it_describes(monkeypatch):
    posted = _capture_session_post(monkeypatch, media_editor)
    context = ToolContext(
        args={
            "instructions": "Join the two clips and add the music",
            "media_files": ["https://example.com/a.mp4", "https://example.com/m.mp3"],
        },
        user="u",
        agent="a",
        session="s",
    )
    result = asyncio.run(media_editor.handler(context))
    assert result == {"output": [{"session": "child"}]}

    args = posted["args"]
    assert args["extra_tools"] == EDITING_TOOLS
    assert args["attachments"] == context.args["media_files"]
    assert "Join the two clips and add the music" in args["content"]
    # Every tool named in the prompt is offered, and the other way round.
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", args["content"]))
    assert named - {"media_editor"} == set(EDITING_TOOLS)


@pytest.mark.parametrize("tool", EDITING_TOOLS + ["media_editor", "reel"])
def test_tools_used_for_editing_exist_and_are_active(tool):
    assert tool in HANDLER_PATHS
    assert callable(load_handler(tool))
    with open(os.path.join(TOOLS_DIR, tool, "api.yaml")) as f:
        schema = yaml.safe_load(f)
    assert schema["active"] is True and schema["handler"] == "modal"


def test_reel_delegates_assembly_to_media_editor(monkeypatch):
    posted = _capture_session_post(monkeypatch, reel)
    context = ToolContext(
        args={"context": "c", "producer_instructions": "a 30s trailer"},
        user="u",
        agent="a",
        session="s",
    )
    asyncio.run(reel.handler(context))
    assert posted["args"]["extra_tools"] == ["media_editor"]
    assert "media_editor" in posted["args"]["content"]


needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)


def _make(path, *ffmpeg_args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ffmpeg_args, str(path)], check=True)
    return str(path)


def _streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return sorted(out.stdout.split())


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return float(out.stdout.strip())


@pytest.fixture
def clips(tmp_path):
    # pytest's tmp_path is inside the temp directory, the only place local
    # input files are accepted from.
    video = ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=24", "-t", "1"]
    tone = ["-f", "lavfi", "-i", "sine=frequency=440", "-t", "1"]
    x264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    return {
        "sound": _make(tmp_path / "sound.mp4", *video[:4], *tone, *x264, "-c:a", "aac"),
        "silent": _make(tmp_path / "silent.mp4", *video, *x264),
        "music": _make(tmp_path / "music.mp3", *tone),
        "voice": _make(
            tmp_path / "voice.mp3", "-f", "lavfi", "-i", "sine=frequency=880", "-t", "2"
        ),
    }


def _run(tool, args):
    return asyncio.run(load_handler(tool)(ToolContext(args=args)))["output"]


@needs_ffmpeg
def test_video_concat_keeps_sound_when_only_some_clips_have_it(clips):
    output = _run("video_concat", {"videos": [clips["silent"], clips["sound"]]})
    assert _streams(output) == ["audio", "video"]
    assert _duration(output) == pytest.approx(2.0, abs=0.3)


@needs_ffmpeg
def test_video_concat_of_silent_clips_stays_silent(clips):
    # The audio added afterwards must not be mixed with a silent track.
    output = _run("video_concat", {"videos": [clips["silent"], clips["silent"]]})
    assert _streams(output) == ["video"]
    assert _duration(output) == pytest.approx(2.0, abs=0.3)


@needs_ffmpeg
def test_audio_tools_work_on_files_from_the_temp_directory(clips):
    combined = _run(
        "audio_video_combine", {"video": clips["silent"], "audio": [clips["music"]]}
    )
    assert _streams(combined) == ["audio", "video"]

    mixed = _run(
        "audio_mix",
        {
            "audio_tracks": [
                {"audio_url": clips["music"], "volume": 0.4},
                {"audio_url": clips["voice"], "volume": 1.0},
            ]
        },
    )
    assert _duration(mixed) == pytest.approx(2.0, abs=0.3)

    padded = _run("audio_pad", {"audio_url": clips["music"], "pad_end": 1})
    assert _duration(padded) == pytest.approx(2.0, abs=0.3)

    joined = _run("audio_concat", {"audio_files": [clips["music"], clips["voice"]]})
    assert _duration(joined) == pytest.approx(3.0, abs=0.3)


@pytest.mark.parametrize(
    "tool,args",
    [
        ("video_concat", {"videos": [__file__, __file__]}),
        ("audio_video_combine", {"video": __file__, "audio": []}),
        ("audio_mix", {"audio_tracks": [{"audio_url": __file__}]}),
        ("audio_pad", {"audio_url": __file__, "pad_end": 1}),
        ("audio_concat", {"audio_files": [__file__, __file__]}),
    ],
)
def test_editing_tools_refuse_local_paths_outside_temp(tool, args):
    with pytest.raises(ValueError, match="local file paths are not accepted"):
        _run(tool, args)
