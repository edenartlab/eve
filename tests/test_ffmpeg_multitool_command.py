"""The ffmpeg tool runs an allow-listed binary from an argument list, never a
shell, only inside an isolated sandbox, and only on files in its working
directory."""

import asyncio
import os

import pytest

from eve.tools.media_utils.ffmpeg_multitool import handler as ffmpeg_handler
from eve.tools.media_utils.ffmpeg_multitool import sandbox as ffmpeg_sandbox
from eve.tools.media_utils.ffmpeg_multitool.handler import (
    FFmpegError,
    check_relative_path,
    execute_ffmpeg_command,
    parse_ffmpeg_command,
)


def test_plain_ffmpeg_command_becomes_argv():
    argv = parse_ffmpeg_command('ffmpeg -y -i "in 1.mp4" -c:v libopenh264 out.mp4')
    assert argv == ["ffmpeg", "-y", "-i", "in 1.mp4", "-c:v", "libopenh264", "out.mp4"]


def test_shell_syntax_is_inert_tokens_not_commands():
    argv = parse_ffmpeg_command("ffmpeg -i a.mp4 out.mp4 ; cat notes.txt | sh")
    # Everything stays an argument to ffmpeg; nothing is interpreted by a shell.
    assert argv[0] == "ffmpeg"
    assert ";" in argv and "|" in argv


@pytest.mark.parametrize(
    "command",
    [
        "",
        "bash -c 'ffmpeg -i a b'",
        "/usr/bin/ffmpeg -i a b",
        "./ffmpeg -i a b",
        "env ffmpeg -i a b",
        "sh -c ffmpeg",
        "curl http://example.com",
        "ffmpeg -i 'unterminated",
    ],
)
def test_other_programs_and_bad_syntax_are_refused(command):
    with pytest.raises(FFmpegError):
        parse_ffmpeg_command(command)


@pytest.mark.parametrize(
    "command",
    [
        # Typical generated commands must keep working.
        "ffmpeg -y -i tmp_ab12cd34/video1.mp4 -vf scale=iw/2:ih/2 -c:v libopenh264 out.mp4",
        "ffmpeg -y -i tmp_ab12cd34/video1.mp4 -ss 00:00:05 -t 10 -r 30000/1001 out.mp4",
        'ffmpeg -y -i a.mp4 -i b.mp3 -filter_complex "[0:v]pad=1280:720:(ow-iw)/2:(oh-ih)/2[v];[1:a]volume=0.5[a]" -map "[v]" -map "[a]" out.mp4',
        "ffmpeg -y -f concat -safe 0 -i tmp_ab12cd34/list.txt -c copy out.mp4",
        "ffmpeg -y -loop 1 -i tmp_ab12cd34/img1.png -f lavfi -i anullsrc=r=44100:cl=stereo -t 5 -aspect 16:9 out.mp4",
        "ffmpeg -y -i a.mp4 -vf \"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Hi':x=(w-text_w)/2:y=h-50\" out.mp4",
        "ffmpeg -y -i a.mp4 -filter_complex '[0:v][0:v]concat=n=2:v=1:a=0' out.mp4",
        # the concat filter with its defaults, followed by another filter
        "ffmpeg -y -i a.mp4 -i b.mp4 -filter_complex '[0:v][1:v]concat,format=yuv420p[v]' -map '[v]' out.mp4",
        "ffmpeg -y -i a.mp4 -vf \"drawtext=font='DejaVu Sans':text='Hello, World 50/50':x=(w-text_w)/2:y=h-40:box=1:boxcolor=black@0.5\" out.mp4",
        "ffmpeg -y -i a.mp4 -filter_complex '[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]' -map '[v]' -map '[a]' out.mp4",
        "ffmpeg -y -i a.mp4 -i b.mp4 -filter_complex '[0:v][1:v]xfade=transition=fade:duration=0.5:offset=1.5[v]' -map '[v]' out.mp4",
        "ffmpeg -y -loop 1 -i img1.png -vf \"zoompan=z='min(zoom+0.0015,1.5)':d=48:s=320x240:fps=24,fade=t=in:st=0:d=0.5\" -t 2 out.mp4",
        "ffmpeg -y -i a.mp4 -t 1 part1.mp4 -ss 1 part2.mp4",
        "ffprobe -v quiet -print_format json -show_streams tmp_ab12cd34/video1.mp4",
    ],
)
def test_normal_commands_are_accepted(command):
    assert parse_ffmpeg_command(command)[0] in {"ffmpeg", "ffprobe"}


@pytest.mark.parametrize(
    "command",
    [
        # absolute paths, as input, output, or inside a filtergraph
        "ffmpeg -i /etc/passwd out.mp4",
        "ffmpeg -i a.mp4 /tmp/out.mp4",
        "ffmpeg -f lavfi -i movie=/proc/self/environ out.mp4",
        "ffmpeg -i a.mp4 -vf drawtext=textfile=/proc/self/environ out.mp4",
        "ffmpeg -i a.mp4 -vf \"drawtext=textfile='/etc/hostname'\" out.mp4",
        "ffmpeg -i a.mp4 -vf drawtext=textfile=\\\\/etc\\\\/hostname out.mp4",
        "ffmpeg -i a.mp4 -vf subtitles=/etc/passwd out.mp4",
        "ffmpeg -i ~/secret.mp4 out.mp4",
        "ffmpeg -i a.mp4 -vf drawtext=fontfile=/usr/share/fonts/../../../etc/passwd out.mp4",
        # traversal
        "ffmpeg -i ../other/video.mp4 out.mp4",
        "ffmpeg -i tmp_ab12cd34/../../x.mp4 out.mp4",
        "ffmpeg -i a.mp4 ..",
        "ffmpeg -f lavfi -i movie=../x.mp4 out.mp4",
        # URLs and protocol prefixes
        "ffmpeg -i http://169.254.169.254/latest/meta-data out.mp4",
        "ffmpeg -i https://example.com/a.mp4 out.mp4",
        "ffmpeg -i a.mp4 -f mpegts tcp://example.com:9000",
        "ffmpeg -i file:a.mp4 out.mp4",
        "ffmpeg -i file:///etc/passwd out.mp4",
        "ffmpeg -i concat:a.ts|b.ts out.mp4",
        "ffmpeg -i a.mp4 -f mp4 pipe:1",
        "ffmpeg -i subfile,,start,0,end,100,,:a.mp4 out.mp4",
        "ffmpeg -i data:text/plain;base64,AAAA out.mp4",
        "ffmpeg -i a.mp4 -f tee out.mp4|unix:sock",
        # option values loaded from a file
        "ffmpeg -i a.mp4 -/filter_complex graph.txt out.mp4",
    ],
)
def test_arguments_outside_the_working_directory_are_refused(command):
    with pytest.raises(FFmpegError):
        parse_ffmpeg_command(command)


@pytest.mark.parametrize("path", ["out.mp4", "tmp_ab12cd34/out.mp4", "./out.mp4"])
def test_relative_output_paths_are_accepted(path):
    assert check_relative_path(path)


@pytest.mark.parametrize(
    "path",
    ["", "/tmp/out.mp4", "../out.mp4", "a/../../out.mp4", "~/out.mp4", "pipe:1"],
)
def test_output_paths_outside_the_working_directory_are_refused(path):
    with pytest.raises(FFmpegError):
        check_relative_path(path)


class _FakeSandbox:
    command_timeout = 60

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    async def run(self, argv):
        self.calls.append(argv)
        return self.returncode, "boom" if self.returncode else None


def test_execution_goes_to_the_sandbox_not_a_local_process(monkeypatch):
    def no_local_process(*args, **kwargs):
        raise AssertionError("ffmpeg must not run in the calling process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_local_process)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", no_local_process)

    fake = _FakeSandbox()
    asyncio.run(execute_ffmpeg_command(fake, "ffmpeg -y -i a.mp4 out.mp4"))
    assert fake.calls == [["ffmpeg", "-nostdin", "-y", "-i", "a.mp4", "out.mp4"]]


def test_refused_command_never_reaches_the_sandbox():
    fake = _FakeSandbox()
    with pytest.raises(FFmpegError):
        asyncio.run(execute_ffmpeg_command(fake, "ffmpeg -i /etc/passwd out.mp4"))
    assert fake.calls == []


def test_failed_command_reports_stderr():
    with pytest.raises(FFmpegError) as excinfo:
        asyncio.run(execute_ffmpeg_command(_FakeSandbox(1), "ffmpeg -i a.mp4 o.mp4"))
    assert excinfo.value.stderr == "boom"


def test_handler_has_no_local_execution_path():
    source = open(ffmpeg_handler.__file__).read()
    assert "create_subprocess_shell" not in source
    # The only local subprocess is the fixed-argument ffprobe used for metadata.
    assert source.count("create_subprocess_exec") == 1


def test_sandbox_is_created_without_secrets_volumes_or_network(monkeypatch):
    import modal

    created = {}

    class _Aio:
        def __init__(self, fn):
            self.aio = fn

    class _FS:
        def __init__(self):
            self.copied = []

            async def make_directory(path):
                created["workdir"] = path

            async def copy_from_local(local, remote):
                self.copied.append((local, remote))

            self.make_directory = _Aio(make_directory)
            self.copy_from_local = _Aio(copy_from_local)

    class _Sandbox:
        def __init__(self):
            self.filesystem = _FS()

            async def terminate():
                created["terminated"] = True

            self.terminate = _Aio(terminate)

    sb = _Sandbox()

    async def fake_create(*args, **kwargs):
        created["args"] = args
        created["kwargs"] = kwargs
        return sb

    async def fake_app():
        return "the-sandbox-app"

    monkeypatch.setattr(ffmpeg_sandbox, "_sandbox_app", fake_app)
    monkeypatch.setattr(modal.Sandbox, "create", _Aio(fake_create))

    async def go():
        box = ffmpeg_sandbox.FFmpegSandbox(command_timeout=60)
        await box.start(["tmp_ab12cd34/video1.mp4"])
        await box.close()

    asyncio.run(go())

    kwargs = created["kwargs"]
    assert created["args"] == ()
    assert kwargs["app"] == "the-sandbox-app"
    assert kwargs["block_network"] is True
    assert kwargs["secrets"] == []
    assert kwargs["timeout"] == 60 + 600
    for forbidden in ("volumes", "env", "network_file_systems", "proxy"):
        assert forbidden not in kwargs
    assert sb.filesystem.copied == [
        ("tmp_ab12cd34/video1.mp4", "/work/tmp_ab12cd34/video1.mp4")
    ]
    assert created["terminated"] is True


def test_sandbox_image_names_its_debian_release():
    # `debian_slim()` follows the workspace image-builder version; an
    # end-of-life release there cannot install ffmpeg any more.
    source = open(ffmpeg_sandbox.__file__).read()
    assert "Image.debian_slim(" not in source
    assert 'from_registry("debian:' in "".join(source.split())


def test_generator_is_told_what_the_sandbox_provides():
    # The sandbox's ffmpeg has libx264, not libopenh264.
    assert "libx264" in ffmpeg_handler.SYSTEM_CONTEXT
    assert "libopenh264" not in open(ffmpeg_handler.__file__).read()
    for phrase in ("without a shell", "concat filter", "relative", "drawtext"):
        assert phrase in ffmpeg_handler.EXECUTION_RULES


@pytest.mark.parametrize(
    "source,kind,expected",
    [
        ("https://cdn.example.com/a/clip.MP4", "video", "mp4"),
        (
            "https://bucket.s3.amazonaws.com/k.webm?X-Amz-Signature=abc&x=1.png",
            "video",
            "webm",
        ),
        ("https://example.com/download?id=5", "video", "mp4"),
        ("https://example.com/image", "img", "jpg"),
        ("https://example.com/track.mp3#t=5", "audio", "mp3"),
        ("https://example.com/x.tar.gz;rm", "audio", "mp3"),
    ],
)
def test_input_extension_comes_from_the_url_path(source, kind, expected):
    assert ffmpeg_handler._input_extension(source, kind) == expected


def test_inputs_are_downloaded_into_the_job_directory(monkeypatch, tmp_path):
    downloads = []

    def fake_download(url, path, overwrite=False):
        downloads.append((url, str(path), overwrite))
        with open(path, "wb") as f:
            f.write(b"x")
        return str(path)

    monkeypatch.setattr(ffmpeg_handler.utils, "download_file", fake_download)
    monkeypatch.chdir(tmp_path)

    media = ffmpeg_handler.validate_and_prepare_media(
        {
            "images": ["https://example.com/pic.png?sig=1", None],
            "videos": ["https://example.com/v"],
            "audios": ["https://example.com/a.wav"],
        },
        "tmp_job",
    )
    assert media.images == ["tmp_job/img1.png"]
    assert media.videos == ["tmp_job/video1.mp4"]
    assert media.audios == ["tmp_job/audio1.wav"]
    # Nothing is written outside the job directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tmp_job"]
    assert all(overwrite for _, _, overwrite in downloads)


def test_local_input_outside_temp_is_refused(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FFmpegError):
        ffmpeg_handler.validate_and_prepare_media(
            {"videos": [ffmpeg_handler.__file__]}, "tmp_job"
        )
    assert not (tmp_path / "tmp_job").exists()


class _ScriptedSandbox:
    """Stands in for FFmpegSandbox in handler tests: no Modal, no ffmpeg."""

    instances = []

    def __init__(self, command_timeout, attempts=1):
        self.command_timeout = command_timeout
        self.attempts = attempts
        self.inputs = None
        self.commands = []
        self.closed = False
        self.results = list(self.script)
        _ScriptedSandbox.instances.append(self)

    async def start(self, input_paths):
        self.inputs = list(input_paths)

    async def run(self, argv):
        self.commands.append(argv)
        return self.results.pop(0)

    async def fetch(self, rel_path, local_path):
        # `outputs` maps a path to the number of commands after which it exists.
        if len(self.commands) >= self.outputs.get(rel_path, float("inf")):
            if rel_path in self.too_large:
                raise ffmpeg_sandbox.OutputTooLargeError("Output file is too large")
            with open(local_path, "wb") as f:
                f.write(b"video")
            return True
        return False

    async def close(self):
        self.closed = True


def _run_handler(
    monkeypatch, tmp_path, script, outputs, responses, args=None, too_large=()
):
    from eve.tool import ToolContext

    _ScriptedSandbox.instances = []
    _ScriptedSandbox.script = script
    _ScriptedSandbox.outputs = outputs
    _ScriptedSandbox.too_large = too_large
    attempts = []

    async def fake_generate(task_instruction, media, previous_attempt=None):
        attempts.append(previous_attempt)
        command, output_path = responses[len(attempts) - 1]
        return ffmpeg_handler.FFmpegResponse(command=command, output_path=output_path)

    def fake_download(url, path, overwrite=False):
        with open(path, "wb") as f:
            f.write(b"x")
        return str(path)

    monkeypatch.setattr(ffmpeg_handler, "FFmpegSandbox", _ScriptedSandbox)
    monkeypatch.setattr(ffmpeg_handler, "generate_ffmpeg_command", fake_generate)
    monkeypatch.setattr(ffmpeg_handler.utils, "download_file", fake_download)
    monkeypatch.chdir(tmp_path)

    context = ToolContext(
        args={
            "task_instruction": "trim it",
            "videos": ["https://example.com/v.mp4"],
            **(args or {}),
        }
    )
    try:
        result = asyncio.run(ffmpeg_handler.handler(context))
        error = None
    except ValueError as e:
        result, error = None, e
    return result, error, attempts, _ScriptedSandbox.instances[0]


def test_handler_retries_in_one_sandbox_and_returns_the_output(monkeypatch, tmp_path):
    result, error, attempts, sandbox = _run_handler(
        monkeypatch,
        tmp_path,
        script=[(8, "Unknown encoder"), (0, None), (0, None)],
        outputs={"out.mp4": 3},
        responses=[
            ("ffmpeg -y -i a.mp4 -c:v nope out.mp4", "out.mp4"),
            ("ffmpeg -y -i a.mp4 other.mp4", "out.mp4"),  # wrong output name
            ("ffmpeg -y -i a.mp4 out.mp4", "out.mp4"),
        ],
    )
    assert error is None
    assert os.path.exists(result["output"]) and result["output"].endswith("_out.mp4")
    assert len(sandbox.commands) == 3 and sandbox.closed
    assert sandbox.inputs and sandbox.inputs[0].endswith("/video1.mp4")
    # Each failure is fed back to the generator.
    assert attempts[0] is None
    assert attempts[1]["stderr"] == "Unknown encoder"
    assert "does not exist or is empty" in attempts[2]["error"]
    # The job directory is removed; only the output remains.
    assert [p.name for p in tmp_path.iterdir()] == [result["output"]]


def test_handler_reports_time_limit_and_closes_the_sandbox(monkeypatch, tmp_path):
    result, error, attempts, sandbox = _run_handler(
        monkeypatch,
        tmp_path,
        script=[(ffmpeg_sandbox.TIMEOUT_EXIT_CODE, None)],
        outputs={},
        responses=[("ffmpeg -y -i a.mp4 out.mp4", "out.mp4")],
        args={"n_retries": 1, "timeout": 7},
    )
    assert result is None
    assert "time limit of 7 seconds" in str(error)
    assert sandbox.closed and list(tmp_path.iterdir()) == []


def test_handler_refuses_oversized_output(monkeypatch, tmp_path):
    result, error, attempts, sandbox = _run_handler(
        monkeypatch,
        tmp_path,
        script=[(0, None)],
        outputs={"out.mp4": 1},
        too_large={"out.mp4"},
        responses=[("ffmpeg -y -i a.mp4 out.mp4", "out.mp4")],
        args={"n_retries": 1},
    )
    assert result is None and "too large" in str(error)
    assert sandbox.closed


def test_handler_never_runs_a_refused_command(monkeypatch, tmp_path):
    result, error, attempts, sandbox = _run_handler(
        monkeypatch,
        tmp_path,
        script=[],
        outputs={},
        responses=[
            ("ffmpeg -y -i /proc/self/environ out.mp4", "out.mp4"),
            ("ffmpeg -y -i a.mp4 out.mp4", "/tmp/out.mp4"),
        ],
        args={"n_retries": 2},
    )
    assert result is None and "Security validation failed" in str(error)
    assert sandbox.commands == [] and sandbox.closed
