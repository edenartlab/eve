"""The ffmpeg tool runs an allow-listed binary from an argument list, never a
shell, only inside an isolated sandbox, and only on files in its working
directory."""

import asyncio

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

    monkeypatch.setattr(modal, "is_local", lambda: False)
    monkeypatch.setattr(modal.Sandbox, "create", _Aio(fake_create))

    async def go():
        box = ffmpeg_sandbox.FFmpegSandbox(command_timeout=60)
        await box.start(["tmp_ab12cd34/video1.mp4"])
        await box.close()

    asyncio.run(go())

    kwargs = created["kwargs"]
    assert created["args"] == ()
    assert kwargs["block_network"] is True
    assert kwargs["secrets"] == []
    assert kwargs["timeout"] == 60 + 600
    for forbidden in ("volumes", "env", "network_file_systems", "proxy"):
        assert forbidden not in kwargs
    assert sb.filesystem.copied == [
        ("tmp_ab12cd34/video1.mp4", "/work/tmp_ab12cd34/video1.mp4")
    ]
    assert created["terminated"] is True
