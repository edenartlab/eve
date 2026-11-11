"""The ffmpeg tool runs an allow-listed binary from an argument list, never a shell."""

import pytest

from eve.tools.media_utils.ffmpeg_multitool.handler import (
    FFmpegError,
    parse_ffmpeg_command,
)


def test_plain_ffmpeg_command_becomes_argv():
    argv = parse_ffmpeg_command('ffmpeg -y -i "in 1.mp4" -c:v libopenh264 out.mp4')
    assert argv == ["ffmpeg", "-y", "-i", "in 1.mp4", "-c:v", "libopenh264", "out.mp4"]


def test_shell_syntax_is_inert_tokens_not_commands():
    argv = parse_ffmpeg_command("ffmpeg -i a.mp4 out.mp4 ; cat /etc/passwd | sh")
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
