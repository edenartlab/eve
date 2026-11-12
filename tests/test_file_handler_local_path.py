import os
import tempfile

import pytest

from eve.utils.file_utils import get_file_handler


def test_rejects_local_path_outside_temp():
    with pytest.raises(ValueError):
        get_file_handler(".mp4", "/proc/self/environ" if os.path.exists("/proc/self/environ") else "/etc/hosts")


def test_rejects_symlink_escaping_temp():
    with tempfile.TemporaryDirectory() as d:
        link = os.path.join(d, "x.mp4")
        os.symlink("/etc/hosts", link)
        with pytest.raises(ValueError):
            get_file_handler(".mp4", link)


def test_accepts_file_in_temp():
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        assert get_file_handler(".mp4", f.name) == f.name
