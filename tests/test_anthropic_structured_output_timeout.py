"""Regression guard for the 2026-09-18 structured-output outage.

anthropic 1.x moved its transport from `httpx` to `httpx2`. Passing an
`httpx.Timeout` to the new SDK raises a TypeError inside `_build_request`,
which the SDK's own retry loop then re-raises as `APIConnectionError`, whose
message is the entirely unhelpful "Connection error.".

Only the two Anthropic structured-output call sites passed a Timeout object,
so nothing else broke: regular streaming chat kept working while every
structured-output call failed. That is why `media_editor` and `reel` sub
sessions ran to completion and then dropped their results on the floor -- the
extraction call in `session_post` is a structured-output call.

A plain number is accepted by every SDK generation, so these tests pin that.
"""

import ast
import pathlib

from eve.agent.llm.providers.anthropic import STRUCTURED_OUTPUT_TIMEOUT_S

PROVIDER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "eve"
    / "agent"
    / "llm"
    / "providers"
    / "anthropic.py"
)


def test_timeout_is_a_plain_number():
    """Not an httpx.Timeout, not an httpx2.Timeout -- just seconds."""
    assert isinstance(STRUCTURED_OUTPUT_TIMEOUT_S, (int, float))
    assert not isinstance(STRUCTURED_OUTPUT_TIMEOUT_S, bool)
    assert STRUCTURED_OUTPUT_TIMEOUT_S > 0


def _timeout_arguments():
    """Every `timeout=` keyword passed to a call in the provider module."""
    tree = ast.parse(PROVIDER_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "timeout":
                    yield kw.value


def test_no_call_passes_a_transport_timeout_object():
    """`timeout=` must never be built from a transport library's Timeout class.

    Whichever library the SDK vendors this month, constructing its Timeout
    ourselves couples us to that choice and fails silently when it changes.
    """
    found = list(_timeout_arguments())
    assert found, "expected the provider to pass timeout= somewhere"

    for value in found:
        assert not isinstance(value, ast.Call), (
            f"line {value.lineno}: timeout= is constructed by calling "
            f"{ast.unparse(value)!r}. Pass a plain number of seconds instead -- "
            "anthropic 1.x rejects a foreign Timeout object and reports it as "
            '"Connection error.".'
        )
