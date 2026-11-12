"""The websearch tool fetches a page without a browser, refuses non-public
addresses on every hop, and turns HTML into text and links."""

import asyncio

import httpx
import pytest

from eve.tool import ToolContext
from eve.tools.tool_handlers import load_handler
from eve.tools.websearch import handler as websearch
from eve.tools.websearch.handler import (
    PageFetchError,
    check_public_host,
    extract_page,
    fetch_page,
    format_page,
)

PUBLIC = "http://93.184.216.34"  # a literal public address: no DNS lookup

PAGE = """<!doctype html>
<html><head><title>  Test &amp; Page </title>
<style>body { color: red }</style><script>var hidden = "nope";</script></head>
<body class="has-header">
<nav><a href="/home">Home page</a><a href="/home">Home page</a></nav>
<div id="cookie-banner">We use cookies <a href="/c">Accept all</a></div>
<h1>Main heading</h1>
<p>First <b>bold</b> paragraph.</p>
<p>Second paragraph with a <a href="https://other.example/x">useful link</a>.</p>
<ul><li>One</li><li>Two</li></ul>
<span>new</span><span>past</span>
<div hidden>invisible</div>
<footer>Footer text <a href="about">About us</a></footer>
<a href="mailto:x@example.com">Mail me</a><a href="/cdn-cgi/x">Protected</a>
</body></html>"""


def test_handler_is_registered_and_needs_no_browser():
    assert load_handler("websearch") is websearch.handler
    assert "playwright" not in open(websearch.__file__).read().split('"""', 2)[2]


def test_extracts_title_text_and_links():
    title, text, links = extract_page(PAGE, "https://site.example/dir/page")
    assert title == "Test & Page"
    assert "## Main heading" in text
    assert "First bold paragraph." in text
    assert "One\nTwo" in text
    assert "new past" in text
    for hidden in ("nope", "color: red", "Home page", "cookies", "invisible", "Footer"):
        assert hidden not in text
    hrefs = [link["href"] for link in links]
    assert hrefs.count("https://site.example/home") == 2
    assert "https://other.example/x" in hrefs
    assert "https://site.example/dir/about" in hrefs
    # Cookie buttons, non-http links and challenge pages are left out.
    assert not any("/c" == h[-2:] or "mailto" in h or "cdn-cgi" in h for h in hrefs)


def test_output_lists_most_linked_pages_first():
    title, text, links = extract_page(PAGE, "https://site.example/dir/page")
    output = format_page("https://site.example/dir/page", title, text, links, 2, 30)
    assert "## Title\nTest & Page" in output
    assert "..." in output  # text cut at max_chars
    link_lines = [line for line in output.splitlines() if line.startswith("- [")]
    assert link_lines[0] == "- [Home page](https://site.example/home) (appears 2 times)"
    assert len(link_lines) == 2


def test_unclosed_and_malformed_markup_does_not_break_extraction():
    title, text, links = extract_page(
        "<p>one<p>two<li>three</div></span><h2>Head<a href='/x'>link text", "http://h/"
    )
    assert "one" in text and "two" in text and "three" in text and "## Head" in text
    assert links == [{"text": "link text", "href": "http://h/x"}]


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "10.0.0.5",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "[::1]",
        "::ffff:127.0.0.1",
        "fd00::1",
        "",
        None,
    ],
)
def test_non_public_hosts_are_refused(host):
    with pytest.raises(PageFetchError):
        asyncio.run(check_public_host(host))


def test_public_address_is_accepted():
    asyncio.run(check_public_host("93.184.216.34"))


def _transport(routes):
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["user-agent"]))
        return routes(request)

    return httpx.MockTransport(respond), seen


def test_fetch_follows_public_redirects_and_decodes_the_body():
    def routes(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=iso-8859-1"},
            content="<p>caf\xe9</p>".encode("iso-8859-1"),
        )

    transport, seen = _transport(routes)
    url, body = asyncio.run(fetch_page(f"{PUBLIC}/start", transport))
    assert url == f"{PUBLIC}/final"
    assert body == "<p>café</p>"
    assert len(seen) == 2


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.1.2.3/",
        "file:///etc/passwd",
        "ftp://93.184.216.34/x",
    ],
)
def test_redirect_to_a_non_public_target_is_refused(location):
    transport, seen = _transport(
        lambda request: httpx.Response(302, headers={"location": location})
    )
    with pytest.raises(PageFetchError):
        asyncio.run(fetch_page(f"{PUBLIC}/start", transport))
    # Only the first, public hop was requested.
    assert [url for url, _ in seen] == [f"{PUBLIC}/start"]


def test_redirect_loops_end():
    transport, seen = _transport(
        lambda request: httpx.Response(302, headers={"location": "/again"})
    )
    with pytest.raises(PageFetchError, match="too many redirects"):
        asyncio.run(fetch_page(f"{PUBLIC}/start", transport))
    assert len(seen) == websearch.MAX_REDIRECTS + 1


def test_refused_client_is_retried_once_with_a_browser_user_agent():
    def routes(request):
        if "EdenBot" in request.headers["user-agent"]:
            return httpx.Response(403)
        return httpx.Response(200, headers={"content-type": "text/html"}, text="ok")

    transport, seen = _transport(routes)
    url, body = asyncio.run(fetch_page(f"{PUBLIC}/", transport))
    assert body == "ok"
    assert len(seen) == 2 and "Chrome" in seen[1][1]


def test_non_text_and_error_responses_are_reported():
    transport, _ = _transport(
        lambda request: httpx.Response(
            200, headers={"content-type": "video/mp4"}, content=b"\x00"
        )
    )
    with pytest.raises(PageFetchError, match="not a text page"):
        asyncio.run(fetch_page(f"{PUBLIC}/v.mp4", transport))

    transport, _ = _transport(lambda request: httpx.Response(404))
    with pytest.raises(PageFetchError, match="404"):
        asyncio.run(fetch_page(f"{PUBLIC}/missing", transport))


def test_large_bodies_are_cut_off(monkeypatch):
    monkeypatch.setattr(websearch, "MAX_BODY_BYTES", 1000)

    async def chunks():
        for _ in range(100):
            yield b"a" * 500

    transport, _ = _transport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=chunks()
        )
    )
    _, body = asyncio.run(fetch_page(f"{PUBLIC}/big", transport))
    assert 1000 <= len(body) < 50000


def test_handler_returns_page_or_error_text(monkeypatch):
    async def fake_fetch(url, transport=None):
        if "bad" in url:
            raise PageFetchError("only public web addresses can be fetched")
        return url, PAGE

    monkeypatch.setattr(websearch, "fetch_page", fake_fetch)

    ok = asyncio.run(
        websearch.handler(ToolContext(args={"url": "https://site.example/"}))
    )
    assert ok["output"].startswith("# Page Analysis: https://site.example/")
    assert "## Main heading" in ok["output"]

    bad = asyncio.run(websearch.handler(ToolContext(args={"url": "http://bad/"})))
    assert bad["output"] == (
        "Error processing webpage: only public web addresses can be fetched"
    )
