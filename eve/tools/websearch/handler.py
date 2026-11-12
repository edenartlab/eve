"""Fetch a web page and return its readable text and most-linked URLs.

This runs without a browser: the page is fetched over HTTP and parsed with
the standard library, so the tool needs nothing beyond the API image. Pages
that render their content only with JavaScript return little text.
"""

import asyncio
import ipaddress
import re
import socket
from collections import Counter
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

import httpx

from eve.tool import ToolContext

# Sites differ in what they refuse: some reject browser-like agents that are
# not browsers, others reject anything that is not browser-like. Identify
# honestly first and fall back once.
USER_AGENTS = (
    "Mozilla/5.0 (compatible; EdenBot/1.0; +https://eden.art)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
)
REFUSED_STATUS_CODES = (401, 403, 406, 429)
REQUEST_TIMEOUT_SECONDS = 15
MAX_REDIRECTS = 5
MAX_BODY_BYTES = 5 * 1024 * 1024
TEXT_CONTENT_TYPES = (
    "text/",
    "application/xhtml",
    "application/xml",
    "application/json",
)

# Elements whose content is never page text.
_NON_CONTENT_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
# Navigation and page furniture, left out of the text (links are still kept).
_BOILERPLATE_TAGS = {"nav", "header", "footer"}
_BOILERPLATE_TERMS = (
    "menu",
    "nav",
    "header",
    "footer",
    "sidebar",
    "cookie",
    "popup",
    "modal",
)
# A class such as "has-header" on a page-level container must not hide the page.
_NEVER_BOILERPLATE = {"html", "body", "main", "article"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "section",
    "article",
    "blockquote",
    "pre",
    "dd",
    "dt",
    "figcaption",
}
_IGNORED_LINK_TEXT = re.compile(r"^(Accept|Cookie|Got it|×|Close)", re.IGNORECASE)


class PageFetchError(Exception):
    """The page could not be fetched, or may not be fetched."""


async def check_public_host(host: Optional[str]) -> None:
    """Refuse hosts that resolve to anything but a public address.

    The URL comes from a user or a model and is fetched from a container that
    holds service credentials, so loopback, private, link-local and metadata
    addresses are never requested.
    """
    if not host:
        raise PageFetchError("URL has no host")
    try:
        addresses = [ipaddress.ip_address(host.strip("[]"))]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, type=socket.SOCK_STREAM
            )
        except socket.gaierror:
            raise PageFetchError(f"could not resolve host {host}")
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    if not addresses or not all(address.is_global for address in addresses):
        raise PageFetchError("only public web addresses can be fetched")


class PageRefusedError(PageFetchError):
    """The site refused this client; another user agent may be accepted."""


async def fetch_page(
    url: str, transport: Optional[httpx.AsyncBaseTransport] = None
) -> Tuple[str, str]:
    """Return (final URL, decoded body). Redirects are checked hop by hop."""
    for user_agent in USER_AGENTS[:-1]:
        try:
            return await _fetch_page_as(url, user_agent, transport)
        except PageRefusedError:
            continue
    return await _fetch_page_as(url, USER_AGENTS[-1], transport)


async def _fetch_page_as(
    url: str, user_agent: str, transport: Optional[httpx.AsyncBaseTransport]
) -> Tuple[str, str]:
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": user_agent, "Accept-Language": "en"},
        transport=transport,
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https"):
                raise PageFetchError("URL must start with http:// or https://")
            await check_public_host(parts.hostname)

            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise PageFetchError("redirect without a location")
                    url = urljoin(url, location)
                    continue
                if response.status_code in REFUSED_STATUS_CODES:
                    raise PageRefusedError(f"HTTP status {response.status_code}")
                if response.status_code >= 400:
                    raise PageFetchError(f"HTTP status {response.status_code}")
                content_type = response.headers.get("content-type", "text/html")
                if not content_type.lower().startswith(TEXT_CONTENT_TYPES):
                    raise PageFetchError(
                        f"not a text page (content type {content_type})"
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) >= MAX_BODY_BYTES:
                        break
                encoding = response.charset_encoding or "utf-8"
                try:
                    return url, bytes(body).decode(encoding, errors="replace")
                except LookupError:
                    return url, bytes(body).decode("utf-8", errors="replace")
        raise PageFetchError("too many redirects")


class _PageParser(HTMLParser):
    """Collect the title, the readable text and the links of a page."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []
        self.links: List[dict] = []
        # Open elements as (tag, hides_text, never_content).
        self._stack: List[Tuple[str, bool, bool]] = []
        self._in_title = False
        self._link_href: Optional[str] = None
        self._link_text: List[str] = []

    def _hidden(self) -> bool:
        return any(hides for _, hides, _ in self._stack)

    def _never_content(self) -> bool:
        return any(never for _, _, never in self._stack)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "title" and not self.title_parts:
            self._in_title = True
        if tag == "a" and attributes.get("href"):
            self._close_link()
            self._link_href = urljoin(self.base_url, attributes["href"])
        if tag in _VOID_TAGS:
            if tag == "br" and not self._hidden():
                self.text_parts.append("\n")
            return

        never_content = tag in _NON_CONTENT_TAGS
        marker = f"{attributes.get('class') or ''} {attributes.get('id') or ''}".lower()
        boilerplate = tag in _BOILERPLATE_TAGS or (
            tag not in _NEVER_BOILERPLATE
            and any(term in marker for term in _BOILERPLATE_TERMS)
        )
        hides = (
            never_content
            or boilerplate
            or "hidden" in attributes
            or attributes.get("aria-hidden") == "true"
        )
        self._stack.append((tag, hides, never_content))
        if tag in _HEADING_TAGS and not self._hidden():
            self.text_parts.append("\n\n## ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "a":
            self._close_link()
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                hidden = self._hidden()
                del self._stack[index:]
                if not hidden and tag in _HEADING_TAGS | {"p"}:
                    self.text_parts.append("\n\n")
                elif not hidden and tag in _BLOCK_TAGS:
                    self.text_parts.append("\n")
                break

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._never_content():
            return
        if self._link_href is not None:
            self._link_text.append(data)
        if not self._hidden() and data.strip():
            # Adjacent inline elements ("<a>new</a><a>past</a>") are separate words.
            self.text_parts.append(data + " ")

    def _close_link(self):
        if self._link_href is None:
            return
        text = " ".join("".join(self._link_text).split())
        href = self._link_href
        self._link_href, self._link_text = None, []
        if (
            len(text) > 1
            and href.startswith("http")
            and "/cdn-cgi/" not in href
            and not _IGNORED_LINK_TEXT.match(text)
        ):
            self.links.append({"text": text, "href": href})

    def close(self):
        super().close()
        self._close_link()


def extract_page(html: str, base_url: str) -> Tuple[str, str, List[dict]]:
    """Return (title, readable text, links) for an HTML document."""
    parser = _PageParser(base_url)
    parser.feed(html)
    parser.close()

    title = " ".join("".join(parser.title_parts).split())
    text = "".join(parser.text_parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, parser.links


def format_page(
    url: str, title: str, text: str, links: List[dict], max_links: int, max_chars: int
) -> str:
    output = f"""# Page Analysis: {url}

## Title
{title}

## Content Summary
{text[:max_chars]}{"..." if len(text) > max_chars else ""}

## Top Links
"""
    # Most-linked first, longer link text first among ties.
    link_counts = Counter(link["href"] for link in links)
    sorted_links = sorted(
        links, key=lambda link: (-link_counts[link["href"]], -len(link["text"]))
    )
    seen_links = set()
    for link in sorted_links:
        if len(seen_links) >= max_links:
            break
        if link["href"] in seen_links:
            continue
        seen_links.add(link["href"])
        count = link_counts[link["href"]]
        output += f"- [{link['text']}]({link['href']}) (appears {count} times)\n"
    return output


async def handler(context: ToolContext):
    url = context.args["url"]
    max_links = int(context.args.get("max_links", 5))
    max_chars = int(context.args.get("max_chars", 20000))

    try:
        final_url, body = await fetch_page(url)
        title, text, links = extract_page(body, final_url)
    except Exception as e:
        return {"output": f"Error processing webpage: {str(e)}"}

    return {"output": format_page(url, title, text, links, max_links, max_chars)}
