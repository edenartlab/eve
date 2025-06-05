"""
Modal deployment for MCP Fetch Server.

This module provides a Modal app that serves the MCP fetch tool over HTTP,
allowing external services to fetch web content via the MCP protocol.
"""

import os
from typing import Annotated, Tuple
from urllib.parse import urlparse, urlunparse

from modal import App, asgi_app, Image
from pydantic import BaseModel, Field

# Create Modal image with all required dependencies
image = Image.debian_slim().pip_install([
    "httpx<0.28",
    "markdownify>=0.13.1", 
    "mcp>=1.1.3",
    "protego>=0.3.1",
    "pydantic>=2.0.0",
    "readabilipy>=0.2.0",
    "requests>=2.32.3",
])

# Create Modal app
app = App("mcp-fetch", image=image)

# Configuration
DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)"

CUSTOM_USER_AGENT = os.getenv("FETCH_USER_AGENT")
IGNORE_ROBOTS_TXT = os.getenv("IGNORE_ROBOTS_TXT", "false").lower() == "true"
PROXY_URL = os.getenv("PROXY_URL")

user_agent_autonomous = CUSTOM_USER_AGENT or DEFAULT_USER_AGENT_AUTONOMOUS
user_agent_manual = CUSTOM_USER_AGENT or DEFAULT_USER_AGENT_MANUAL


class FetchParams(BaseModel):
    """Parameters for the fetch tool."""
    
    url: Annotated[str, Field(description="URL to fetch")]
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ]
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="Start index for content extraction.",
            ge=0,
        ),
    ]
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get raw HTML content without simplification.",
        ),
    ]


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format."""
    import readabilipy.simple_json
    import markdownify
    
    ret = readabilipy.simple_json.simple_json_from_html_string(
        html, use_readability=True
    )
    if not ret["content"]:
        return "<error>Page failed to be simplified from HTML</error>"
    content = markdownify.markdownify(
        ret["content"],
        heading_style=markdownify.ATX,
    )
    return content


def get_robots_txt_url(url: str) -> str:
    """Get the robots.txt URL for a given website URL."""
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    return robots_url


async def check_may_autonomously_fetch_url(url: str, user_agent: str, proxy_url: str | None = None) -> None:
    """Check if the URL can be fetched according to robots.txt."""
    import httpx
    from protego import Protego
    
    robot_txt_url = get_robots_txt_url(url)
    
    async with httpx.AsyncClient(proxies=proxy_url) as client:
        try:
            response = await client.get(
                robot_txt_url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except httpx.HTTPError:
            raise Exception(f"Failed to fetch robots.txt {robot_txt_url} due to a connection issue")
        
        if response.status_code in (401, 403):
            raise Exception(f"Autonomous fetching not allowed according to robots.txt")
        elif 400 <= response.status_code < 500:
            return
            
        robot_txt = response.text
    
    processed_robot_txt = "\n".join(
        line for line in robot_txt.splitlines() if not line.strip().startswith("#")
    )
    robot_parser = Protego.parse(processed_robot_txt)
    if not robot_parser.can_fetch(str(url), user_agent):
        raise Exception(f"Autonomous fetching not allowed according to robots.txt")


async def fetch_url(
    url: str, user_agent: str, force_raw: bool = False, proxy_url: str | None = None
) -> Tuple[str, str]:
    """Fetch the URL and return the content."""
    import httpx
    
    async with httpx.AsyncClient(proxies=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
        except httpx.HTTPError as e:
            raise Exception(f"Failed to fetch {url}: {e!r}")
        
        if response.status_code >= 400:
            raise Exception(f"Failed to fetch {url} - status code {response.status_code}")

        page_raw = response.text

    content_type = response.headers.get("content-type", "")
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html and not force_raw:
        return extract_content_from_html(page_raw), ""

    return (
        page_raw,
        f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
    )


async def fetch_tool_impl(
    url: str,
    max_length: int = 5000,
    start_index: int = 0,
    raw: bool = False
) -> str:
    """Implementation of the fetch tool."""
    try:
        # Validate parameters
        params = FetchParams(
            url=url,
            max_length=max_length,
            start_index=start_index,
            raw=raw
        )
        
        # Check robots.txt if not ignored
        if not IGNORE_ROBOTS_TXT:
            await check_may_autonomously_fetch_url(
                params.url, 
                user_agent_autonomous, 
                PROXY_URL
            )
        
        # Fetch the content
        content, prefix = await fetch_url(
            params.url,
            user_agent_autonomous,
            force_raw=params.raw,
            proxy_url=PROXY_URL
        )
        
        # Handle pagination and truncation
        original_length = len(content)
        if params.start_index >= original_length:
            return "<error>No more content available.</error>"
        
        truncated_content = content[params.start_index:params.start_index + params.max_length]
        if not truncated_content:
            return "<error>No more content available.</error>"
        
        # Add continuation prompt if content was truncated
        actual_content_length = len(truncated_content)
        remaining_content = original_length - (params.start_index + actual_content_length)
        
        if actual_content_length == params.max_length and remaining_content > 0:
            next_start = params.start_index + actual_content_length
            truncated_content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
        
        return f"{prefix}Contents of {params.url}:\n{truncated_content}"
        
    except Exception as e:
        return f"<error>Failed to fetch {url}: {str(e)}</error>"


@app.function()
@asgi_app()
def mcp_server_app():
    """ASGI app serving the MCP server over HTTP."""
    from mcp.server.fastmcp import FastMCP
    
    # Create FastMCP server
    mcp_server = FastMCP(
        name="Fetch",
        stateless_http=True
    )
    
    @mcp_server.tool(
        name="fetch",
        description="""Fetches a URL from the internet and optionally extracts its contents as markdown.

This tool provides internet access and can fetch the most up-to-date information. 
It respects robots.txt by default and can extract content from HTML pages."""
    )
    async def fetch_tool(
        url: str,
        max_length: int = 5000,
        start_index: int = 0,
        raw: bool = False
    ) -> str:
        return await fetch_tool_impl(url, max_length, start_index, raw)
    
    @mcp_server.prompt(
        name="fetch",
        description="Fetch a URL and extract its contents as markdown"
    )
    def fetch_prompt(url: str) -> str:
        return f"Please fetch the content from this URL: {url}"
    
    return mcp_server.streamable_http_app()


if __name__ == "__main__":
    print("MCP Fetch Server ready for deployment")
    print("Deploy with: modal deploy modal_server.py")
    print("Serve locally with: modal serve modal_server.py")