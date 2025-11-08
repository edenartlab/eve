import os
from urllib.parse import urlparse
from typing import Optional, Tuple, Dict, Any, List


def _parse_int(value: Optional[str], default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_filename_from_url(url: str, fallback: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return fallback

    candidate = os.path.basename(parsed.path) if parsed.path else ""
    return candidate or fallback


def get_fake_media_defaults() -> Tuple[str, str, str, int, int]:
    url = os.getenv(
        "FAKE_TOOL_PLACEHOLDER_URL", "https://placehold.co/1024x1024/png?text=EVE"
    )
    filename = os.getenv("FAKE_TOOL_PLACEHOLDER_FILENAME") or _infer_filename_from_url(
        url, "fake-placeholder.png"
    )
    mime_type = os.getenv("FAKE_TOOL_PLACEHOLDER_MIMETYPE", "image/png")
    width = _parse_int(os.getenv("FAKE_TOOL_PLACEHOLDER_WIDTH"), 1024)
    height = _parse_int(os.getenv("FAKE_TOOL_PLACEHOLDER_HEIGHT"), 1024)

    return url, filename, mime_type, width, height


def build_fake_tool_result_payload(
    tool_key: str, tool_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    url, filename, mime_type, width, height = get_fake_media_defaults()
    display_name = tool_name or tool_key

    return [
        {
            "output": [
                {
                    "url": url,
                    "filename": filename,
                    "mediaAttributes": {
                        "mimeType": mime_type,
                        "width": width,
                        "height": height,
                        "placeholder": True,
                    },
                    "description": f"Simulated output for {display_name}",
                }
            ]
        }
    ]
