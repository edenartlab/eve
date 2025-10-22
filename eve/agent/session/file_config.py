"""
Configuration constants and helper utilities for file attachment handling.
"""

import os
import re
from urllib.parse import urlparse, unquote


# File attachment configuration
TEXT_ATTACHMENT_MAX_LENGTH = 20000  # Maximum character limit for text attachments
CSV_DIALECT_SAMPLE_SIZE = 4096  # Number of bytes to sample for CSV dialect detection
FILE_CACHE_DIR = "/tmp/eden_file_cache/"  # Temporary directory for cached files

# Image processing configuration
IMAGE_MAX_SIZE = 512  # Maximum dimension (width/height) for image resizing
IMAGE_QUALITY = 90  # JPEG quality (0-100)

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".webm")
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
SUPPORTED_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".plain")
SUPPORTED_PDF_EXTENSION = (".pdf",)
SUPPORTED_CSV_EXTENSION = (".csv",)

# All non-media file extensions (text, PDF, CSV) - used for Discord attachment URL handling
SUPPORTED_NON_MEDIA_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_PDF_EXTENSION + SUPPORTED_CSV_EXTENSION


# Unsupported file types with conversion instructions
UNSUPPORTED_FILE_FORMATS = {
    "xlsx": "Excel file - Please convert to CSV format and re-upload",
    "xls": "Excel file - Please convert to CSV format and re-upload",
    "xlsm": "Excel macro file - Please convert to CSV format and re-upload",
    "ods": "OpenDocument Spreadsheet - Please convert to CSV format and re-upload",
    "doc": "Word document - Please save as PDF or plain text and re-upload",
    "docx": "Word document - Please save as PDF or plain text and re-upload",
    "ppt": "PowerPoint presentation - Please convert to PDF and re-upload",
    "pptx": "PowerPoint presentation - Please convert to PDF and re-upload",
    "zip": "Compressed archive - Please extract and upload individual files",
    "rar": "Compressed archive - Please extract and upload individual files",
    "7z": "Compressed archive - Please extract and upload individual files",
}


def _extract_extension_from_url(url: str) -> str:
    """
    Extract the file extension from a URL, handling query parameters and other URL components.

    This is more robust than simple .endswith() checks because URLs often contain query parameters
    (e.g., Discord CDN URLs like https://cdn.discordapp.com/.../file.pdf?ex=123&is=456&hm=789)

    Args:
        url: The URL to extract extension from

    Returns:
        Lowercase file extension including the dot (e.g., '.pdf'), or empty string if no extension found
    """
    if not url:
        return ""

    try:
        # Parse the URL to get the path component (excludes query parameters and fragments)
        parsed = urlparse(url)
        path = unquote(parsed.path)  # Decode URL encoding

        # Extract filename from path
        filename = os.path.basename(path)

        # Get extension
        _, ext = os.path.splitext(filename)
        return ext.lower()
    except Exception:
        # Fallback: try to extract extension from the URL string directly
        # This handles cases where URL parsing might fail
        # Match pattern: dot followed by 2-5 alphanumeric characters before query/fragment/end
        match = re.search(r'\.([a-zA-Z0-9]{2,5})(?:[?#]|$)', url)
        if match:
            return f".{match.group(1).lower()}"
        return ""


def _url_has_extension(url: str, extensions: tuple) -> bool:
    """
    Check if a URL has one of the specified file extensions.
    Handles query parameters and other URL components robustly.

    Args:
        url: The URL to check
        extensions: Tuple of extensions to check for (e.g., ('.pdf', '.txt'))

    Returns:
        True if the URL has one of the specified extensions, False otherwise
    """
    ext = _extract_extension_from_url(url)
    return ext in extensions
