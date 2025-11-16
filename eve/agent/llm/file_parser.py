"""
File parsing utilities for processing various file types and injecting their content into chat messages.

This module handles parsing of:
- CSV files (with dialect detection and markdown table formatting)
- Text files (.txt, .md, .markdown, .plain)
- PDF files (text extraction)
- Images, videos, and audio (for attachment context)
"""

import csv
import os
from typing import List, Tuple

import magic
import pdfplumber
from loguru import logger

from eve.agent.llm.file_config import (
    CSV_DIALECT_SAMPLE_SIZE,
    FILE_CACHE_DIR,
    SUPPORTED_CSV_EXTENSION,
    SUPPORTED_PDF_EXTENSION,
    SUPPORTED_TEXT_EXTENSIONS,
    TEXT_ATTACHMENT_MAX_LENGTH,
    UNSUPPORTED_FILE_FORMATS,
    _url_has_extension,
)
from eve.utils import download_file


class ParsedAttachment:
    """Container for parsed attachment data"""

    def __init__(
        self,
        name: str,
        content: str,
        url: str,
        truncated: bool = False,
        is_text: bool = False,
        is_visual: bool = False,
        is_video: bool = False,
        is_audio: bool = False,
        error: str = None,
    ):
        self.name = name
        self.content = content
        self.url = url
        self.truncated = truncated
        self.is_text = is_text
        self.is_visual = is_visual
        self.is_video = is_video
        self.is_audio = is_audio
        self.error = error


def _parse_csv_file(attachment_file: str, attachment_url: str) -> ParsedAttachment:
    """
    Parse a CSV file and format it as a markdown table.

    Args:
        attachment_file: Local path to the downloaded CSV file
        attachment_url: Original URL of the attachment

    Returns:
        ParsedAttachment object with parsed CSV content or error
    """
    try:
        file_name = attachment_url.split("/")[-1]

        # Read the file once with appropriate encoding
        try:
            with open(attachment_file, "r", encoding="utf-8") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            with open(attachment_file, "r", encoding="latin-1") as f:
                file_content = f.read()

        # Check if file is empty or whitespace-only
        if not file_content or not file_content.strip():
            return ParsedAttachment(
                name=file_name,
                content="",
                url=attachment_url,
                error="CSV file is empty",
            )

        # Detect CSV dialect and parse
        try:
            sample = file_content[:CSV_DIALECT_SAMPLE_SIZE]
            dialect = csv.Sniffer().sniff(sample)
            csv_reader = csv.reader(file_content.splitlines(), dialect=dialect)
        except (csv.Error, Exception):
            # Fall back to default CSV reader if dialect detection fails
            csv_reader = csv.reader(file_content.splitlines())

        rows = list(csv_reader)

        # Check if CSV has meaningful content (non-empty cells)
        has_content = any(any(cell.strip() for cell in row) for row in rows)

        if not has_content:
            return ParsedAttachment(
                name=file_name,
                content="",
                url=attachment_url,
                error="CSV file is empty",
            )

        # Helper function to sanitize cell content for markdown tables
        def sanitize_cell(cell):
            if cell is None:
                return ""
            # Replace pipes and newlines that would break markdown tables
            cell_str = (
                str(cell).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            )
            return cell_str.strip()

        # Find the maximum number of columns
        max_cols = max(len(row) for row in rows) if rows else 0

        # Normalize all rows to have the same number of columns
        normalized_rows = []
        for row in rows:
            normalized_row = [sanitize_cell(cell) for cell in row]
            # Pad with empty strings if row is shorter
            while len(normalized_row) < max_cols:
                normalized_row.append("")
            normalized_rows.append(normalized_row)

        # Format CSV data as a readable table
        csv_content = ""
        if normalized_rows:
            # Add header
            csv_content += "| " + " | ".join(normalized_rows[0]) + " |\n"
            csv_content += "|" + "|".join(["---"] * max_cols) + "|\n"

            # Add data rows
            for row in normalized_rows[1:]:
                csv_content += "| " + " | ".join(row) + " |\n"

        was_truncated = False

        # Limit content using the constant
        if len(csv_content) > TEXT_ATTACHMENT_MAX_LENGTH:
            csv_content = (
                csv_content[:TEXT_ATTACHMENT_MAX_LENGTH] + "\n\n[Content truncated...]"
            )
            was_truncated = True

        return ParsedAttachment(
            name=file_name,
            content=csv_content,
            url=attachment_url,
            truncated=was_truncated,
            is_text=True,
        )

    except Exception as read_error:
        logger.error(f"Error reading CSV file {attachment_file}: {read_error}")
        return ParsedAttachment(
            name=attachment_url.split("/")[-1],
            content="",
            url=attachment_url,
            error=f"CSV file, but could not read: {str(read_error)}",
        )


def _parse_text_file(attachment_file: str, attachment_url: str) -> ParsedAttachment:
    """
    Parse a text file (.txt, .md, .markdown, .plain).

    Args:
        attachment_file: Local path to the downloaded text file
        attachment_url: Original URL of the attachment

    Returns:
        ParsedAttachment object with parsed text content or error
    """
    try:
        with open(attachment_file, "r", encoding="utf-8") as f:
            text_content = f.read()
            file_name = attachment_url.split("/")[-1]
            was_truncated = False

            # Limit text content using the constant
            if len(text_content) > TEXT_ATTACHMENT_MAX_LENGTH:
                text_content = (
                    text_content[:TEXT_ATTACHMENT_MAX_LENGTH]
                    + "\n\n[Content truncated...]"
                )
                was_truncated = True

            return ParsedAttachment(
                name=file_name,
                content=text_content,
                url=attachment_url,
                truncated=was_truncated,
                is_text=True,
            )
    except Exception as read_error:
        logger.error(f"Error reading text file {attachment_file}: {read_error}")
        return ParsedAttachment(
            name=attachment_url.split("/")[-1],
            content="",
            url=attachment_url,
            error=f"Text file, but could not read: {str(read_error)}",
        )


def _parse_pdf_file(attachment_file: str, attachment_url: str) -> ParsedAttachment:
    """
    Parse a PDF file and extract text content.

    Args:
        attachment_file: Local path to the downloaded PDF file
        attachment_url: Original URL of the attachment

    Returns:
        ParsedAttachment object with parsed PDF content or error
    """
    try:
        with pdfplumber.open(attachment_file) as pdf:
            # Extract text from all pages
            text_content = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"

            file_name = attachment_url.split("/")[-1]
            was_truncated = False

            # Limit text content using the constant
            if len(text_content) > TEXT_ATTACHMENT_MAX_LENGTH:
                text_content = (
                    text_content[:TEXT_ATTACHMENT_MAX_LENGTH]
                    + "\n\n[Content truncated...]"
                )
                was_truncated = True

            # Only return if we successfully extracted text
            if text_content.strip():
                return ParsedAttachment(
                    name=file_name,
                    content=text_content,
                    url=attachment_url,
                    truncated=was_truncated,
                    is_text=True,
                )
            else:
                return ParsedAttachment(
                    name=file_name,
                    content="",
                    url=attachment_url,
                    error="PDF file with no extractable text",
                )
    except Exception as read_error:
        logger.error(f"Error reading PDF file {attachment_file}: {read_error}")
        return ParsedAttachment(
            name=attachment_url.split("/")[-1],
            content="",
            url=attachment_url,
            error=f"PDF file, but could not extract text: {str(read_error)}",
        )


def parse_attachment(attachment_url: str) -> ParsedAttachment:
    """
    Main entrypoint function for parsing various file types.

    Downloads and parses a file attachment, automatically detecting its type
    and routing to the appropriate parser.

    Args:
        attachment_url: URL of the attachment to parse

    Returns:
        ParsedAttachment object with parsed content, metadata, and any errors
    """
    try:
        # Download the file
        attachment_file = download_file(
            attachment_url,
            os.path.join(FILE_CACHE_DIR, attachment_url.split("/")[-1]),
            overwrite=False,
        )

        # Detect MIME type
        mime_type = magic.from_file(attachment_file, mime=True)
        file_name = attachment_url.split("/")[-1]

        # Handle CSV files first (before text/plain check, since CSVs can be detected as text/plain)
        if (
            mime_type == "text/csv"
            or mime_type == "application/csv"
            or _url_has_extension(attachment_url, SUPPORTED_CSV_EXTENSION)
        ):
            return _parse_csv_file(attachment_file, attachment_url)

        # Handle text files (.txt, .md, .plain)
        elif mime_type in [
            "text/plain",
            "text/markdown",
            "text/x-markdown",
        ] or _url_has_extension(attachment_url, SUPPORTED_TEXT_EXTENSIONS):
            return _parse_text_file(attachment_file, attachment_url)

        # Handle PDF files
        elif mime_type == "application/pdf" or _url_has_extension(
            attachment_url, SUPPORTED_PDF_EXTENSION
        ):
            return _parse_pdf_file(attachment_file, attachment_url)

        # Handle video files
        elif "video" in mime_type:
            return ParsedAttachment(
                name=file_name,
                content=f"{attachment_url} (The asset is a video, the corresponding image attachment is its first frame.)",
                url=attachment_url,
                is_visual=True,
                is_video=True,
            )

        # Handle audio files
        elif "audio" in mime_type:
            return ParsedAttachment(
                name=file_name,
                content=f"{attachment_url} (The asset is an audio file.)",
                url=attachment_url,
                is_audio=True,
            )

        # Handle image files
        elif "image" in mime_type:
            return ParsedAttachment(
                name=file_name,
                content=f"{attachment_url}",
                url=attachment_url,
                is_visual=True,
            )

        # Handle unsupported file types with helpful messages
        else:
            file_ext = file_name.lower().split(".")[-1] if "." in file_name else ""

            if file_ext in UNSUPPORTED_FILE_FORMATS:
                error_msg = f"⚠️ UNSUPPORTED FILE TYPE: {file_name} - {UNSUPPORTED_FILE_FORMATS[file_ext]}"
                return ParsedAttachment(
                    name=file_name, content="", url=attachment_url, error=error_msg
                )
            else:
                return ParsedAttachment(
                    name=file_name,
                    content="",
                    url=attachment_url,
                    error=f"Unsupported file type - Mime type: {mime_type}",
                )

    except Exception as e:
        logger.error(f"Error downloading/parsing attachment {attachment_url}: {e}")
        return ParsedAttachment(
            name=attachment_url.split("/")[-1],
            content="",
            url=attachment_url,
            error=str(e),
        )


def process_attachments_for_message(
    attachments: List[str],
) -> Tuple[List[ParsedAttachment], List[str], List[str]]:
    """
    Process a list of attachment URLs and categorize them.

    Args:
        attachments: List of attachment URLs to process

    Returns:
        Tuple of (all_parsed_attachments, attachment_lines, attachment_errors):
        - all_parsed_attachments: List of all ParsedAttachment objects (text and media)
        - attachment_lines: List of formatted strings for media attachments
        - attachment_errors: List of error messages for failed attachments
    """
    all_parsed = []
    attachment_lines = []
    attachment_errors = []

    for attachment_url in attachments:
        parsed = parse_attachment(attachment_url)
        all_parsed.append(parsed)  # Add ALL parsed attachments

        if parsed.error:
            attachment_errors.append(f"* {attachment_url}: {parsed.error}")
        elif parsed.is_visual:
            attachment_lines.append(f"* {parsed.content}")

    return all_parsed, attachment_lines, attachment_errors
