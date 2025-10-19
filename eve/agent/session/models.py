import json
import os
import csv
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

import magic
import pdfplumber
from bson import ObjectId
from pydantic import ConfigDict, Field, BaseModel, field_serializer
from loguru import logger

from eve.utils import download_file, image_to_base64, prepare_result, dumps_json
from eve.mongo import Collection, Document
from eve.tool import Tool

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


class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]
    task: Optional[ObjectId] = None
    cost: Optional[float] = None
    status: Optional[
        Literal["pending", "running", "completed", "failed", "cancelled"]
    ] = None
    result: Optional[List[Dict[str, Any]]] = None
    reactions: Optional[Dict[str, List[str]]] = None
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def from_openai(tool_call):
        return ToolCall(
            id=tool_call.id,
            tool=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
        )

    @staticmethod
    def from_anthropic(tool_call):
        return ToolCall(id=tool_call.id, tool=tool_call.name, args=tool_call.input)

    def openai_call_schema(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.tool, "arguments": json.dumps(self.args)},
        }

    def anthropic_call_schema(self):
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.tool,
            "input": self.args,
        }

    def anthropic_result_schema(self, truncate_images=False, include_thoughts=False):
        content = {"status": self.status}

        if self.status == "completed":
            content["result"] = prepare_result(self.result)
            file_outputs = [
                o["url"]
                for r in content["result"]
                for o in r.get("output", [])
                if isinstance(o, dict) and o.get("url")
            ]
            file_outputs = [
                o
                for o in file_outputs
                if o and o.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS)
            ]
            try:
                files = [
                    download_file(
                        url,
                        os.path.join(FILE_CACHE_DIR, url.split("/")[-1]),
                        overwrite=False,
                    )
                    for url in file_outputs
                ]
                image_block = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_base64(
                                file_path,
                                max_size=IMAGE_MAX_SIZE,
                                quality=IMAGE_QUALITY,
                                truncate=truncate_images,
                            ),
                        },
                    }
                    for file_path in files
                ]

                if image_block:
                    image_block_content = dumps_json(content["result"])
                    text_block = [{"type": "text", "text": image_block_content}]
                    content = text_block + image_block
                else:
                    content = dumps_json(content)

            except Exception as e:
                logger.warning("Warning: Can not inject image results:", e)
                content = dumps_json(content)

        # For Anthropic: if content is a list (text + images), use it directly
        # If content is a dict/object, JSON encode it
        if isinstance(content, dict):
            content = dumps_json(content)

        result = {"type": "tool_result", "tool_use_id": self.id, "content": content}

        if self.status == "failed":
            result["is_error"] = True

        return result

    def openai_result_schema(self, truncate_images=False):
        content = {"status": self.status}

        if self.status == "failed":
            content["error"] = self.error
        else:
            content["result"] = prepare_result(self.result)

        return {
            "role": "tool",
            "name": self.tool,
            "content": dumps_json(content),
            "tool_call_id": self.id,
        }


class EdenMessageType(Enum):
    AGENT_ADD = "agent_add"
    AGENT_REMOVE = "agent_remove"


class EdenMessageAgentData(BaseModel):
    id: ObjectId
    name: str
    avatar: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EdenMessageData(BaseModel):
    message_type: EdenMessageType
    agents: Optional[List[EdenMessageAgentData]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("message_type")
    def serialize_message_type(self, value: EdenMessageType) -> str:
        return value.value


@Collection("channels")
class Channel(Document):
    type: Literal["eden", "discord", "telegram", "twitter", "farcaster"]
    key: Optional[str] = None


class ChatMessageObservability(BaseModel):
    provider: Literal["langfuse"] = "langfuse"
    session_id: Optional[str] = None
    trace_id: Optional[str] = None  # Langfuse trace ID
    generation_id: Optional[str] = None  # Langfuse generation ID
    tokens_spent: Optional[int] = None
    sentry_trace_id: Optional[str] = None  # Sentry distributed trace ID for correlation
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("messages")
class ChatMessage(Document):
    role: Literal[
        "user",
        "assistant",
        "system",
        "tool",
        "eden",
    ]
    name: Optional[str] = None

    channel: Optional[Channel] = None
    session: Optional[ObjectId] = None
    sender: Optional[ObjectId] = None
    eden_message_data: Optional[EdenMessageData] = None
    reply_to: Optional[ObjectId] = None
    pinned: Optional[bool] = False

    content: str = ""
    reactions: Optional[Dict[str, List[str]]] = {}

    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    cost: Optional[float] = None  # todo: add cost

    trigger: Optional[ObjectId] = None  # monitor if message the result of a trigger
    observability: Optional[ChatMessageObservability] = None
    finish_reason: Optional[str] = None
    thought: Optional[List[Dict[str, Any]]] = None
    llm_config: Optional[Dict[str, Any]] = None
    apiKey: Optional[ObjectId] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def react(self, user: ObjectId, reaction: str):
        if reaction not in self.reactions:
            self.reactions[reaction] = []
        self.reactions[reaction].append(user)

    def as_user_message(self):
        if self.role == "user":
            return self

        attachments = self.attachments.copy()
        for tc in self.tool_calls or []:
            result = prepare_result(tc.result) or []
            urls = [
                item["url"] for r in result for item in r["output"] if item.get("url")
            ]
            attachments.extend([r for r in urls if r])

        content = self.content.strip()
        if self.tool_calls:
            tool_calls = "\n".join(
                [dumps_json(tc.model_dump()) for tc in self.tool_calls]
            )
            content += f"\n\n<Tool calls>\n{tool_calls}\n</Tool calls>"

        return self.model_copy(
            update={"role": "user", "content": content, "attachments": attachments}
        )

    def as_assistant_message(self):
        if self.role == "assistant":
            return self

        return self.model_copy(update={"role": "assistant"})

    def as_system_message(self):
        if self.role == "system":
            return self

        return self.model_copy(update={"role": "system"})

    def filter_cancelled_tool_calls(self):
        """Return a copy of the message with cancelled tool calls filtered out"""
        if not self.tool_calls:
            return self

        filtered_tool_calls = [tc for tc in self.tool_calls if tc.status != "cancelled"]

        return self.model_copy(update={"tool_calls": filtered_tool_calls})

    def _get_content_block(self, schema, truncate_images=False):
        """Assemble user message content block"""

        # start with original message content
        content = self.content.strip() or ""

        # let claude see names
        if self.name and schema == "anthropic":
            content = f"<User>{self.name}</User>\n\n{content}"

        if self.attachments:
            # append attachments info (url and type) to content
            attachment_lines = []
            attachment_files = []
            attachment_errors = []
            text_attachments = []
            truncated_files = []  # Track which files were truncated

            for attachment in self.attachments:
                try:
                    attachment_file = download_file(
                        attachment,
                        os.path.join(FILE_CACHE_DIR, attachment.split("/")[-1]),
                        overwrite=False,
                    )
                    mime_type = magic.from_file(attachment_file, mime=True)

                    # Handle CSV files first (before text/plain check, since CSVs can be detected as text/plain)
                    if mime_type == "text/csv" or attachment.lower().endswith(".csv"):
                        try:
                            file_name = attachment.split("/")[-1]

                            # Read the file once with appropriate encoding
                            try:
                                with open(attachment_file, "r", encoding="utf-8") as f:
                                    file_content = f.read()
                            except UnicodeDecodeError:
                                with open(attachment_file, "r", encoding="latin-1") as f:
                                    file_content = f.read()

                            # Check if file is empty or whitespace-only
                            if not file_content or not file_content.strip():
                                attachment_lines.append(
                                    f"* {attachment}: (CSV file is empty)"
                                )
                                continue

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
                            has_content = any(
                                any(cell.strip() for cell in row)
                                for row in rows
                            )

                            if not has_content:
                                attachment_lines.append(
                                    f"* {attachment}: (CSV file is empty)"
                                )
                            else:
                                # Helper function to sanitize cell content for markdown tables
                                def sanitize_cell(cell):
                                    if cell is None:
                                        return ""
                                    # Replace pipes and newlines that would break markdown tables
                                    cell_str = str(cell).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
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
                                        csv_content[:TEXT_ATTACHMENT_MAX_LENGTH]
                                        + "\n\n[Content truncated...]"
                                    )
                                    was_truncated = True
                                    truncated_files.append(file_name)

                                if csv_content.strip():
                                    text_attachments.append(
                                        {
                                            "name": file_name,
                                            "content": csv_content,
                                            "url": attachment,
                                            "truncated": was_truncated,
                                        }
                                    )
                        except Exception as read_error:
                            logger.error(
                                f"Error reading CSV file {attachment_file}: {read_error}"
                            )
                            attachment_lines.append(
                                f"* {attachment}: (CSV file, but could not read: {str(read_error)})"
                            )

                    # Handle text files (.txt, .md, .plain)
                    elif mime_type in [
                        "text/plain",
                        "text/markdown",
                        "text/x-markdown",
                    ] or attachment.lower().endswith(SUPPORTED_TEXT_EXTENSIONS):
                        try:
                            with open(attachment_file, "r", encoding="utf-8") as f:
                                text_content = f.read()
                                file_name = attachment.split("/")[-1]
                                was_truncated = False

                                # Limit text content using the constant
                                if len(text_content) > TEXT_ATTACHMENT_MAX_LENGTH:
                                    text_content = (
                                        text_content[:TEXT_ATTACHMENT_MAX_LENGTH]
                                        + "\n\n[Content truncated...]"
                                    )
                                    was_truncated = True
                                    truncated_files.append(file_name)

                                text_attachments.append(
                                    {
                                        "name": file_name,
                                        "content": text_content,
                                        "url": attachment,
                                        "truncated": was_truncated,
                                    }
                                )
                        except Exception as read_error:
                            logger.error(
                                f"Error reading text file {attachment_file}: {read_error}"
                            )
                            attachment_lines.append(
                                f"* {attachment}: (Text file, but could not read: {str(read_error)})"
                            )

                    # Handle PDF files
                    elif mime_type == "application/pdf" or attachment.lower().endswith(SUPPORTED_PDF_EXTENSION):
                        try:
                            with pdfplumber.open(attachment_file) as pdf:
                                # Extract text from all pages
                                text_content = ""
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text_content += page_text + "\n"

                                file_name = attachment.split("/")[-1]
                                was_truncated = False

                                # Limit text content using the constant
                                if len(text_content) > TEXT_ATTACHMENT_MAX_LENGTH:
                                    text_content = (
                                        text_content[:TEXT_ATTACHMENT_MAX_LENGTH]
                                        + "\n\n[Content truncated...]"
                                    )
                                    was_truncated = True
                                    truncated_files.append(file_name)

                                # Only add if we successfully extracted text
                                if text_content.strip():
                                    text_attachments.append(
                                        {
                                            "name": file_name,
                                            "content": text_content,
                                            "url": attachment,
                                            "truncated": was_truncated,
                                        }
                                    )
                                else:
                                    attachment_lines.append(
                                        f"* {attachment}: (PDF file with no extractable text)"
                                    )
                        except Exception as read_error:
                            logger.error(
                                f"Error reading PDF file {attachment_file}: {read_error}"
                            )
                            attachment_lines.append(
                                f"* {attachment}: (PDF file, but could not extract text: {str(read_error)})"
                            )

                    elif "video" in mime_type:
                        attachment_lines.append(
                            f"* {attachment} (The asset is a video, the corresponding image attachment is its first frame.)"
                        )
                        attachment_files.append(attachment_file)
                    elif "image" in mime_type:
                        attachment_lines.append(f"* {attachment}")
                        attachment_files.append(attachment_file)
                    else:
                        # Handle unsupported file types with helpful messages
                        file_name = attachment.split("/")[-1]
                        file_ext = file_name.lower().split(".")[-1] if "." in file_name else ""

                        if file_ext in UNSUPPORTED_FILE_FORMATS:
                            warning_msg = f"⚠️ UNSUPPORTED FILE TYPE: {file_name} - {UNSUPPORTED_FILE_FORMATS[file_ext]}"
                            attachment_lines.append(f"* {attachment}: {warning_msg}")
                        else:
                            attachment_lines.append(
                                f"* {attachment}: (Unsupported file type - Mime type: {mime_type})"
                            )
                except Exception as e:
                    logger.error("error downloading attachment", e)
                    attachment_errors.append(f"* {attachment}: {str(e)}")

            attachments = ""

            # Add text file contents directly to the message
            if text_attachments:
                for text_att in text_attachments:
                    attachments += f'\n<attached_file name="{text_att["name"]}" url="{text_att["url"]}">\n'
                    attachments += text_att["content"]
                    attachments += f"\n</attached_file>\n"

            # Add truncation notification if any files were truncated
            if truncated_files:
                if attachments:
                    attachments += "\n"
                attachments += f"<truncation_notice>\nNote: The following file(s) exceeded the {TEXT_ATTACHMENT_MAX_LENGTH} character limit and were truncated: {', '.join(truncated_files)}\n</truncation_notice>\n"

            if attachment_lines:
                if attachments:
                    attachments += "\n"
                attachments += "The attached images correspond to the following urls:\n"
                attachments += "\n".join(attachment_lines)

            if attachment_errors:
                if attachments:
                    attachments += "\n"
                attachments += "The following files failed to attach:\n"
                attachments += "\n".join(attachment_errors)

            if attachments:
                attachments = f"<attachments>\n{attachments}\n</attachments>"
                content += f"\n{attachments}"

            # add image blocks
            block = []

            if schema == "anthropic":
                for file_path in attachment_files:
                    try:
                        block.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_to_base64(
                                        file_path,
                                        max_size=IMAGE_MAX_SIZE,
                                        quality=IMAGE_QUALITY,
                                        truncate=truncate_images,
                                    ),
                                },
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error processing image {file_path}: {e}")
                        # Skip this image and continue with others
                        continue
            elif schema == "openai":
                for file_path in attachment_files:
                    try:
                        block.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"""data:image/jpeg;base64,{
                                        image_to_base64(
                                            file_path,
                                            max_size=IMAGE_MAX_SIZE,
                                            quality=IMAGE_QUALITY,
                                            truncate=truncate_images,
                                        )
                                    }"""
                                },
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error processing image {file_path}: {e}")
                        # Skip this image and continue with others
                        continue

            if content:
                block.extend([{"type": "text", "text": content.strip()}])

            # Only replace content if we have blocks (images or text)
            if block:
                content = block

        return content

    def anthropic_schema(self, truncate_images=False, include_thoughts=False):
        # System Message
        if self.role == "system":
            return [
                {
                    "role": "system",
                    "content": self.content.strip(),
                }
            ]

        # User Message
        if self.role == "user":
            content = self._get_content_block(
                "anthropic", truncate_images=truncate_images
            )
            return [{"role": "user", "content": content}] if content else []

        # Assistant Message
        else:
            if not self.content and not self.tool_calls:
                return []
            schema = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.content.strip()}]
                    if self.content
                    else [],
                }
            ]

            if self.thought and include_thoughts:
                thinking_blocks = []

                for t in self.thought:
                    if t.get("type") == "thinking" and t.get("signature"):
                        thinking_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": t["thinking"],
                                "signature": t["signature"],
                            }
                        )
                    elif t.get("type") == "redacted_thinking":
                        thinking_blocks.append(
                            {"type": "redacted_thinking", "data": t.get("data", "")}
                        )

                schema[0]["content"] = thinking_blocks + schema[0]["content"]

            if self.tool_calls:
                schema[0]["content"].extend(
                    [t.anthropic_call_schema() for t in self.tool_calls]
                )
                schema.append(
                    {
                        "role": "user",
                        "content": [
                            t.anthropic_result_schema(
                                truncate_images=truncate_images,
                                include_thoughts=include_thoughts,
                            )
                            for t in self.tool_calls
                        ],
                    }
                )
            return schema

    def openai_schema(self, truncate_images=False, include_thoughts=False):
        # System Message
        if self.role == "system":
            return [
                {
                    "role": "system",
                    "content": self.content.strip(),
                }
            ]

        # User Message
        elif self.role == "user":
            return [
                {
                    "role": "user",
                    "content": self._get_content_block(
                        "openai",
                        truncate_images=truncate_images,
                    ),
                    **({"name": self.name} if self.name else {}),
                }
            ]

        # Assistant Message
        else:
            schema = [
                {
                    "role": "assistant",
                    "content": self.content.strip(),
                    "function_call": None,
                    "tool_calls": None,
                }
            ]

            if self.thought and include_thoughts:
                schema[0]["thinking_blocks"] = []
                for t in self.thought:
                    # injecting anthropic thinking blocks into openai schema because litellm uses opnenai schema
                    # this might fail if going through openai api directly.
                    # not sure if there is a better way ¯\_(ツ)_/¯
                    if t.get("type") == "thinking":
                        schema[0]["thinking_blocks"].append(
                            {
                                "type": "thinking",
                                "thinking": t["thinking"],
                                **(
                                    {"signature": t["signature"]}
                                    if "signature" in t
                                    else {}
                                ),
                            }
                        )
                    elif t.get("type") == "redacted_thinking":
                        schema[0]["thinking_blocks"].append(
                            {"type": "redacted_thinking", "data": t.get("data", "")}
                        )

            if self.tool_calls:
                schema[0]["tool_calls"] = [
                    t.openai_call_schema() for t in self.tool_calls
                ]
                schema.extend(
                    [
                        t.openai_result_schema(truncate_images=truncate_images)
                        for t in self.tool_calls
                    ]
                )

                image_blocks = []
                image_urls = []

                for tool_call in self.tool_calls:
                    if tool_call.status == "completed" and tool_call.result:
                        result = prepare_result(tool_call.result)
                        file_outputs = [
                            o["url"]
                            for r in result
                            for o in r.get("output", [])
                            if isinstance(o, dict) and o.get("url")
                        ]
                        image_outputs = [
                            o
                            for o in file_outputs
                            if o and o.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS)
                        ]

                        for image_url in image_outputs:
                            try:
                                image_path = download_file(
                                    image_url,
                                    os.path.join(
                                        FILE_CACHE_DIR,
                                        image_url.split("/")[-1],
                                    ),
                                    overwrite=False,
                                )

                                image_blocks.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_to_base64(image_path, max_size=IMAGE_MAX_SIZE, quality=IMAGE_QUALITY, truncate=truncate_images)}"
                                        },
                                    }
                                )

                                if image_url.lower().endswith(("webm", "mp4")):
                                    image_urls.append(
                                        f"{image_url} (This url is a video, the corresponding image attachment is its first frame.)"
                                    )
                                else:
                                    image_urls.append(image_url)

                            except Exception as e:
                                logger.error(f"Error processing image {image_url}: {e}")
                                continue

                # Create single synthetic user message if we have any images
                if image_blocks:
                    if len(image_blocks) == 1:
                        content = f"The attached image corresponds to the tool result with url {image_urls[0]}"
                    else:
                        content = "The order of the attached images corresponds to the tool results whose URLs are: \n"
                        content += "\n* ".join(image_urls)

                    schema.append(
                        {
                            "role": "user",
                            "name": "system_tool_result",
                            "content": [{"type": "text", "text": content}]
                            + image_blocks,
                        }
                    )

            return schema


@dataclass
class ChatMessageRequestInput:
    content: str
    role: Optional[Literal["user", "system"]] = "user"
    attachments: Optional[List[str]] = None
    sender_name: Optional[str] = None


class UpdateType(Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_MESSAGE = "assistant_message"
    USER_MESSAGE = "user_message"
    TOOL_COMPLETE = "tool_complete"
    TOOL_CANCELLED = "tool_cancelled"
    ERROR = "error"
    END_PROMPT = "end_prompt"


class SessionUpdateConfig(BaseModel):
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    deployment_id: Optional[str] = None
    discord_channel_id: Optional[str] = None
    discord_message_id: Optional[str] = None
    discord_user_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_message_id: Optional[str] = None
    telegram_thread_id: Optional[str] = None
    farcaster_hash: Optional[str] = None
    farcaster_author_fid: Optional[int] = None
    farcaster_message_id: Optional[str] = None
    twitter_tweet_to_reply_id: Optional[str] = None
    user_is_bot: Optional[bool] = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionUpdate(BaseModel):
    type: UpdateType
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    update_config: Optional[SessionUpdateConfig] = None
    agent: Optional[Dict[str, Any]] = None
    session_run_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMTraceMetadata(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMContextMetadata(BaseModel):
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    trace_name: Optional[str] = None
    generation_name: Optional[str] = None
    generation_id: Optional[str] = None
    trace_metadata: Optional[LLMTraceMetadata] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMThinkingSettings(BaseModel):
    """LLM configuration and thinking settings for an agent."""

    policy: Optional[str] = "auto"  # "auto", "off", "always"
    effort_cap: Optional[str] = "medium"  # "low", "medium", "high"
    effort_instructions: Optional[str] = (
        "Use low for simple tasks, high for complex reasoning-intensive tasks."
    )


@dataclass
class LLMConfig:
    model: Optional[str] = "gpt-4o-mini"
    fallback_models: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    response_format: Optional[BaseModel] = None
    thinking: Optional[LLMThinkingSettings] = None
    reasoning_effort: Optional[str] = (
        None  # Final resolved reasoning effort (low/medium/high)
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class LLMContext:
    messages: List[ChatMessage]
    config: LLMConfig = field(default_factory=LLMConfig)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    metadata: LLMContextMetadata = None
    enable_tracing: bool = True


class ActorSelectionMethod(Enum):
    RANDOM = "random"
    RANDOM_EXCLUDE_LAST = "random_exclude_last"


class SessionAutonomySettings(BaseModel):
    auto_reply: bool = False
    reply_interval: int = 0
    actor_selection_method: ActorSelectionMethod = ActorSelectionMethod.RANDOM

    @field_serializer("actor_selection_method")
    def serialize_actor_selection_method(self, value: ActorSelectionMethod) -> str:
        return value.value


class SessionBudget(BaseModel):
    token_budget: Optional[int] = None
    manna_budget: Optional[float] = None
    turn_budget: Optional[int] = None
    tokens_spent: Optional[int] = 0
    manna_spent: Optional[float] = 0
    turns_spent: Optional[int] = 0


class SessionMemoryContext(BaseModel):
    """Session-level memory context and caching"""

    # Core memory caching
    cached_memory_context: Optional[str] = None
    memory_context_timestamp: Optional[datetime] = None

    # Session activity tracking
    last_activity: Optional[datetime] = None
    last_memory_message_id: Optional[ObjectId] = None
    messages_since_memory_formation: int = 0

    # Memory freshness timestamps
    agent_memory_timestamp: Optional[datetime] = None
    user_memory_timestamp: Optional[datetime] = None

    # Episode memory caching
    cached_episode_memories: Optional[List[Dict[str, Any]]] = None
    episode_memories_timestamp: Optional[datetime] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionExtras(BaseModel):
    """Additional session configuration flags"""

    exclude_memory: Optional[bool] = False  # If True, memory won't be passed to system prompt

    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("sessions")
class Session(Document):
    owner: ObjectId
    users: Optional[List[ObjectId]] = None  # List of allowed users (defaults to null)
    session_key: Optional[str] = None
    channel: Optional[Channel] = None
    parent_session: Optional[ObjectId] = None
    agents: List[ObjectId] = Field(default_factory=list)
    status: Literal["active", "archived"] = "active"
    messages: List[ObjectId] = Field(default_factory=list)
    memory_context: Optional[SessionMemoryContext] = Field(
        default_factory=SessionMemoryContext
    )
    title: Optional[str] = None
    scenario: Optional[str] = None
    autonomy_settings: Optional[SessionAutonomySettings] = None
    last_actor_id: Optional[ObjectId] = None
    budget: SessionBudget = SessionBudget()
    platform: Optional[Literal["discord", "telegram", "twitter", "farcaster"]] = None
    trigger: Optional[ObjectId] = None
    active_requests: Optional[List[str]] = []
    extras: Optional[SessionExtras] = None  # Additional session configuration flags

    def get_messages(self):
        messages = ChatMessage.find({"session": self.id})
        messages = sorted(messages, key=lambda x: x.createdAt)
        return messages

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance"""
        collection = cls.get_collection()

        # Compound index for cold session detection query
        # Optimizes: Session.find({"memory_context.last_activity": {"$lt": cutoff_time},
        #                         "memory_context.messages_since_memory_formation": {"$gt": 0},
        #                         "status": "active"})
        collection.create_index(
            [
                ("memory_context.last_activity", 1),
                ("memory_context.messages_since_memory_formation", 1),
                ("status", 1),
            ],
            name="cold_sessions_idx",
            background=True,
        )


@dataclass
class NotificationConfig:
    """Configuration for notifications to send upon session completion"""

    user_id: str
    notification_type: str = "session_complete"
    title: str = "Session Complete"
    message: str = "Your session has completed successfully"
    trigger_id: Optional[str] = None
    agent_id: Optional[str] = None
    priority: str = "normal"
    metadata: Optional[Dict[str, Any]] = None
    success_notification: bool = True
    failure_notification: bool = True
    success_title: Optional[str] = None
    success_message: Optional[str] = None
    failure_title: Optional[str] = None
    failure_message: Optional[str] = None


@dataclass
class SessionCreationArgs:
    """Arguments for creating a new session"""
    owner_id: Optional[str] = None
    agents: Optional[List[str]] = None
    title: Optional[str] = None
    scenario: Optional[str] = None
    budget: Optional[float] = None
    trigger: Optional[str] = None
    session_key: Optional[str] = None
    platform: Optional[str] = None


@dataclass
class PromptSessionContext:
    # Session can be provided directly, or will be loaded/created based on session_id/creation_args
    session: Optional[Session] = None
    session_id: Optional[str] = None
    session_creation_args: Optional[SessionCreationArgs] = None

    initiating_user_id: str = None  # The user who owns/initiates the session
    message: Optional[ChatMessageRequestInput] = None
    update_config: Optional[SessionUpdateConfig] = None
    actor_agent_ids: Optional[List[str]] = None
    llm_config: Optional[LLMConfig] = None

    # overrides all tools if set, otherwise uses actor's tools
    tools: Optional[Dict[str, Any]] = None
    # extra tools added to the base or actor's tools
    extra_tools: Optional[Dict[str, Any]] = None
    tool_choice: Optional[str] = None

    notification_config: Optional[NotificationConfig] = None
    thinking_override: Optional[bool] = None
    acting_user_id: Optional[str] = None
    trigger: Optional[ObjectId] = None
    api_key_id: Optional[str] = None  # API key ID to attach to messages

    session_run_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop: Optional[str] = None
    tokens_spent: Optional[int] = None
    thought: Optional[List[Dict[str, Any]]] = None


class ClientType(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
    TWITTER = "twitter"
    SHOPIFY = "shopify"
    PRINTIFY = "printify"
    CAPTIONS = "captions"
    TIKTOK = "tiktok"


class NotificationType(Enum):
    TRIGGER_COMPLETE = "trigger_complete"
    TRIGGER_FAILED = "trigger_failed"
    TRIGGER_STARTED = "trigger_started"
    SESSION_COMPLETE = "session_complete"
    SESSION_FAILED = "session_failed"
    SESSION_MESSAGE = "session_message"
    AGENT_MENTIONED = "agent_mentioned"
    SYSTEM_ALERT = "system_alert"
    AGENT_PERMISSION_ADDED = "agent_permission_added"
    AGENT_PERMISSION_REMOVED = "agent_permission_removed"


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    IN_APP = "in_app"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"


# Base Models
class AllowlistItem(BaseModel):
    id: str
    note: Optional[str] = None


# Discord Models
class DiscordAllowlistItem(AllowlistItem):
    pass


class DeploymentSettingsDiscord(BaseModel):
    oauth_client_id: Optional[str] = None
    oauth_url: Optional[str] = None
    channel_allowlist: Optional[List[DiscordAllowlistItem]] = None
    read_access_channels: Optional[List[DiscordAllowlistItem]] = None
    enable_discord_dm: Optional[bool] = False


class DeploymentSecretsDiscord(BaseModel):
    token: str
    application_id: Optional[str] = None


# Telegram Models
class TelegramAllowlistItem(AllowlistItem):
    pass


class DeploymentSettingsTelegram(BaseModel):
    topic_allowlist: Optional[List[TelegramAllowlistItem]] = None


class DeploymentSecretsTelegram(BaseModel):
    token: str
    webhook_secret: Optional[str] = None


# Farcaster Models
class DeploymentSettingsFarcaster(BaseModel):
    webhook_id: Optional[str] = None
    auto_reply: Optional[bool] = False


class DeploymentSecretsFarcaster(BaseModel):
    mnemonic: str
    neynar_webhook_secret: Optional[str] = None


# Twitter Models
class DeploymentSettingsTwitter(BaseModel):
    username: Optional[str] = None


class DeploymentSecretsTwitter(BaseModel):
    # OAuth 2.0 fields
    access_token: str
    refresh_token: Optional[str] = None
    twitter_id: str
    expires_at: Optional[datetime] = None
    username: str


# Shopify Models
class DeploymentSettingsShopify(BaseModel):
    pass


class DeploymentSecretsShopify(BaseModel):
    store_name: str
    access_token: str
    location_id: str


# Printify Models
class DeploymentSettingsPrintify(BaseModel):
    pass


class DeploymentSecretsPrintify(BaseModel):
    api_token: str
    shop_id: str


# Captions Models
class DeploymentSettingsCaptions(BaseModel):
    pass


class DeploymentSecretsCaptions(BaseModel):
    api_key: str


# TikTok Models
class DeploymentSettingsTiktok(BaseModel):
    pass


class DeploymentSecretsTiktok(BaseModel):
    access_token: str
    refresh_token: str
    open_id: str
    expires_at: datetime
    username: Optional[str] = None


# Combined Models
class DeploymentSecrets(BaseModel):
    discord: DeploymentSecretsDiscord | None = None
    telegram: DeploymentSecretsTelegram | None = None
    farcaster: DeploymentSecretsFarcaster | None = None
    twitter: DeploymentSecretsTwitter | None = None
    shopify: DeploymentSecretsShopify | None = None
    printify: DeploymentSecretsPrintify | None = None
    captions: DeploymentSecretsCaptions | None = None
    tiktok: DeploymentSecretsTiktok | None = None


class DeploymentConfig(BaseModel):
    discord: DeploymentSettingsDiscord | None = None
    telegram: DeploymentSettingsTelegram | None = None
    farcaster: DeploymentSettingsFarcaster | None = None
    twitter: DeploymentSettingsTwitter | None = None
    shopify: DeploymentSettingsShopify | None = None
    printify: DeploymentSettingsPrintify | None = None
    captions: DeploymentSettingsCaptions | None = None
    tiktok: DeploymentSettingsTiktok | None = None


@Collection("deployments2")
class Deployment(Document):
    agent: ObjectId
    user: ObjectId
    platform: ClientType
    valid: Optional[bool] = None
    secrets: Optional[DeploymentSecrets]
    config: Optional[DeploymentConfig]

    def __init__(self, **data):
        # Convert string to ClientType enum if needed
        if "platform" in data and isinstance(data["platform"], str):
            data["platform"] = ClientType(data["platform"])
        super().__init__(**data)

    def model_dump(self, *args, **kwargs):
        """Override model_dump to convert enum to string for MongoDB"""
        data = super().model_dump(*args, **kwargs)
        if "platform" in data and isinstance(data["platform"], ClientType):
            data["platform"] = data["platform"].value
        return data

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent", 1), ("platform", 1)], unique=True)

    def get_allowed_channels(self):
        """Get allowed channels for the deployment"""
        if self.platform == ClientType.DISCORD:
            return self.config.discord.channel_allowlist
        elif self.platform == ClientType.TELEGRAM:
            return self.config.telegram.topic_allowlist
        return []


@Collection("usernotifications")
class Notification(Document):
    """Notification model for delivering app notifications to users"""

    # Core fields
    user: ObjectId  # recipient user
    type: NotificationType
    title: str
    message: str

    # Status tracking
    read: bool = False
    read_at: Optional[datetime] = None
    priority: NotificationPriority = NotificationPriority.NORMAL

    # Related entities
    trigger: Optional[ObjectId] = None
    session: Optional[ObjectId] = None
    agent: Optional[ObjectId] = None

    # Delivery management
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )
    delivered_channels: List[NotificationChannel] = Field(default_factory=list)
    delivery_attempted_at: Optional[datetime] = None
    delivery_failed_channels: List[NotificationChannel] = Field(default_factory=list)

    # Metadata and context
    metadata: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None  # URL to navigate to when clicked

    # Lifecycle management
    expires_at: Optional[datetime] = None
    archived: bool = False
    archived_at: Optional[datetime] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("type")
    def serialize_type(self, value: NotificationType) -> str:
        return value.value

    @field_serializer("priority")
    def serialize_priority(self, value: NotificationPriority) -> str:
        return value.value

    @field_serializer("channels")
    def serialize_channels(self, value: List[NotificationChannel]) -> List[str]:
        return [channel.value for channel in value]

    @field_serializer("delivered_channels")
    def serialize_delivered_channels(
        self, value: List[NotificationChannel]
    ) -> List[str]:
        return [channel.value for channel in value]

    @field_serializer("delivery_failed_channels")
    def serialize_delivery_failed_channels(
        self, value: List[NotificationChannel]
    ) -> List[str]:
        return [channel.value for channel in value]

    def mark_as_read(self):
        """Mark notification as read"""
        if not self.read:
            self.read = True
            self.read_at = datetime.now(timezone.utc)
            self.save()

    def mark_delivered(self, channel: NotificationChannel):
        """Mark notification as delivered via specific channel"""
        if channel not in self.delivered_channels:
            self.delivered_channels.append(channel)
            self.save()

    def mark_delivery_failed(self, channel: NotificationChannel):
        """Mark delivery as failed for specific channel"""
        if channel not in self.delivery_failed_channels:
            self.delivery_failed_channels.append(channel)
            self.save()

    def archive(self):
        """Archive the notification"""
        if not self.archived:
            self.archived = True
            self.archived_at = datetime.now(timezone.utc)
            self.save()

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for efficient queries"""
        collection = cls.get_collection()
        # Index for user notifications queries
        collection.create_index([("user", 1), ("created_at", -1)])
        # Index for unread notifications
        collection.create_index([("user", 1), ("read", 1)])
        # Index for cleanup of expired notifications
        collection.create_index([("expires_at", 1)], sparse=True)
        # Index for trigger-related notifications
        collection.create_index([("trigger", 1)], sparse=True)
        # Index for session-related notifications
        collection.create_index([("session", 1)], sparse=True)
