import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from bson import ObjectId
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from eve.agent.llm.file_config import (
    FILE_CACHE_DIR,
    IMAGE_MAX_SIZE,
    IMAGE_QUALITY,
    SUPPORTED_MEDIA_EXTENSIONS,
    TEXT_ATTACHMENT_MAX_LENGTH,
    _url_has_extension,
)
from eve.agent.llm.file_parser import process_attachments_for_message
from eve.mongo import Collection, Document
from eve.tool import Tool
from eve.utils import download_file, dumps_json, image_to_base64, prepare_result

if TYPE_CHECKING:  # pragma: no cover
    from eve.agent.session.instrumentation import PromptSessionInstrumentation


class Reaction(BaseModel):
    user_id: Union[str, ObjectId]
    reaction: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _convert_reactions_from_dict(reactions) -> Optional[List[dict]]:
    """Convert old dict-format reactions to new list format during model validation.

    Handles various legacy formats:
    - None -> None
    - [] (empty list) -> []
    - {} (empty dict) -> []
    - {"emoji": ["user1", "user2"]} -> [{"user_id": "user1", "reaction": "emoji"}, ...]
    - {"user_id": ["emoji1", "emoji2"]} -> [{"user_id": "user_id", "reaction": "emoji1"}, ...]
    - [{"user_id": "x", "reaction": "y"}] -> unchanged (already correct format)
    """
    if reactions is None:
        return None
    if isinstance(reactions, list):
        # Already in list format - validate each item is a proper dict
        # Filter out any malformed entries
        valid_reactions = []
        for item in reactions:
            if isinstance(item, dict) and "user_id" in item and "reaction" in item:
                valid_reactions.append(item)
            elif isinstance(item, dict):
                # Try to salvage partial data
                if "user_id" in item:
                    valid_reactions.append({"user_id": item["user_id"], "reaction": ""})
        return valid_reactions if valid_reactions else []
    if isinstance(reactions, dict):
        if not reactions:  # Empty dict
            return []
        result = []
        for key, values in reactions.items():
            if not isinstance(values, list):
                continue
            # Heuristic: if key looks like an ObjectId or "anonymous", it's {user_id: [reactions]}
            is_user_id_key = key == "anonymous" or (
                len(key) == 24 and all(c in "0123456789abcdef" for c in key.lower())
            )
            if is_user_id_key:
                for reaction in values:
                    result.append({"user_id": key, "reaction": str(reaction)})
            else:
                for user_id in values:
                    uid = str(user_id) if isinstance(user_id, ObjectId) else user_id
                    result.append({"user_id": uid, "reaction": key})
        return result
    # For any other type, return empty list to prevent validation errors
    return []


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
    reactions: Optional[List[Reaction]] = None
    error: Optional[str] = None
    child_session: Optional[ObjectId] = None
    thought_signature: Optional[str] = None  # Gemini thought signature for tool calls

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("reactions", mode="before")
    @classmethod
    def convert_reactions(cls, v):
        return _convert_reactions_from_dict(v)

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
            "id": self.id[:40],  # OpenAI max tool_call_id length is 40
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

        # Include appropriate result based on status
        if self.status == "pending":
            content["result"] = [
                {
                    "status": "pending",
                    "message": "The tool call has not started running yet",
                }
            ]
        elif self.status == "running":
            content["result"] = [
                {"status": "running", "message": "The tool call is still running"}
            ]
        elif self.status == "cancelled":
            content["result"] = (
                prepare_result(self.result)
                if self.result
                else [
                    {
                        "status": "cancelled",
                        "error": "The tool call was cancelled by the user",
                    }
                ]
            )
        elif self.status == "completed":
            content["result"] = prepare_result(self.result)
            file_outputs = [
                o["url"]
                for r in content["result"]
                for o in r.get("output", [])
                if isinstance(o, dict) and isinstance(o.get("url"), str)
            ]
            file_outputs = [
                o
                for o in file_outputs
                if o and _url_has_extension(o, SUPPORTED_MEDIA_EXTENSIONS)
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
            result["content"] = self.error

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
            "tool_call_id": self.id[:40],  # OpenAI max tool_call_id length is 40
        }


class EdenMessageType(Enum):
    AGENT_ADD = "agent_add"
    AGENT_REMOVE = "agent_remove"
    RATE_LIMIT = "rate_limit"
    TRIGGER = "trigger"
    # Legacy conductor types (kept for backwards compatibility with old session data)
    CONDUCTOR_INIT = "conductor_init"  # DEPRECATED - use MODERATOR_START
    CONDUCTOR_TURN = "conductor_turn"  # DEPRECATED - use MODERATOR_PROMPT
    CONDUCTOR_HINT = "conductor_hint"  # DEPRECATED - vestigial
    CONDUCTOR_FINISH = "conductor_finish"  # DEPRECATED - use MODERATOR_FINISH
    # Moderator message types for multi-agent orchestration
    MODERATOR_START = "moderator_start"  # Session initialization output
    MODERATOR_PROMPT = "moderator_prompt"  # Agent prompt action
    MODERATOR_VOTE = "moderator_vote"  # Vote results
    MODERATOR_FINISH = "moderator_finish"  # Session summary/conclusion
    # Private messaging for multi-agent sessions
    PRIVATE_MESSAGE = "private_message"  # Agent-to-agent private DMs


class EdenMessageAgentData(BaseModel):
    id: ObjectId
    name: str
    avatar: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EdenMessageData(BaseModel):
    message_type: EdenMessageType
    agents: Optional[List[EdenMessageAgentData]] = (
        None  # Recipients for PRIVATE_MESSAGE
    )
    sender: Optional[EdenMessageAgentData] = None  # Sender for PRIVATE_MESSAGE
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("message_type")
    def serialize_message_type(self, value: EdenMessageType) -> str:
        return value.value


@Collection("channels")
class Channel(Document):
    type: Literal["eden", "discord", "telegram", "twitter", "farcaster", "app"]
    key: Optional[str] = None
    url: Optional[str] = None  # Permanent link to the message on the platform


class TokenUsageBreakdown(BaseModel):
    total: Optional[int] = None
    cached: Optional[int] = None
    uncached: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatMessageCostDetails(BaseModel):
    model: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "usd"
    input_tokens: Optional[TokenUsageBreakdown] = None
    output_tokens: Optional[TokenUsageBreakdown] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    cached_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class DiscordPlatformObservability(BaseModel):
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    guild_id: Optional[str] = None
    url: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlatformObservability(BaseModel):
    discord: Optional[DiscordPlatformObservability] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatMessageObservability(BaseModel):
    provider: Literal["langfuse"] = "langfuse"
    session_id: Optional[str] = None
    session_run_id: Optional[str] = None
    trace_id: Optional[str] = None  # Langfuse trace ID
    generation_id: Optional[str] = None  # Langfuse generation ID
    langfuse_url: Optional[str] = None
    sentry_url: Optional[str] = None
    input_tokens: Optional[int] = None
    input_tokens_breakdown: Optional[Dict[str, int]] = None
    tokens_spent: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    cached_completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    sentry_trace_id: Optional[str] = None  # Sentry distributed trace ID for correlation
    usage: Optional[LLMUsage] = None
    platforms: Optional[PlatformObservability] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("llm_calls")
class LLMCall(Document):
    """Stores raw LLM API call payloads for inspection and debugging.

    Each LLMCall represents a single call to an LLM provider's API,
    storing the exact payload sent and response received.
    """

    # Provider info
    provider: str  # e.g., "anthropic", "openai", "google"
    model: str  # e.g., "claude-sonnet-4-5"

    # Raw request payload - exactly what was sent to the API
    request_payload: Dict[str, Any]  # Contains system, messages, tools, etc.

    # Raw response data
    response_payload: Optional[Dict[str, Any]] = None

    # Usage/cost info
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Context references
    session: Optional[ObjectId] = None
    agent: Optional[ObjectId] = None
    user: Optional[ObjectId] = None
    session_run_id: Optional[str] = None

    # Status
    status: Literal["pending", "completed", "failed"] = "pending"
    error: Optional[str] = None

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
    session: Optional[List[ObjectId]] = Field(default_factory=list)
    sender: Optional[ObjectId] = None
    triggering_user: Optional[ObjectId] = None
    billed_user: Optional[ObjectId] = None
    agent_owner: Optional[ObjectId] = None
    eden_message_data: Optional[EdenMessageData] = None
    reply_to: Optional[ObjectId] = None
    pinned: Optional[bool] = False

    content: str = ""
    reactions: Optional[List[Reaction]] = None

    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    cost: Optional[float] = None  # todo: add cost

    trigger: Optional[ObjectId] = None  # monitor if message the result of a trigger
    observability: Optional[ChatMessageObservability] = None
    finish_reason: Optional[str] = None
    thought: Optional[List[Dict[str, Any]]] = None
    llm_config: Optional[Dict[str, Any]] = None
    apiKey: Optional[ObjectId] = None
    llm_call: Optional[ObjectId] = None  # Reference to LLMCall document

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("reactions", mode="before")
    @classmethod
    def convert_reactions(cls, v):
        return _convert_reactions_from_dict(v)

    def react(self, user: ObjectId, reaction: str):
        if self.reactions is None:
            self.reactions = []
        self.reactions.append(Reaction(user_id=user, reaction=reaction))

    def as_user_message(self):
        if self.role == "user":
            return self

        attachments = list(self.attachments or [])
        for tc in self.tool_calls or []:
            result = prepare_result(tc.result) or []
            urls = []
            results_iter = result if isinstance(result, list) else [result]
            for entry in results_iter:
                if not isinstance(entry, dict):
                    continue
                output = entry.get("output")
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict):
                            url_value = item.get("url")
                            if url_value:
                                urls.append(url_value)
                        elif isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
                elif isinstance(output, dict):
                    url_value = output.get("url")
                    if url_value:
                        urls.append(url_value)
                elif isinstance(output, str) and output.startswith("http"):
                    urls.append(output)
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

    def update_tool_call(self, tool_call_index: int, **fields: dict):
        set_fields = {f"tool_calls.{tool_call_index}.{k}": v for k, v in fields.items()}
        self.get_collection().update_one(
            {"_id": self.id},
            {"$set": set_fields},
        )

    def _get_content_block(self, schema, truncate_images=False):
        """Assemble user message content block"""

        # start with original message content
        content = self.content.strip() or ""

        # let claude see names
        if self.name and schema == "anthropic":
            content = f"<User>{self.name}</User>\n\n{content}"

        if self.attachments:
            # Process all attachments using the new file parser module
            parsed_attachments, attachment_lines, attachment_errors = (
                process_attachments_for_message(self.attachments)
            )

            # Separate text attachments from media files for different processing
            text_attachments = [att for att in parsed_attachments if att.is_text]
            audio_attachments = [att for att in parsed_attachments if att.is_audio]

            attachment_files = []

            # Download media files for image block processing
            for att in parsed_attachments:
                if att.is_visual:
                    try:
                        attachment_file = download_file(
                            att.url,
                            os.path.join(FILE_CACHE_DIR, att.url.split("/")[-1]),
                            overwrite=False,
                        )
                        attachment_files.append(attachment_file)
                    except Exception as e:
                        logger.error(f"Error downloading media file {att.url}: {e}")

            # Track which files were truncated
            truncated_files = [att.name for att in text_attachments if att.truncated]

            attachments = ""

            # Add text file contents directly to the message
            if text_attachments:
                for text_att in text_attachments:
                    attachments += f'\n<attached_file type="text" name="{text_att.name}" url="{text_att.url}">\n'
                    attachments += text_att.content
                    attachments += "\n</attached_file>\n"

            if audio_attachments:
                for audio_att in audio_attachments:
                    attachments += f'\n<attached_file type="audio" name="{audio_att.name}" url="{audio_att.url}" />\n'

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
                            if isinstance(o, dict) and isinstance(o.get("url"), str)
                        ]
                        image_outputs = [
                            o
                            for o in file_outputs
                            if o and _url_has_extension(o, SUPPORTED_MEDIA_EXTENSIONS)
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

                                if _url_has_extension(image_url, (".webm", ".mp4")):
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

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance"""
        collection = cls.get_collection()

        # Index for looking up messages by channel (Discord message ID, etc.)
        # Used for deduplication when multiple agents share a ChatMessage
        collection.create_index(
            [("channel.type", 1), ("channel.key", 1)],
            name="channel_lookup_idx",
            background=True,
            sparse=True,
        )

        # Index for fast lookup by Discord message id in observability
        collection.create_index(
            [("observability.platforms.discord.message_id", 1)],
            name="discord_message_id_idx",
            background=True,
            sparse=True,
        )


@dataclass
class ChatMessageRequestInput:
    content: str
    channel: Optional[Channel] = None
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
    twitter_tweet_id: Optional[str] = None
    twitter_author_id: Optional[str] = None
    twitter_tweet_to_reply_id: Optional[str] = None
    social_match_reason: Optional[Literal["mention", "reply"]] = None
    user_is_bot: Optional[bool] = False
    email_sender: Optional[str] = None
    email_recipient: Optional[str] = None
    email_subject: Optional[str] = None
    email_message_id: Optional[str] = None
    email_thread_id: Optional[str] = None
    gmail_thread_id: Optional[str] = None
    gmail_message_id: Optional[str] = None
    gmail_history_id: Optional[str] = None
    gmail_from_address: Optional[str] = None
    gmail_to_address: Optional[str] = None
    gmail_subject: Optional[str] = None
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
    agent_name: Optional[str] = None
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
    instrumentation: Optional["PromptSessionInstrumentation"] = None


class SessionSettings(BaseModel):
    conductor_prompt: Optional[str] = None
    delay_interval: int = 0
    mention_force_reply: bool = False


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
    weighted_tokens_since_memory_formation: int = 0

    # Memory freshness timestamps
    agent_memory_timestamp: Optional[datetime] = None
    user_memory_timestamp: Optional[datetime] = None

    # Episode memory caching
    cached_episode_memories: Optional[List[Dict[str, Any]]] = None
    episode_memories_timestamp: Optional[datetime] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionExtras(BaseModel):
    """Additional session configuration flags"""

    incognito: Optional[bool] = (
        False  # If True, session excluded from user and collective memory formation
    )
    is_public: Optional[bool] = False  # If True, session is publicly accessible
    gmail_thread_id: Optional[str] = None
    gmail_initial_message_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("sessions")
class Session(Document):
    owner: ObjectId
    users: List[ObjectId] = Field(
        default_factory=list
    )  # Non-agent users in this session
    session_key: Optional[str] = None
    channel: Optional[Channel] = None
    parent_session: Optional[ObjectId] = None
    agents: List[ObjectId] = Field(default_factory=list)
    status: Literal["active", "paused", "running", "archived", "finished"] = "active"
    messages: List[ObjectId] = Field(default_factory=list)
    memory_context: Optional[SessionMemoryContext] = Field(
        default_factory=SessionMemoryContext
    )
    title: Optional[str] = None
    session_type: Literal["passive", "natural", "automatic"] = "passive"
    settings: Optional[SessionSettings] = Field(default_factory=SessionSettings)
    last_actor_id: Optional[ObjectId] = None
    budget: SessionBudget = SessionBudget()
    platform: Optional[
        Literal["discord", "telegram", "twitter", "farcaster", "gmail", "app", "web"]
    ] = None
    discord_channel_id: Optional[str] = None  # Discord channel ID for discord_post tool
    telegram_chat_id: Optional[str] = None  # Telegram chat ID for telegram_post tool
    trigger: Optional[ObjectId] = None
    active_requests: Optional[List[str]] = []
    extras: Optional[SessionExtras] = None  # Additional session configuration flags
    deleted: Optional[bool] = False
    visible: Optional[bool] = (
        None  # Visibility flag for public listing (None = default behavior)
    )
    context: Optional[str] = None  # Scenario/premise for automatic multi-agent sessions

    # Agent sessions: Maps agent_id (str) to their private agent_session ObjectId
    # Used for multi-agent orchestration where each agent has their own workspace
    agent_sessions: Optional[Dict[str, ObjectId]] = Field(default_factory=dict)

    # The moderator agent's private workspace session (for automatic sessions)
    moderator_session: Optional[ObjectId] = None

    # For agent_sessions: track last synced parent message for bulk updates
    last_parent_message_id: Optional[ObjectId] = None

    # For automatic sessions: timestamp when current delay will end
    # Client can use this to show countdown/progress. None = not waiting.
    waiting_until: Optional[datetime] = None

    @field_validator("agent_sessions", mode="before")
    @classmethod
    def coerce_agent_sessions_to_dict(cls, v):
        """Coerce agent_sessions to dict if it's a list or None."""
        if v is None:
            return {}
        if isinstance(v, list):
            return {}
        return v

    @field_validator("settings", mode="before")
    @classmethod
    def coerce_settings_to_default(cls, v):
        """Coerce settings to default SessionSettings if None (handles old sessions without this field)."""
        if v is None:
            return SessionSettings()
        return v

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
class PromptSessionContext:
    session: Session
    initiating_user_id: str
    message: ChatMessageRequestInput
    session_run_id: Optional[str] = None
    update_config: Optional[SessionUpdateConfig] = None
    actor_agent_ids: Optional[List[str]] = None
    llm_config: Optional[LLMConfig] = None
    tools: Optional[Dict[str, Any]] = None
    extra_tools: Optional[Dict[str, Any]] = None
    tool_choice: Optional[str] = None
    notification_config: Optional[NotificationConfig] = None
    thinking_override: Optional[bool] = None
    acting_user_id: Optional[str] = None
    trigger: Optional[ObjectId] = None
    api_key_id: Optional[str] = None
    selection_limit: Optional[int] = None  # Override default message selection limit


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop: Optional[str] = None
    tokens_spent: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    cached_completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    usage: Optional[LLMUsage] = None
    thought: Optional[List[Dict[str, Any]]] = None
    cost_metadata: Optional[ChatMessageCostDetails] = None
    llm_call_id: Optional[ObjectId] = None  # Reference to LLMCall document


class ClientType(Enum):
    DISCORD = "discord"
    DISCORD_V3 = "discord_v3"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    SHOPIFY = "shopify"
    PRINTIFY = "printify"
    CAPTIONS = "captions"
    TIKTOK = "tiktok"
    GMAIL = "gmail"
    EMAIL = "email"
    APP = "app"
    GOOGLE_CALENDAR = "google_calendar"
    MOLTBOOK = "moltbook"


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


class DiscordChannelConfig(BaseModel):
    """Config for a single Discord channel in webhook-based deployments."""

    guild_id: Optional[str] = None
    channel_id: str
    channel_name: Optional[str] = None  # cached for display
    access: Literal["read_write", "read_only"] = "read_write"
    webhook_id: Optional[str] = None
    webhook_token: Optional[str] = None


class DiscordGuildConfig(BaseModel):
    """Config for a single Discord guild in webhook-based deployments."""

    guild_id: str
    guild_name: Optional[str] = None  # cached for display
    role_id: Optional[str] = None  # Discord role ID for @mentions
    role_name: Optional[str] = None  # role name (e.g., "chatsubo")


class DeploymentSettingsDiscord(BaseModel):
    # Legacy token-based fields (for migration)
    oauth_client_id: Optional[str] = None
    oauth_url: Optional[str] = None
    channel_allowlist: Optional[List[DiscordAllowlistItem]] = None
    read_access_channels: Optional[List[DiscordAllowlistItem]] = None
    dm_user_allowlist: Optional[List[DiscordAllowlistItem]] = None

    # New webhook-based fields (multi-guild)
    guilds: Optional[List[DiscordGuildConfig]] = None

    # Legacy single-guild fields (backward compatibility)
    guild_id: Optional[str] = None
    guild_name: Optional[str] = None  # cached for display
    role_id: Optional[str] = None  # Discord role ID for @mentions
    role_name: Optional[str] = None  # role name (e.g., "chatsubo")
    channel_configs: Optional[List[DiscordChannelConfig]] = None


class DeploymentSecretsDiscord(BaseModel):
    token: Optional[str] = None  # Optional for webhook-based deployments
    application_id: Optional[str] = None


# Telegram Models
class TelegramAllowlistItem(AllowlistItem):
    pass


class DeploymentSettingsTelegram(BaseModel):
    topic_allowlist: Optional[List[TelegramAllowlistItem]] = None
    bot_username: Optional[str] = None


class DeploymentSecretsTelegram(BaseModel):
    token: str
    webhook_secret: Optional[str] = None


# Farcaster Models
class DeploymentSettingsFarcaster(BaseModel):
    username: Optional[str] = None
    webhook_id: Optional[str] = None
    enable_cast: Optional[bool] = False
    instructions: Optional[str] = None


class DeploymentSecretsFarcaster(BaseModel):
    mnemonic: Optional[str] = (
        None  # For backwards compatibility with direct mnemonic auth
    )
    signer_uuid: Optional[str] = None  # For Neynar managed signers
    neynar_webhook_secret: Optional[str] = None

    def model_post_init(self, __context):
        """Validate that either mnemonic or signer_uuid is provided"""
        if not self.mnemonic and not self.signer_uuid:
            raise ValueError("Either mnemonic or signer_uuid must be provided")
        super().model_post_init(__context)


# Twitter Models
class DeploymentSettingsTwitter(BaseModel):
    username: Optional[str] = None
    enable_tweet: Optional[bool] = False
    instructions: Optional[str] = None


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


# Instagram Models (placeholder until full implementation)
class DeploymentSettingsInstagram(BaseModel):
    username: Optional[str] = None
    ig_user_id: Optional[str] = None


class DeploymentSecretsInstagram(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    username: Optional[str] = None


class DeploymentSettingsEmail(BaseModel):
    domain_id: Optional[str] = None
    sender_email: Optional[str] = None
    autoreply_enabled: Optional[bool] = None
    reply_delay_average_minutes: Optional[int] = None
    reply_delay_variance_minutes: Optional[int] = None


class DeploymentSecretsEmail(BaseModel):
    domain: Optional[str] = None
    provider_domain_id: Optional[str] = None
    inbound_route_id: Optional[str] = None
    inbound_route_expression: Optional[str] = None
    forwarding_address: Optional[str] = None
    webhook_signing_key: Optional[str] = None


# Gmail Models
class DeploymentSettingsGmail(BaseModel):
    reply_delay_minimum: Optional[float] = None
    reply_delay_variance: Optional[float] = None
    email_instructions: Optional[str] = None
    reply_from_address: Optional[str] = None
    reply_display_name: Optional[str] = None
    last_history_id: Optional[str] = None
    watch_expiration: Optional[datetime] = None


class DeploymentSecretsGmail(BaseModel):
    service_account_info: Dict[str, Any]  # Full service account dictionary
    delegated_user: str  # User to impersonate (e.g. solienne@solienne.ai)
    token_scopes: Optional[List[str]] = None
    pubsub_subscription: Optional[str] = None
    pubsub_topic: Optional[str] = None
    watch_label_ids: Optional[List[str]] = None
    reply_alias: Optional[str] = None


# Google Calendar Models
class DeploymentSettingsGoogleCalendar(BaseModel):
    """User-configurable settings for Google Calendar deployment"""

    calendar_id: str  # The specific calendar to use (e.g., "primary" or calendar ID)
    calendar_name: Optional[str] = None  # Human-readable calendar name for UI
    allow_write: bool = False  # Whether agent can create/modify events
    allow_delete: bool = False  # Whether agent can delete events
    default_reminder_minutes: Optional[int] = (
        None  # Default reminder for created events
    )
    time_zone: Optional[str] = (
        None  # Override timezone (defaults to calendar's timezone)
    )


class DeploymentSecretsGoogleCalendar(BaseModel):
    """OAuth credentials for Google Calendar - automatically encrypted by KMS"""

    access_token: str
    refresh_token: str  # Required for long-term access
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: str
    client_secret: str
    scopes: List[str] = ["https://www.googleapis.com/auth/calendar"]
    expires_at: Optional[datetime] = None  # Token expiration time
    google_user_id: Optional[str] = None  # Google account ID (sub claim)
    google_email: str  # Email of the connected Google account


# Moltbook Models
class DeploymentSettingsMoltbook(BaseModel):
    agent_name: Optional[str] = None


class DeploymentSecretsMoltbook(BaseModel):
    api_key: str


# Combined Models
class DeploymentSecrets(BaseModel):
    discord: DeploymentSecretsDiscord | None = None
    telegram: DeploymentSecretsTelegram | None = None
    farcaster: DeploymentSecretsFarcaster | None = None
    twitter: DeploymentSecretsTwitter | None = None
    instagram: DeploymentSecretsInstagram | None = None
    shopify: DeploymentSecretsShopify | None = None
    printify: DeploymentSecretsPrintify | None = None
    captions: DeploymentSecretsCaptions | None = None
    tiktok: DeploymentSecretsTiktok | None = None
    email: DeploymentSecretsEmail | None = None
    gmail: DeploymentSecretsGmail | None = None
    google_calendar: DeploymentSecretsGoogleCalendar | None = None
    moltbook: DeploymentSecretsMoltbook | None = None


class DeploymentConfig(BaseModel):
    discord: DeploymentSettingsDiscord | None = None
    telegram: DeploymentSettingsTelegram | None = None
    farcaster: DeploymentSettingsFarcaster | None = None
    twitter: DeploymentSettingsTwitter | None = None
    instagram: DeploymentSettingsInstagram | None = None
    shopify: DeploymentSettingsShopify | None = None
    printify: DeploymentSettingsPrintify | None = None
    captions: DeploymentSettingsCaptions | None = None
    tiktok: DeploymentSettingsTiktok | None = None
    email: DeploymentSettingsEmail | None = None
    gmail: DeploymentSettingsGmail | None = None
    google_calendar: DeploymentSettingsGoogleCalendar | None = None
    moltbook: DeploymentSettingsMoltbook | None = None


@Collection("deployments2")
class Deployment(Document):
    agent: ObjectId
    user: ObjectId
    platform: ClientType
    valid: Optional[bool] = None
    local: Optional[bool] = None
    secrets: Optional[DeploymentSecrets]
    config: Optional[DeploymentConfig]
    encrypted: Optional[bool] = None  # Track if secrets are encrypted

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
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Encrypt secrets before saving to MongoDB"""
        from eve.utils.data_utils import serialize_json
        from eve.utils.kms_encryption import (
            encrypt_deployment_secrets,
            get_kms_encryption,
        )

        kms = get_kms_encryption()

        if kms.enabled and "secrets" in schema and schema["secrets"]:
            secrets_dict = serialize_json(schema["secrets"])
            encrypted_secrets = encrypt_deployment_secrets(secrets_dict)
            schema["secrets"] = encrypted_secrets
            schema["encrypted"] = True

        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Decrypt secrets after loading from MongoDB"""
        from eve.utils.kms_encryption import (
            decrypt_deployment_secrets,
            get_kms_encryption,
        )

        get_kms_encryption()

        if "secrets" in schema and schema["secrets"]:
            # Check if secrets are encrypted
            if (
                isinstance(schema["secrets"], dict)
                and "encryption_metadata" in schema["secrets"]
            ):
                # Decrypt the secrets
                try:
                    decrypted_secrets = decrypt_deployment_secrets(schema["secrets"])
                    schema["secrets"] = decrypted_secrets
                except Exception as e:
                    logger.error(f"Failed to decrypt deployment secrets: {e}")
                    # Keep encrypted data in schema, will fail validation
                    # This is better than losing data
            # else: secrets are not encrypted (backward compatibility)

        return schema

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent", 1), ("platform", 1)], unique=True)

    def get_allowed_channels(self):
        """Get allowed channels for the deployment"""
        if self.platform == ClientType.DISCORD:
            return self.config.discord.channel_allowlist
        elif self.platform == ClientType.DISCORD_V3:
            # Convert channel_configs to channel_allowlist format for compatibility
            if not self.config.discord.channel_configs:
                return []
            # Return list of objects with .id and .note attributes
            from types import SimpleNamespace

            return [
                SimpleNamespace(id=ch.channel_id, note=ch.channel_name)
                for ch in self.config.discord.channel_configs
            ]
        elif self.platform == ClientType.TELEGRAM:
            return self.config.telegram.topic_allowlist
        return []

    def get_dm_allowed_users(self):
        """Get allowed DM users for the deployment"""
        if self.platform in [ClientType.DISCORD, ClientType.DISCORD_V3]:
            if self.config and self.config.discord:
                return self.config.discord.dm_user_allowlist or []
        return []


@Collection("email_domains")
class EmailDomain(Document):
    agent: ObjectId
    user: ObjectId
    provider: Literal["mailgun", "sendgrid", "twilio"] = "mailgun"
    domain: str
    provider_domain_id: Optional[str] = None
    inbound_route_id: Optional[str] = None
    inbound_route_expression: Optional[str] = None
    forwarding_address: Optional[str] = None
    webhook_signing_key: Optional[str] = None
    status: Literal["pending", "verified", "failed"] = "pending"
    lastSyncedAt: Optional[datetime] = None
    dnsRecords: Optional[List[Dict[str, Any]]] = None
    verificationErrors: Optional[List[str]] = None
    deployment: Optional[ObjectId] = None

    @classmethod
    def find_by_domain(cls, domain: str) -> Optional["EmailDomain"]:
        collection = cls.get_collection()
        doc = collection.find_one({"domain": domain.lower()})
        return cls(**doc) if doc else None


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
