from __future__ import annotations
import os
import json
import magic
from bson import ObjectId
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional, Dict, Any, Literal, Union


from ..eden_utils import download_file, image_to_base64, prepare_result, dump_json
from ..mongo import Document, Collection, get_collection
from ..user import User
from ..api.api_requests import UpdateConfig




class Channel(BaseModel):
    type: Literal["eden", "discord", "telegram", "twitter"]
    key: str
    # dm: Optional[bool] = False
    # agents: Optional[List[ObjectId]] = []
    
    # def get_messages(self, limit: int = 25):
    #     messages = get_collection("messages")
    #     messages = messages.find({"channel.key": self.key}).sort("createdAt", -1)
    #     if limit:
    #         messages = messages.limit(limit)
    #     return [ChatMessage(**msg) for msg in messages]


@Collection("messages")
class ChatMessage(Document):
    # channel: Optional[Channel] = None
    session: ObjectId
    
    reply_to: Optional[ObjectId] = None
    # hidden: Optional[bool] = False
    sender: ObjectId = None
    reactions: Optional[Dict[str, List[ObjectId]]] = {}
    content: Optional[str] = None
    #metadata: Optional[Dict[str, Any]] = {}
    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def react(self, user: ObjectId, reaction: str):
        if reaction not in self.reactions:
            self.reactions[reaction] = []
        self.reactions[reaction].append(user)

    def user_message(self):
        return UserMessage(**self.model_dump())

    def assistant_message(self):
        return AssistantMessage(**self.model_dump())
    
    def to_thread(self, assistant_id: ObjectId):
        if self.sender == assistant_id:
            return self.assistant_message()
        else:
            return self.user_message()
    

class UserMessage(ChatMessage):
    role: Literal["user"] = Field(default="user")
    name: Optional[str] = "NamePlaceholder"

    def _get_content(self, schema, truncate_images=False):
        """Assemble user message content block"""

        # start with original message content
        content = self.content or ""

        # Let claude see names
        if self.name and schema == "anthropic" and content:
            content = f"<User>{self.name}</User>\n\n{content}"

        # If this message contains tool calls, extract media from them and add to attachments
        if self.tool_calls:
            tool_attachments = []
            for tool_call in self.tool_calls:
                print("HERE IS THE TOOL CALL")
                print(tool_call)
                result = tool_call.result.copy() #prepare_result(tool_call.result)
                print("HERE IS THE RESULT")
                print(result)
                if result["status"] == "completed":
                    for r in prepare_result(result["result"]):
                        for o in r.get("output", []):
                            if o.get("url"):
                                tool_attachments.append(o["url"])
            if tool_attachments:
                self.attachments.extend(tool_attachments)

        if self.attachments:
            # append attachments info (url and type) to content
            attachment_lines = []
            attachment_files = []
            attachment_errors = []
            for attachment in self.attachments:
                try:
                    attachment_file = download_file(
                        attachment,
                        os.path.join(
                            "/tmp/eden_file_cache/", attachment.split("/")[-1]
                        ),
                        overwrite=False,
                    )
                    mime_type = magic.from_file(attachment_file, mime=True)
                    if "video" in mime_type:
                        attachment_lines.append(
                            f"* {attachment} (The asset is a video, the corresponding image attachment is its first frame.)"
                        )
                        attachment_files.append(attachment_file)
                    elif "image" in mime_type:
                        attachment_lines.append(f"* {attachment}")
                        attachment_files.append(attachment_file)
                    else:
                        attachment_errors.append(
                            f"* {attachment}: (Mime type: {mime_type})"
                        )
                except Exception as e:
                    attachment_errors.append(f"* {attachment}: {str(e)}")

            attachments = ""
            if attachment_lines:
                attachments += "The attached images correspond to the following urls:\n"
                attachments += "\n".join(attachment_lines)
            if attachment_errors:
                attachments += "The following files failed to attach:\n"
                attachments += "\n".join(attachment_errors)
            attachments = f"<attachments>\n{attachments}\n</attachments>"
            content += f"\n{attachments}"

            # add image blocks
            if schema == "anthropic":
                block = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_base64(
                                file_path,
                                max_size=512,
                                quality=95,
                                truncate=truncate_images,
                            )
                        }
                    }
                    for file_path in attachment_files
                ]
            elif schema == "openai":
                block = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{
                                image_to_base64(
                                    file_path, 
                                    max_size=512, 
                                    quality=95, 
                                    truncate=truncate_images
                                )
                            }"
                        }
                    }
                    for file_path in attachment_files
                ]

            if content:
                block.extend([{"type": "text", "text": content.strip()}])

            content = block

        return content

    def anthropic_schema(self, truncate_images=False):
        content = self._get_content("anthropic", truncate_images=truncate_images)
        return [{"role": "user", "content": content}] if content else []

    def openai_schema(self, truncate_images=False):
        return [
            {
                "role": "user",
                "content": self._get_content("openai", truncate_images=truncate_images),
                **({"name": self.name} if self.name else {}),
            }
        ]

class AssistantMessage(ChatMessage):
    role: Literal["assistant"] = Field(default="assistant")

    def openai_schema(self, truncate_images=False):
        schema = [
            {
                "role": "assistant",
                "content": self.content,
                "function_call": None,
                "tool_calls": None,
            }
        ]
        if self.tool_calls:
            schema[0]["tool_calls"] = [t.openai_call_schema() for t in self.tool_calls]
            schema.extend(
                [
                    t.openai_result_schema(truncate_images=truncate_images)
                    for t in self.tool_calls
                ]
            )
        return schema

    def anthropic_schema(self, truncate_images=False):
        if not self.content and not self.tool_calls:
            return []
        schema = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.content}]
                if self.content
                else [],
            }
        ]
        if self.tool_calls:
            schema[0]["content"].extend(
                [t.anthropic_call_schema() for t in self.tool_calls]
            )
            schema.append(
                {
                    "role": "user",
                    "content": [
                        t.anthropic_result_schema(truncate_images=truncate_images)
                        for t in self.tool_calls
                    ],
                }
            )
        return schema


class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]

    task: Optional[ObjectId] = None
    result: Optional[Dict[str, Any]] = None
    reactions: Optional[Dict[str, List[ObjectId]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def get_result(self, schema, truncate_images=False):
        result = self.result.copy()

        if result["status"] == "completed":
            result["result"] = prepare_result(result["result"])

        print(result)

        file_outputs = [
            o["url"]
            for r in result.get("result", [])
            # for r in result
            for o in r.get("output", [])
            if isinstance(o, dict) and o.get("url")
        ]
        file_outputs = [
            o
            for o in file_outputs
            if o and o.endswith((".jpg", ".png", ".webp", ".mp4", ".webm"))
        ]
        try:
            if schema == "openai":
                raise ValueError(
                    "OpenAI does not support image outputs in tool messages :("
                )

            files = [
                download_file(
                    url,
                    os.path.join("/tmp/eden_file_cache/", url.split("/")[-1]),
                    overwrite=False,
                )
                for url in file_outputs
            ]

            if schema == "anthropic":
                image_block = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_base64(
                                file_path,
                                max_size=512,
                                quality=95,
                                truncate=truncate_images,
                            ),
                        },
                    }
                    for file_path in files
                ]
            elif schema == "openai":
                image_block = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{
                                image_to_base64(
                                    file_path, 
                                    max_size=512, 
                                    quality=95, 
                                    truncate=truncate_images
                                )
                            }"
                        },
                    }
                    for file_path in files
                ]

            if image_block:
                content = "Tool results follow. The attached images match the URLs in the order they appear below: "
                content += dump_json(result, exclude="blurhash")
                text_block = [{"type": "text", "text": content}]
                result = text_block + image_block
            else:
                result = dump_json(result, exclude="blurhash")

        except Exception as e:
            # print("Warning: Can not inject image results:", e)
            result = dump_json(result, exclude="blurhash")

        return result

    def react(self, user: ObjectId, reaction: str):
        pass

    @staticmethod
    def from_openai(tool_call):
        return ToolCall(
            id=tool_call.id,
            tool=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
        )

    @staticmethod
    def from_anthropic(tool_call):
        return ToolCall(
            id=tool_call.id, 
            tool=tool_call.name, 
            args=tool_call.input
        )

    def openai_call_schema(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.tool, 
                "arguments": json.dumps(self.args)
            },
        }

    def anthropic_call_schema(self):
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.tool,
            "input": self.args,
        }

    def anthropic_result_schema(self, truncate_images=False):
        # todo: add "is_error": true
        return {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": self.get_result(
                schema="anthropic", 
                truncate_images=truncate_images
            ),
        }

    def openai_result_schema(self, truncate_images=False):
        return {
            "role": "tool",
            "name": self.tool,
            "content": self.get_result(
                schema="openai", 
                truncate_images=truncate_images
            ),
            "tool_call_id": self.id,
        }


@Collection("tests")
class Session(Document):
	user: ObjectId
	channel: Optional[Channel] = None
	title: str
	agents: List[ObjectId] = Field(default_factory=list)
	scenario: Optional[str] = None
	current: Optional[ObjectId] = None
	# messages: List[ChatMessage] = Field(default_factory=list)
	budget: Optional[float] = None
	spent: Optional[float] = 0
	cursor: Optional[ObjectId] = None


def get_chat_log(
    messages: List[ChatMessage], 
    you_id: Optional[ObjectId] = None, 
) -> str:
    senders = User.find({"_id": {"$in": [message.sender for message in messages]}})
    senders = {sender.id: sender.username for sender in senders}
    
    chat = ""
    for message in messages:
        content = message.content
        name = senders[message.sender]
        if message.attachments:
            content += f" (attachments: {message.attachments})"
        for tc in message.tool_calls:
            args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
            tc_result = dump_json(tc.result, exclude="blurhash")
            content += f"\n -> {tc.tool}({args}) -> {tc_result}"
        time_str = message.createdAt.strftime("%H:%M")
        if you_id and message.sender == you_id:
            chat += f"<{name} (You) {time_str}> {content}\n"
        else:
            chat += f"<{name} {time_str}> {content}\n"

    return chat
