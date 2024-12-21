import os
import json
import magic
import copy
from bson import ObjectId
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.json_schema import SkipJsonSchema
from typing import List, Optional, Dict, Any, Literal, Union

from .mongo import Document, Collection
from .eden_utils import download_file, image_to_base64, prepare_result, dump_json


class ChatMessage(BaseModel):
    id: ObjectId = Field(default_factory=ObjectId)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: Literal["user", "assistant"]
    reply_to: Optional[ObjectId] = None
    reactions: Optional[Dict[str, List[ObjectId]]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def react(self, user: ObjectId, reaction: str):
        if reaction not in self.reactions:
            self.reactions[reaction] = []
        self.reactions[reaction].append(user)
    

class UserMessage(ChatMessage):
    role: Literal["user"] = "user"
    name: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    attachments: Optional[List[str]] = []

    def _get_content(self, schema, truncate_images=False):
        """Assemble user message content block"""

        # start with original message content
        content = self.content or ""

        # let claude see names
        if self.name and schema == "anthropic":
            content = f"<User>{self.name}</User>\n\n{content}"

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
                    attachment_files.append(attachment_file)
                    mime_type = magic.from_file(attachment_file, mime=True)
                    if "video" in mime_type:
                        attachment_lines.append(
                            f"* {attachment} (The asset is a video, the corresponding image attachment is its first frame.)"
                        )
                    else:
                        attachment_lines.append(f"* {attachment}")
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
                            ),
                        },
                    }
                    for file_path in attachment_files
                ]
            elif schema == "openai":
                block = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"""data:image/jpeg;base64,{image_to_base64(
                            file_path, 
                            max_size=512, 
                            quality=95, 
                            truncate=truncate_images
                            
                        )}"""
                        },
                    }
                    for file_path in attachment_files
                ]

            if content:
                block.extend([{"type": "text", "text": content.strip()}])

            content = block

        return content

    def anthropic_schema(self, truncate_images=False):
        content = self._get_content(
            "anthropic", 
            truncate_images=truncate_images
        )
        return [{
            "role": "user",
            "content": content
        }] if content else []

    def openai_schema(self, truncate_images=False):
        return [
            {
                "role": "user",
                "content": self._get_content("openai", truncate_images=truncate_images),
                **({"name": self.name} if self.name else {}),
            }
        ]


class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]

    db: SkipJsonSchema[str]
    task: Optional[ObjectId] = None
    status: Optional[
        Literal["pending", "running", "completed", "failed", "cancelled"]
    ] = None
    result: Optional[List[Dict[str, Any]]] = None
    reactions: Optional[Dict[str, List[ObjectId]]] = None
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_result(self, schema, truncate_images=False):
        result = {"status": self.status}

        if self.status == "completed":
            result["result"] = prepare_result(self.result, db=self.db)
            outputs = [
                o.get("url")
                for r in result.get("result", [])
                for o in r.get("output", [])
            ]
            outputs = [
                o
                for o in outputs
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
                    for url in outputs
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
                                "url": f"""data:image/jpeg;base64,{image_to_base64(
                                file_path, 
                                max_size=512, 
                                quality=95, 
                                truncate=truncate_images
                            )}"""
                            },
                        }
                        for file_path in files
                    ]

                if image_block:
                    content = "Tool results follow. The attached images match the URLs in the order they appear below: "
                    # content += json.dumps(result["result"])
                    content += dump_json(result["result"])
                    text_block = [{"type": "text", "text": content}]
                    result = text_block + image_block
                else:
                    result = dump_json(result)

            except Exception as e:
                print("Error injecting image results:", e)
                result = dump_json(result)

        elif self.status == "failed":
            result["error"] = self.error
            result = dump_json(result)

        else:
            result = dump_json(result)

        return result

    def react(self, user: ObjectId, reaction: str):
        pass

    @staticmethod
    def from_openai(tool_call, db="STAGE"):
        return ToolCall(
            id=tool_call.id,
            tool=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
            db=db,
        )

    @staticmethod
    def from_anthropic(tool_call, db="STAGE"):
        return ToolCall(
            id=tool_call.id, tool=tool_call.name, args=tool_call.input, db=db
        )

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

    def anthropic_result_schema(self, truncate_images=False):
        # todo: add "is_error": true
        return {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": self.get_result(
                schema="anthropic", truncate_images=truncate_images
            ),
        }

    def openai_result_schema(self, truncate_images=False):
        return {
            "role": "tool",
            "name": self.tool,
            "content": self.get_result(
                schema="openai", truncate_images=truncate_images
            ),
            "tool_call_id": self.id,
        }


class AssistantMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    agent_id: Optional[ObjectId] = None
    thought: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = []

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
        schema = [{
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": self.content
                }
            ] if self.content else [],
        }]
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

    
@Collection("threads3")
class Thread(Document):
    key: Optional[str] = None
    title: Optional[str] = None
    agent: Optional[ObjectId] = None
    user: Optional[ObjectId] = None
    messages: List[Union[UserMessage, AssistantMessage]] = Field(default_factory=list)
    active: List[ObjectId] = Field(default_factory=list)

    @classmethod
    def load(cls, key, agent=None, user=None, create_if_missing=False, db="STAGE"):
        filter = {"key": key}
        if agent:
            filter["agent"] = agent
        if user:
            filter["user"] = user        
        thread = cls.get_collection(db).find_one(filter)
        if thread:
            thread = Thread(db=db, **thread)
        else:
            if create_if_missing:
                thread = cls(db=db, key=key, agent=agent, user=user)
                thread.save()
            else:
                raise Exception(f"Thread {key} with agent {agent} not found in {cls.collection_name}:{db}")        
        return thread

    def update_tool_call(self, message_id, tool_call_index, updates):
        # Update the in-memory object
        message = next(m for m in self.messages if m.id == message_id)
        for key, value in updates.items():
            setattr(message.tool_calls[tool_call_index], key, value)
        # Update the database
        updates = {
            f"messages.$.tool_calls.{tool_call_index}.{k}": v
            for k, v in updates.items()
        }
        self.set_against_filter(updates, filter={"messages.id": message_id})

    def get_messages(self, filters=None):
        # filter by time, number, or prompt
        # if reply to inside messages, mark it
        # if reply to by old message, include context leading up to it
        # self.reload()
        messages = copy.deepcopy(self.messages[-15:])
        # hack to remove any spurious assistant messages at end
        # todo: should try to actually fix this bug
        while messages and messages[-1].role == "assistant":
            messages.pop()
        return messages