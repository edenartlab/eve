from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from jinja2 import Template
from typing import List, Literal, Optional

from eve.agent.dispatch import async_dispatch
from eve.eden_utils import dump_json

from ..mongo import Document, Collection
from ..api.api_requests import UpdateConfig
from .message import ChatMessage, Channel
from bson.objectid import ObjectId




@Collection("tests")
class Session(Document):
	name: str
	channel: Optional[Channel] = None
	agents: List[ObjectId] = Field(default_factory=list)
	messages: List[ChatMessage] = Field(default_factory=list)

	def get_chat_log(self, limit: int = 25) -> str:
		chat = ""
		for message in self.messages[-limit:]:
			content = message.content
			name = message.name
			if message.attachments:
				content += f" (attachments: {message.attachments})"
			for tc in message.tool_calls:
				args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
				tc_result = dump_json(tc.result, exclude="blurhash")
				content += f"\n -> {tc.tool}({args}) -> {tc_result}"
			time_str = message.createdAt.strftime("%H:%M")
			chat += f"<{name} {time_str}> {content}\n"

		return chat
