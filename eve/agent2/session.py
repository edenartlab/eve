from bson.objectid import ObjectId
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from jinja2 import Template
from typing import List, Literal, Optional

from ..eden_utils import dump_json
from ..mongo import Document, Collection
from ..user import User
from ..api.api_requests import UpdateConfig
from .message import ChatMessage, Channel


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


	def get_chat_log(self, limit: int = 25) -> str:
		chat = ""
		
		
		messages = ChatMessage.find({"session": self.id}, sort="createdAt", limit=25)
		senders = User.find({"_id": {"$in": [message.sender for message in messages]}})
		senders = {sender.id: sender.username for sender in senders}
		

		for message in messages[-limit:]:
			content = message.content
			name = senders[message.sender]
			if message.attachments:
				content += f" (attachments: {message.attachments})"
			for tc in message.tool_calls:
				args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
				tc_result = dump_json(tc.result, exclude="blurhash")
				content += f"\n -> {tc.tool}({args}) -> {tc_result}"
			time_str = message.createdAt.strftime("%H:%M")
			chat += f"<{name} {time_str}> {content}\n"


		print("------------33--------------------")
		print(chat)
		print("--------------44------------------")

		return chat
