import openai
import anthropic
import instructor

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from .agent import Agent, refresh_agent
from .thread import UserMessage, Thread
from ..tool import TOOL_CATEGORIES
from ..models import Model
from ..mongo import get_collection
from ..eden_utils import dump_json, load_template
from .llm import async_prompt


knowledge_think_template = load_template("knowledge_think") 
knowledge_reply_template = load_template("knowledge_reply")
thought_template = load_template("thought")
tools_template = load_template("tools")




class SearchResult(BaseModel):
    """A matching result from the database search."""
    
    id: str = Field(..., description="The MongoDB ID of the result")
    name: str = Field(..., description="The name/title of the result")
    description: str = Field(..., description="A brief description of the result")
    relevance: str = Field(
        ..., 
        description="A brief explanation of why this result matches the search query"
    )

class SearchResults(BaseModel):
    """Results from searching the database."""
    
    results: List[SearchResult] = Field(
        ...,
        description="The matching results, ordered by relevance. Include only truly relevant results."
    )


search_template = """<mongodb_documents>
{{documents}}
</mongodb_documents>
<query>
{{query}}
</query>
<task>
Return a list of matching documents to the query.
</task>"""

agent_template = """<document>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <username>{{username}}</username>
  <description>{{description}}</description>
  <knowledge_description>{{knowledge_description}}</knowledge_description>
  <persona>{{persona[:750]}}</persona>
  <created_at>{{createdAt}}</created_at>
</document>"""

model_template = """<document>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <lora_model>{{lora_model}}</lora_model>
  <lora_trigger_text>{{lora_trigger_text}}</lora_trigger_text>
  <created_at>{{createdAt}}</created_at>
</document>"""

from jinja2 import Template
model_template = Template(model_template)
agent_template = Template(agent_template)
search_template = Template(search_template)

async def search_mongo(type: Literal["model", "agent"], query: str):
    """Search MongoDB for models or agents matching the query."""
    
    docs = []
    id_map = {}
    counter = 1
    
    if type == "model":
        collection = get_collection(Model.get_collection_name())
        for doc in collection.find({"base_model": "flux-dev", "public": True, "deleted": {"$ne": True}}):
            # Map the real ID to a counter
            id_map[counter] = str(doc["_id"])
            doc["_id"] = counter
            counter += 1
            docs.append(model_template.render(doc))

    elif type == "agent":
        collection = get_collection(Agent.collection_name)
        for doc in collection.find({"type": "agent", "public": True, "deleted": {"$ne": True}}):
            # Map the real ID to a counter
            id_map[counter] = str(doc["_id"])
            doc["_id"] = counter
            counter += 1
            docs.append(agent_template.render(doc))

    # Create context for LLM
    context = search_template.render(
        documents="\n".join(docs), 
        query=query
    )

    # Make LLM call
    system_message = f"""You are a search assistant that helps find relevant {type}s based on natural language queries. 
    Analyze the provided items and return only the most relevant matches for the query.
    Be selective - only return items that truly match the query's intent."""

    prompt = f"""<{type}s>
{context}
</{type}s>
<query>
"{query}"
</query>
<task>
Analyze these items and return only the ones that are truly relevant to this search query. 
Explain why each result matches the query criteria.
</task>"""

    print(context)



    # raise Exception("stop")

    client = instructor.from_openai(openai.AsyncOpenAI())
    results = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=SearchResults,
    )

    # Map the simple IDs back to real MongoDB IDs
    for result in results.results:
        result.id = id_map[int(result.id)]

    return results.results


async def async_think(
    agent: Agent,
    thread: Thread,
    user_message: UserMessage,
    force_reply: bool = True,
):
    # intention_description = "Response class to the last user message. Ignore if irrelevant, reply if relevant and you intend to say something."

    # if agent.reply_criteria:
    #     intention_description += (
    #         f"\nAdditional criteria for replying spontaneously: {agent.reply_criteria}"
    #     )

    class ChatThought(BaseModel):
        """A response to a chat message."""

        intention: Literal["ignore", "reply"] = Field(
            ..., description="Ignore if last message is irrelevant, reply if relevant or criteria met."
        )
        thought: str = Field(
            ...,
            description="A very brief thought about what relevance, if any, the last user message has to you, and a justification of your intention.",
        )
        tools: Optional[Literal[tuple(TOOL_CATEGORIES.keys())]] = Field(
            ...,
            description=f"Which tools to include in reply context",
        )
        recall_knowledge: bool = Field(
            ...,
            description="Whether to recall, refer to, or consult your knowledge base.",
        )

    # generate text blob of chat history
    chat = ""
    messages = thread.get_messages(25)
    for msg in messages:
        content = msg.content
        if msg.role == "user":
            if msg.attachments:
                content += f" (attachments: {msg.attachments})"
            name = "You" if msg.name == agent.name else msg.name or "User"
        elif msg.role == "assistant":
            name = agent.name
            for tc in msg.tool_calls:
                args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
                tc_result = dump_json(tc.result, exclude="blurhash")
                content += f"\n -> {tc.tool}({args}) -> {tc_result}"
        time_str = msg.createdAt.strftime("%H:%M")
        chat += f"<{name} {time_str}> {content}\n"

    # user message text
    content = user_message.content
    if user_message.attachments:
        content += f" (attachments: {user_message.attachments})"
    time_str = user_message.createdAt.strftime("%H:%M")
    message = f"<{user_message.name} {time_str}> {content}"

    if agent.knowledge:
        # if knowledge is requested but no knowledge description, create it now
        if not agent.knowledge_description:
            await refresh_agent(agent)
            agent.reload()

        knowledge_description = f"Summary: {agent.knowledge_description.summary}. Recall if: {agent.knowledge_description.retrieval_criteria}"
        knowledge_description = knowledge_think_template.render(
            knowledge_description=knowledge_description
        )
    else:
        knowledge_description = ""

    if agent.reply_criteria:
        reply_criteria = f"Note: You should additionally set reply to true if any of the follorwing criteria are met: {agent.reply_criteria}"
    else:
        reply_criteria = ""

    tool_descriptions = "\n".join([f"{k}: {v}" for k, v in TOOL_CATEGORIES.items()])
    tools_description = tools_template.render(
        tool_categories=tool_descriptions
    )

    prompt = thought_template.render(
        name=agent.name,
        chat=chat,
        tools_description=tools_description,
        knowledge_description=knowledge_description,
        message=message,
        reply_criteria=reply_criteria,
    )

    thought = await async_prompt(
        [UserMessage(content=prompt)],
        system_message=f"You analyze the chat on behalf of {agent.name} and generate a thought.",
        model="gpt-4o-mini",
        response_model=ChatThought,
    )

    if force_reply:
        thought.intention = "reply"

    return thought

