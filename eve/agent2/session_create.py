from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List, Literal

from ..mongo import get_collection
from ..user import User
from .message import UserMessage
from .session import Session, Channel
from .llm import async_prompt
from .agent import Agent



agent_template = Template("""<Agent>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <username>{{username}}</username>
  {% if description is not none %}<description>{{description}}</description>{% endif %}
  {% if knowledge_description is not none %}<knowledge_description>{{knowledge_description}}</knowledge_description>{% endif %}
  {% if persona is not none %}<persona>{{persona}}</persona>{% endif %}
  <created_at>{{createdAt}}</created_at>
</Agent>""")


session_creation_template = Template("""
<Summary>
You are given a chat conversation, and a new message which is requesting to create a new multi-agent scenario. Your goal is to extract the premise of the scenario, agents involved, and a budget in manna for the scenario.
</Summary>

<EligibleAgents>
Only the following agents are allowed to be involved in this scenario. Do not include any other agents in the scenario.
                                     
{{agents}}
</EligibleAgents>

<ChatLog>
{{chat_log}}
</ChatLog>

<NewMessage>
{{latest_message}}
</NewMessage>

<Task>
1. Decide which agents are involved in this scenario. These are the primary actors in the scenario.
2. Summarize the premise of the scenario. Be specific about what the goal or desired outcome of the scenario is and a way to determine when the scenario is complete.
3. Allocate a budget in manna for the scenario, using the following guidelines:
 - 1 manna is roughly equivalent to 1 LLM call or text-to-image generation, while video or model generation is more expensive, getting to the 50-100 range.
 - Most projects should use anywhere from 100-500 manna, but a more ambitious project may cost 1000-2000 manna. Do not go above 2500 manna.
 - If the user requests a specific amount of manna, just do that.
</Task>""")


async def async_create_session(
    user: User,
    channel: Channel,
    prompt: str
):
    # get the last 25 messages of the chat
    messages = channel.get_messages(limit=25)
    chat = "\n".join([f"({m.createdAt}): {m.content}" for m in messages])

    # get all agents that are listening to this channel
    deployments = get_collection("deployments").find({f"config.{channel.type}.channel_allowlist.id": channel.key})
    eligible_agents = Agent.find({"_id": {"$in": [dep["agent"] for dep in deployments]}})
    agent_names = [a.username for a in eligible_agents]
    agents_text = "\n".join([agent_template.render(a) for a in eligible_agents])

    # get the last user message
    latest_user_message = messages[-1].content

    session_creation_prompt = session_creation_template.render(
        agents=agents_text,
        chat_log=chat,
        latest_message=latest_user_message,
    )

    print("--------------------------------")
    print(session_creation_prompt)
    print("--------------------------------")

    class NewSession(BaseModel):
        """A Session is a multi-agent chat involving a group of agents, a scenario, and a budget."""

        agents: List[Literal[*agent_names]] = Field(
            ...,
            description="A list of agents that are involved in or requested to be in the scenario.",
        )
        title: str = Field(
            ...,
            description="A title for the scenario, 5-10 words.",
        )
        scenario: str = Field(
            ...,
            description="A description of the scenario, including the premise, activity, goal or desired outcome, and a way to determine when the scenario is complete.",
        )
        budget: int = Field(
            ...,
            description="The amount of manna to allocate for the scenario.",
        )

    new_session = await async_prompt(
        messages=[UserMessage(content=prompt)],
        system_message="You are a scenario creator who creates new multi-agent scenarios.",
        model="gpt-4o-mini",
        response_model=NewSession,
    )

    selected_agents = [
        a.id for a in eligible_agents if a.username in new_session.agents
    ]

    new_session = Session(
        user=user.id,
        channel=channel,
        title=new_session.title,
        agents=selected_agents,
        scenario=new_session.scenario,
        budget=new_session.budget,
    )
    new_session.save()
    print("!!!")

    return new_session
