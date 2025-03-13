import asyncio
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from ..eden_utils import dump_json
from .llm import async_prompt
from .agent import Agent
# from .thread import UserMessage
from .message import ChatMessage, UserMessage
from .session import Session


agent_template = Template("""<Agent>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <username>{{username}}</username>
  {% if description is not none %}<description>{{description}}</description>{% endif %}
  {% if knowledge_description is not none %}<knowledge_description>{{knowledge_description}}</knowledge_description>{% endif %}
  {% if persona is not none %}<persona>{{persona}}</persona>{% endif %}
  <created_at>{{createdAt}}</created_at>
</Agent>""")


dispatcher_template = Template("""
<Summary>
You are the Dispatcher. You orchestrate this multi-agent scenario by deciding:
- Which agent(s) should speak next (if any)
- The current state of the scenario (progress, unresolved issues, next steps)
- Whether the scenario is complete
You do not speak or act on your own; you remain invisible, only orchestrating.
</Summary>

<Rules>
- Maintain a fluid, coherent conversation that moves toward the scenario's end goal.
- Avoid infinite loops or repetitive commentary.
- Agents should only speak if they have something meaningful or new to contribute.
- Stop the scenario when the end condition is truly met (goal is reached, or conversation logically concludes).
</Rules>

<Agents>
{{agents}}
</Agents>

<Scenario>
This is the original premise of the scenario:
                               
{{scenario}}
</Scenario>

<Current>
This is the current situation:
                               
{{current}}
</Current>

<ChatLog>
{{chat_log}}
</ChatLog>

<NewMessage>
{{latest_message}}
</NewMessage>

<Task>
1. Decide which agents, if any, should speak next in response to the new message or the current situation.
2. Summarize the current state of the scenario, including progress towards the goal and remaining steps.
3. Determine if the scenario has reached its end condition. If yes, set end_condition to true.
4. Respond with the following:
 - speakers: optional array of agents who should speak, in order of priority
 - state: a description of the current state of the scenario
 - end_condition: true if the scenario is complete, false otherwise
</Task>""")



async def async_run_dispatcher(
    session: Session,
    message: ChatMessage,
):



    raise Exception("stop here!")

    agents = [Agent.from_mongo(a) for a in session.agents]
    agent_names = [a.username for a in agents]
    agents_text = "\n".join([agent_template.render(a) for a in agents])

    # generate text blob of chat history
    chat = session.get_chat_log(25)
    
    print(chat)

    latest_user_message = message.content

    prompt = dispatcher_template.render(
        agents=agents_text,
        scenario="session.scenario",
        current="session.current",
        chat_log=chat,
        latest_message=latest_user_message,
    )

    print("--------------------------------")
    print(prompt)
    print("--------------------------------")

    

    class DispatcherThought(BaseModel):
        """A thought about how to respond to the last message in the chat."""

        speakers: Optional[List[Literal[*agent_names]]] = Field(
            None,
            description="An optional list of agents that the dispatcher encourages to spontaneously speak or respond to the last message.",
        )
        state: str = Field(
            ...,
            description="A description about the current state of the scenario, including a summary of progress towards the goal, and what remains to be done.",
        )
        end_condition: bool = Field(
            ...,
            description="Only true if scenario is completed.",
        )

    thought = await async_prompt(
        [UserMessage(content=prompt)],
        system_message=f"You are a dispatcher who guides the conversation towards the end goal.",
        model="gpt-4o-mini",
        response_model=DispatcherThought,
    )

    print(thought)

    return thought



def dispatch(session: Session, message: ChatMessage):
    return asyncio.run(async_run_dispatcher(session, message))