import asyncio
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from ..eden_utils import dump_json
from .llm import async_prompt
from .agent import Agent
# from .thread import UserMessage
from .message import ChatMessage, UserMessage, Session, get_chat_log


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
You are the Dispatcher. You orchestrate this multi-agent scenario by deciding which agent should speak next and giving them a hint about the overall status of the scenario they are in and what they should do to move closer towards the final goal. You do not speak or act on your own; you remain invisible, only orchestrating.
</Summary>

<Rules>
- Maintain a fluid, coherent conversation that moves toward the scenario's end goal.
- If you have two agents, they should alternate speaking. Don't ever let the same agent speak twice in a row.
- Avoid infinite loops or repetitive commentary.
- Agents should only speak if they have something meaningful or new to contribute.
- Stop the scenario when the end condition is truly met (goal is reached, or conversation logically concludes).
</Rules>

<Agents>
{{agents}}
</Agents>
                               
{{scenario}}
{{current}}
{{processed_chat}}
{{new_chat}}
{{manna_info}}

<Task>
1. Decide which agents, if any, should speak next in response to the new message or the current situation.
2. Summarize the current state of the scenario, including progress towards the goal and remaining steps.
3. Determine if the scenario has reached its end condition. If yes, set end_condition to true.
4. Respond with the following:
 - speakers: optional array of agents who should speak, in order of priority
 - state: a description of the current state of the scenario
 - end_condition: true if the scenario is complete, false otherwise
</Task>""")

scenario_template = Template("""
<Scenario>
{{scenario}}
</Scenario>
""")

current_template = Template("""
<Current>
This is a summary of the current state of the scenario with respect to what's happened, what remains to be done.

{{current}}
</Current>
""")

processed_chat_template = Template("""
<ChatLog>
This is the log of the messages that have been previously considered already.

{{chat_log}}
</ChatLog>
""")

new_chat_template = Template("""
<NewMessages>
The following messages have been received since the last message was processed:

{{new_chat}}
</NewMessages>
""")

manna_info_template = Template("""
<Manna>
All actions cost manna. 1 manna is roughly equivalent to 1 chat message or one text-to-image generation, while video or model generation is more expensive, getting to the 50-100 range.

Manna budgeted for this scenario: {{ manna }}
Manna spent so far: {{ manna_spent }}
Manna left: {{ manna - manna_spent }}
                               
Make sure you are on track to finish the scenario before you run out of manna. If you are starting to run low, gently guide the agents towards concluding the scenario.
</Manna>
""")



async def async_run_dispatcher(
    session: Session,
):
    # get agents
    agents = Agent.find({"_id": {"$in": session.agents}})
    agent_names = [a.username for a in agents]
    agents_text = "\n".join([agent_template.render(a) for a in agents])

    # convert chat history to text
    messages = ChatMessage.find({"session": session.id}, sort="createdAt", limit=50)
    last_processed_message = next((m for m in messages if m.id == session.cursor), None)
    processed_messages = [m for m in messages if m.createdAt <= last_processed_message.createdAt] if last_processed_message else []
    processed_chat_log = get_chat_log(processed_messages)
    new_messages = [m for m in messages if m.createdAt > last_processed_message.createdAt] if last_processed_message else messages
    new_chat_log = get_chat_log(new_messages)

    # assemble dispatcher prompt
    prompt = dispatcher_template.render(
        agents=agents_text,
        scenario=scenario_template.render(scenario=session.scenario) if session.scenario else "",
        current=current_template.render(current=session.current) if session.current else "",
        processed_chat=processed_chat_template.render(chat_log=processed_chat_log) if processed_chat_log else "",
        new_chat=new_chat_template.render(new_chat=new_chat_log) if new_chat_log else "",
        manna_info=manna_info_template.render(manna=session.budget, manna_spent=session.spent) if session.budget else ""
    )

    print("--------------------------------")
    print(prompt)
    print("--------------------------------")

    
    class DispatcherThought(BaseModel):
        """A thought about how to respond to the last message in the chat."""

        speaker: Literal[*agent_names] = Field(
            None,
            description="The agent that the dispatcher encourages to spontaneously speak or respond to the last message.",
        )
        hint: str = Field(
            ...,
            description="A hint to the chosen speaker about how to respond.",
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