import asyncio
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional

from ..eden_utils import dump_json, load_template
from .llm import async_prompt
from .agent import Agent, refresh_agent
from .thread import UserMessage, Thread
from .session import Session, SessionMessage


knowledge_think_template = load_template("knowledge_think")
thought_template = load_template("thought")


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
            ...,
            description="Ignore if last message is irrelevant, reply if relevant or criteria met.",
        )
        thought: str = Field(
            ...,
            description="A very brief thought about what relevance, if any, the last user message has to you, and a justification of your intention.",
        )
        recall_knowledge: bool = Field(
            ...,
            description="Whether to look up relevant knowledge from the knowledge base.",
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

    prompt = thought_template.render(
        name=agent.name,
        message_count=len(messages),
        chat=chat,
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


def think(
    agent: Agent, thread: Thread, user_message: UserMessage, force_reply: bool = True
):
    return asyncio.run(async_think(agent, thread, user_message, force_reply))



















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



async def dispatch(
    session: Session,
    new_message: SessionMessage,
):
    agents = session.agents
    agent_names = [a.username for a in agents]
    agents_text = "\n".join([agent_template.render(a) for a in agents])

    # generate text blob of chat history
    chat = ""
    messages = session.get_messages(25)
    for msg in messages:
        content = msg.content
        if msg.role == "user":
            if msg.attachments:
                content += f" (attachments: {msg.attachments})"
            name = msg.name or "User"
        elif msg.role == "assistant":
            name = msg.name or "Assistant"
            for tc in msg.tool_calls:
                args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
                tc_result = dump_json(tc.result, exclude="blurhash")
                content += f"\n -> {tc.tool}({args}) -> {tc_result}"
        time_str = msg.createdAt.strftime("%H:%M")
        chat += f"<{name} {time_str}> {content}\n"


    scenario = (
        "They are discussing their relationship dynamics to remain good friends. "
        "The goal is to reach a clear understanding of each other's feelings."
    )
    current = (
        "So far, Alice has tried to confess to Bob, Bob has expressed interest in Carol, "
        "and Carol is politely unsure how to respond."
    )

    latest_user_message = new_message.content

    prompt = dispatcher_template.render(
        agents=agents_text,
        scenario=scenario,
        current=current,
        chat_log=chat,
        latest_message=latest_user_message,
    )

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





# async def test_dispatcher():

#     user = get_my_eden_user()

#     agents = [
#         Agent.load("oppenheimer"),
#         Agent.load("banny"),
#         Agent.load("abraham"),
#     ]
    
#     session = Session(
#         key="test-session",
#         title="Testing Session",
#         scenario_setup="A conversation between Eve and Banny",
#         current_situation="Banny needs help with image generation",
#         agents=agents,
#         message_limit=15
#     )

#     session.messages = [
#         SessionMessage(
#             sender_id=user.id,
#             content="Hello, can someone help me?"
#         ),
#         SessionMessage(
#             sender_id=agents[0].id,
#             content="Of course! How can I assist you today?"
#         ),
#         SessionMessage(
#             sender_id=user.id,
#             content="I need an image of a cat"
#         ),
#         SessionMessage(
#             sender_id=agents[1].id,
#             content="I know about cats! They're fuzzy."
#         )
#     ]






#     # result = await async_run_dispatcher(
#     #     thread=thread,
#     #     user_message=messages[0],
#     #     force_reply=True,
#     # )
    
#     # print(result)



# import asyncio
# # asyncio.run(test_sub())

# asyncio.run(test_dispatcher())


# """



# """


