import asyncio
import os
from pydantic import BaseModel
from typing import Optional

from ..api.api_requests import UpdateConfig
from .dispatcher import async_run_dispatcher
from .message import ChatMessage
from .session import Session
from .agent import Agent



import os
from eve.user import User
from eve.agent2 import Agent
from eve.agent2.message import ChatMessage
# from eve.tools import Tool
from eve.eden_utils import load_template

system_template = load_template("system2")
knowledge_reply_template = load_template("knowledge_reply")
models_instructions_template = load_template("models_instructions")
model_template = load_template("model_doc")




db = os.getenv("DB", "STAGE")

# todo: force-reply, use-thinking, model, user-is-bot
class MessageRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: ChatMessage
    update_config: Optional[UpdateConfig] = None    
    # force_reply: bool = False
    # use_thinking: bool = True
    # model: Optional[str] = None
    # user_is_bot: Optional[bool] = False


"""
"""


"""

Channel

Session
- channel

ChatMessage
- channel
- session

Handlers
- receive user message (cancel)
  - if session
    - run dispatch
  - if no session
    - check if agent mentioned
    - (optional) run dispatch (if >1 agent), run think otherwise
      - use knowledge T/F
      - extended history
- run dispatcher (or think)
  - run prompt (emit update)
  - schedule next dispatch 

- create new session


Events
- on new user message -> receive user message
  - maybe create and schedule new session (tool?)


Session Handling
- new session triggers while loop
- delay between speakers
- run dispatcher between
  - dispatcher schedules next speaker



Types
- group chat (dispatcher)
- DM (no dispatcher, no think)


if agent mentioned, DM'd, or replied to
 - skip dispatcher
 - think
    - use knowledge T/F
    - extended history
 - tools
    - edit constitution (memories)
    - search_db



"""

from .message import Channel




# todo: force-reply, use-thinking, model, user-is-bot
class MessageRequest(BaseModel):
    user_id: str
    channel_id: str
    session_id: Optional[str] = None
    message: ChatMessage
    update_config: Optional[UpdateConfig] = None    
    # force_reply: bool = False
    # use_thinking: bool = True
    # model: Optional[str] = None
    # user_is_bot: Optional[bool] = False




async def async_receive_message(
    request: MessageRequest
):  
    """
    message
    channel
    session (optional), assert session in channel
    """
    
    """
    if channel is dm or reply or mentions
        - reply = true
    if >1 agent in channel, run dispatcher
    if 1 agent in channel, run think
        - speak T/F
        - use knowledge T/F
        - extend history T/F
        - hint
    """

    user = User.from_mongo(request.user_id)
        
    # Todo: check rate limits
    # if rate limit exceeded, don't save message


        
    channel = Channel.from_mongo(request.channel_id)


    # session messages are already processed
    if request.session_id:
        return
    




    result = await async_run_dispatcher(session, request.message)
    print(result)


    speakers = result.speakers or []
    for speaker in speakers:
        agent = Agent.load(speaker)
        print("get agent", agent)
        # tools = agent.get_tools()
        # thread = agent.request_thread()
        # async for msg in async_prompt_thread(
        #     user, 
        #     agent, 
        #     thread, 
        #     user_messages, 
        #     tools, 
        #     force_reply=True, 
        #     use_thinking=False, 
        #     model=DEFAULT_MODEL
        # ):
        #     print(msg)

    return result







from typing import List

async def async_create_session(
    user: User, 
    channel: Channel, 
    agents: List[Agent],
    prompt: str,
):
    """
    - create session
    - set agents
    - set scenario
    - set current
    - save
    """

    pass

from eve.agent2.dispatcher import async_run_dispatcher


async def async_advance_session(session: Session):
    """
    - get cursor
    - get next speaker
    - set current
    - save
    """

    messages = ChatMessage.find({"session": session.id}, sort="createdAt", limit=25)

    last_processed_message = next((m for m in messages if m.id == session.cursor), None)
    
    processed_messages = [m for m in messages if m.createdAt <= last_processed_message.createdAt] if last_processed_message else []
    new_messages = [m for m in messages if m.createdAt > last_processed_message.createdAt] if last_processed_message else messages

    result = await async_run_dispatcher(session, new_messages)



async def async_run_think(    
):
    pass



from eve.agent2.llm import async_prompt
from eve.api.helpers import emit_update


async def async_playout_session(session: Session, n_turns: int = 10):
    user = User.from_mongo(session.user)
    session = Session.from_mongo(session.id)

    # while True:
    for i in range(n_turns):
        dispatch = await async_run_dispatcher(session)
        agent = Agent.load(dispatch.speaker)
        await async_prompt_agent(user, agent, session)
        await asyncio.sleep(1)



async def async_prompt_agent(
    user: User,
    agent: Agent,
    session: Session,
):
    session.reload()

    # while True:
    if True:

        model = "claude-3-5-haiku-latest"
        # model = "gpt-4.5-preview"
        model = "claude-3-7-sonnet-latest"



        system_message = system_template.render(
            name=agent.name, 
            persona=agent.persona, 
            scenario=session.scenario,
            current=session.current,
            manna=session.budget,
            manna_spent=session.spent,
            manna_left=session.budget - session.spent,
        )


        # print("\n\n\n\n\n================================================")
        # print(system_message)
        # print("================================================\n\n\n\n\n")


        # get all messages from session
        messages = ChatMessage.find({"session": session.id}, limit=25)


        if not messages:
            init_message = ChatMessage(
                sender=user.id,
                session=session.id,
                content=system_message,
            )
            messages.append(init_message)

        # convert to thread messages
        messages = [m.to_thread(assistant_id=agent.id) for m in messages]
        # maybe this should be an arg?
        last_user_message = messages[-1]

        tools = agent.get_tools()
        #tools = {k: v for k, v in tools.items() if k in ["flux_dev_lora", "outpaint"]}

        # if not has_flux_dev_lora:
        #     tools.pop("flux_dev_lora", None)

        print("AVAILABLE TOOLS")
        print(tools.keys())
        # tools.update({"flux_inpainting": {}})


        content, tool_calls, stop = await async_prompt(
            messages,
            system_message=system_message,
            model=model,
            tools=tools,
        )

        # attachments = []
        if tool_calls:
            for tool_call in tool_calls:
                tool = tools.get(tool_call.tool)
                if not tool:
                    raise Exception(f"Tool {tool_call.tool} not found.")

                # Start task
                task = await tool.async_start_task(user.id, agent.id, tool_call.args)
                result = await tool.async_wait(task)

                tool_call.result = result
                tool_call.task = task.id

                print(tool_call)

                # # Task completed
                
                # if result["status"] == "completed":
                #     print("THE RESULT")
                #     print(result)
                #     res = result["result"][0]["output"][0]["filename"]
                #     res = f"https://edenartlab-prod-data.s3.amazonaws.com/{res}"
                #     if res:
                #         content += f" {res}"
                #         attachments.append(res)
                # else:
                #     err = result.get("error")
                #     if err:
                #         content += f" {err}"



        # Create assistant message
        assistant_message = ChatMessage(
            sender=agent.id,
            session=session.id,
            # attachments=[],
            reply_to=last_user_message.id,
            content=content or "",
            tool_calls=tool_calls,
        )


        for tc in tool_calls:
            print(tc.result)


        from eve.eden_utils import prepare_result
        
        full_content = assistant_message.content

        print("the tool calls??")
        print(tool_calls)
        for tool_call in tool_calls:
            print("HERE IS THE TOOL CALL")
            print(tool_call)
            result = tool_call.result.copy() #prepare_result(tool_call.result)
            print("HERE IS THE RESULT")
            print(result)
            if result["status"] == "completed":
                for r in prepare_result(result["result"]):
                    for o in r.get("output", []):
                        if o.get("url"):
                            full_content += f'\n{o["url"]}\n'


        print("THE CONTENT IS??")
        print(full_content)



        assistant_message.save()

        print("---45534------- ?234324 ----------------------")
        print(assistant_message)
        print("----345---- 234324 ------------------------")

        from eve.mongo import get_collection
        deployment = get_collection("deployments").find_one({"platform": "discord", "agent": agent.id})
        print(deployment["_id"])
        db = os.getenv("DB", "STAGE")
        update_config = UpdateConfig(
            sub_channel_name=None,
            update_endpoint=f'https://edenartlab--api-{db.lower()}-fastapi-app.modal.run/emissions/platform/discord',
            deployment_id=str(deployment["_id"]),
            discord_channel_id='1003581679916548207', #'1268682080263606443',
            discord_message_id='1351473271065018410' #'1350045471162368033',
        )
        data = {
            'type': 'assistant_message',
            'update_config': update_config.model_dump(),
            'content': full_content
        }

        print(update_config)

        print("THE CONTENT")
        print(assistant_message.content)



        # data = {
        #     "type": UpdateType.ASSISTANT_MESSAGE.value,
        #     "update_config": update_config.model_dump() if update_config else {},
        # }
        # data["content"] = "this is a test" #assistant_message.content

        print("UDAT1231231212EDING")
        print(data)
        print(update_config)
        await emit_update(update_config, data)


        # if stop:
        #     break
        return stop

    
