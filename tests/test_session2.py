from typing import Literal
from bson import ObjectId
from eve.auth import get_my_eden_user
from eve.agent2.handlers import MessageRequest, async_receive_message
from eve.agent2.message import ChatMessage, Channel
from eve.agent2.session_create import async_create_session
from eve.agent2.session import Session


from eve.agent2.dispatcher import async_run_dispatcher
from eve.agent2.agent import Agent





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



async def test_create_session():
    user = get_my_eden_user()
    channel = Channel(type="discord", key="1268682080263606443")
    prompt = "Eve is applying for a job to work at McDonalds, and Hypotheticards is the interviewer."
    #result = await async_create_session(user, channel, prompt)

    

    session = Session(
        id=ObjectId('67d36370787d995afc3471bd'),
        user=ObjectId('65284b18f8bbb9bff13ebe65'),
        channel=Channel(type='discord', key='1268682080263606443'),
        title='Job Interview at McDonalds',
        agents=[ObjectId('675fd3af79e00297cdac1324'), ObjectId('678c849671f58075bc837456')],
        scenario="Eve is applying for a job at McDonalds and is having an interview with Hypotheticards, who is the interviewer. The goal of the scenario is for Eve to present her qualifications, skills, and enthusiasm for the job, while Hypotheticards will assess her fit for the position. The scenario will be complete when Hypotheticards provides feedback or a decision on Eve's application.",
        current=None,
        messages=[],
        budget=100.0,
        spent=0
    )

    for i in range(5):
        session = Session.from_mongo(session.id)

        result2 = await async_run_dispatcher(session)
        
        print("DISPATCHER...")
        print(result2)

        speaker = result2.speakers[0]
        agent = Agent.load(speaker)
        
        await async_prompt_thread2(user, agent, session)




async def test_session2():
    user = get_my_eden_user()

    # Create a new session
    request = MessageRequest(
        user_id=str(user.id),
        session_id=None, #"67d115430deaf0504325447a",
        message=ChatMessage(
            content="Eve is applying for a job to work at McDonalds, and Hypotheticards is the interviewer."
        ),
        update_config=None
    )

    result = await async_receive_message(request)    
    print(result)




from eve.user import User
from eve.agent2 import Agent
from eve.agent2.message import ChatMessage
# from eve.tools import Tool



from eve.agent2.llm import async_prompt


async def async_prompt_thread2(
    user: User,
    agent: Agent,
    session: Session,
    # user_messages: Union[UserMessage, List[UserMessage]],
    # tools: Dict[str, Tool],
    # force_reply: bool = False,
    # use_thinking: bool = True,
    # model: Literal[tuple(MODELS)] = DEFAULT_MODEL,
    # user_is_bot: bool = False,
    # stream: bool = False,
):

    model = "claude-3-5-haiku-latest"



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


    print("here are all the messages")
    for m in messages:
        print(m)

    # convert to thread messages
    messages = [m.to_thread(assistant_id=agent.id) for m in messages]


    # maybe this should be an arg?
    last_user_message = messages[-1]

    tools = {} # agent.get_tools()


    


    content, tool_calls, stop = await async_prompt(
        messages,
        system_message=system_message,
        model=model,
        tools=tools,
    )

    # Create assistant message
    assistant_message = ChatMessage(
        sender=agent.id,
        session=session.id,
        reply_to=last_user_message.id,
        content=content or "",
        tool_calls=tool_calls,
    )


    assistant_message.save()

    print("---------- ?234324 ----------------------")
    print(assistant_message)
    print("-------- 234324 ------------------------")




    


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_create_session())

