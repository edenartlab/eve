from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent.run_thread import prompt_thread, async_prompt_thread
from eve.agent.think import async_think
from eve.agent import Agent

from eve.agent import Tool

async def test_session():

    user = get_my_eden_user()

    messages = [
        UserMessage(name="kate", content="make a picture of a cat"),
    ]

    agent = Agent.load("eve")
    agent.tools = {
        "txt2img": {},
    }
    tools = agent.get_tools()
    print("TOOLS", tools)
    print("TOOLS", tools.keys())
    thread = agent.request_thread()

    async for msg in async_prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        force_reply=True,
        use_thinking=False,
        model="gpt-4o-mini"
    ):
        print(msg)




from bson import ObjectId
from eve.agent.session import Session, SessionMessage

# # Example usage of Session/SessionMessage/AgentInfo
# if __name__ == "__main__":

    
#     agents = [
#         Agent.load("eve"),
#         Agent.load("banny"),
#         Agent.load("oppenheimer"),
#     ]
    
#     session = Session(
#         key="session-example",
#         title="How to use Eden",
#         scenario_setup="Eve is helping Banny and Oppenheimer understand how to use Eden.",
#         current_situation="This is the beginning of the conversation.",
#         agents=agents,
#         message_limit=10
#     )

#     user = get_my_eden_user()

#     messages = [
#         SessionMessage(
#             sender=agents[0],
#             content="I am here to help you understand how to use Eden."
#         ),
#         SessionMessage(
#             sender=agents[1],
#             content="help me depict the Juicebox project in images"
#         ),
#         SessionMessage(
#             sender=agents[0],
#             content="Here is a picture of the Juicebox project. https://juicebox.com/examples"
#         ),
#         SessionMessage(
#             sender=agents[2],
#             content="That's great"
#         ),
#         SessionMessage(
#             sender=user,
#             content="can you be more specific?"
#         ),
#         SessionMessage(
#             sender=agents[1],
#             content="how do i make a picture of a cat with a fancy model?"
#         )
#     ]

#     session.messages = messages
#     # 4. Convert the session to a single-agent Thread for 'Agent A' (the math tutor)
#     #    All messages from agent_a_id become assistant messages, everything else becomes user messages.
#     thread_for_agent_a = session.to_thread(
#         target_agent=agents[0], 
#         user=user,
#     )

#     # 5. Print out the Thread messages to see how they cast
#     #    We expect agent A's messages to be "assistant" role, all others to be "user"
#     print("=== Thread for Agent A ===")
#     for msg in thread_for_agent_a.get_messages():
#         print(f"{msg.role.upper()} (id={msg.id}): {msg.content}", type(msg))

#     # 6. Alternatively, convert the same session to a Thread for Agent B (the 'assistant') 
#     thread_for_agent_b = session.to_thread(
#         target_agent=agents[1],
#         user=user,
#     )
    
#     print("\n=== Thread for Agent B ===")
#     for msg in thread_for_agent_b.get_messages():
#         print(f"{msg.role.upper()} (id={msg.id}): {msg.content}", type(msg))



def test_session():
    user = get_my_eden_user()
    
    # Create a few agents for our session
    agent1 = Agent.load("eve")
    agent2 = Agent.load("banny")

    # Create a new session
    session = Session(
        key="test-session",
        title="Testing Session",
        scenario_setup="A conversation between Eve and Banny",
        current_situation="Banny needs help with image generation",
        agents=[agent1, agent2],
        message_limit=5
    )
    
    # Test adding messages to the session
    session.messages = [
        SessionMessage(
            sender_id=user.id,
            content="Hello, can someone help me?"
        ),
        SessionMessage(
            sender_id=agent1.id,
            content="Of course! How can I assist you today?"
        ),
        SessionMessage(
            sender_id=user.id,
            content="I need an image of a cat"
        ),
        SessionMessage(
            sender_id=agent2.id,
            content="I know about cats! They're fuzzy."
        )
    ]
    
    # Test message limit functionality
    for i in range(10):
        session.messages.append(SessionMessage(sender_id=user.id, content=f"Message {i}"))
    
    # Verify message limit is enforced
    assert len(session.messages) > 5  # Before applying limit
    messages = session.get_messages()
    assert len(messages) == 5  # After applying limit
    
    # Test hidden messages
    session.messages = [
        SessionMessage(sender_id=user.id, content="First message"),
        SessionMessage(sender_id=agent1.id, content="Hidden message", hidden=True),
        SessionMessage(sender_id=agent2.id, content="Response")
    ]
    
    # Check that hidden messages are excluded (except the last one)
    visible_messages = session.get_messages()
    assert len(visible_messages) == 2
    assert visible_messages[0].content == "First message"
    assert visible_messages[1].content == "Response"
    
    # Test with hidden message as the last message
    session.messages.append(SessionMessage(sender_id=user.id, content="Trigger message", hidden=True))
    visible_messages = session.get_messages()
    assert len(visible_messages) == 3
    assert visible_messages[2].content == "Trigger message"
    
    # Convert to thread for agent1 (Eve)
    thread_for_eve = session.to_thread(
        target_agent=agent1,
        user=user,
    )

    # Verify the conversion worked correctly
    assert len(thread_for_eve.messages) == 3  # Only visible messages
    assert thread_for_eve.agent == agent1.id
    assert thread_for_eve.user == user.id
    
    # Eve's messages should be assistant messages, others should be user messages
    assert thread_for_eve.messages[0].role == "user"
    assert thread_for_eve.messages[0].content == "First message"
    
    assert thread_for_eve.messages[1].role == "user"
    assert thread_for_eve.messages[1].content == "Response"
    assert thread_for_eve.messages[1].name == "Eve"
    
    assert thread_for_eve.messages[2].role == "user"
    assert thread_for_eve.messages[2].content == "Trigger message"
    
    # Test converting to thread for agent2 (Banny)
    thread_for_banny = session.to_thread(
        target_agent=agent2,
        user=user,
    )
    
    assert thread_for_banny.messages[1].role == "assistant"
    assert thread_for_banny.messages[1].content == "Response"








################################################################################
# Below this is experimental


################################################################################
# Session api tests
################################################################################

from eve.api.handlers import handle_session_message
from eve.api.handlers import SessionMessageRequest

async def test_session2():
    user = get_my_eden_user()

    # Create a new session
    request = SessionMessageRequest(
        user_id=str(user.id),
        user_message=UserMessage(content="eve and banny say hello to each other, then banny requests an image of a banana at a party and eve makes it for him. they then say goodnight to each other and stop."),
    )
    
    result = await handle_session_message(request)
    
    print(result)



if __name__ == "__main__":
    import asyncio
    asyncio.run(test_session2())