import json
from bson import ObjectId
from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.agent.thread import Thread, UserMessage, AssistantMessage, ToolCall
from eve.agent.tasks import title_thread
from eve.agent.session.models import UpdateType
from eve.agent.run_thread import prompt_thread


def test_title_thread():
    """Pytest entry point"""

    thread = Thread.from_mongo("67cb86a67b67ef56fa99d000")
    results = title_thread(thread=thread)


def test_thread():
    user = get_my_eden_user()

    # Create a new thread
    thread = Thread(key="test-thread", user=user.id, message_limit=5)

    # Test adding messages
    thread.messages = [
        UserMessage(content="Hello, can you help me?"),
        AssistantMessage(content="Of course! What do you need help with?"),
        UserMessage(content="Make an image of a cat"),
        AssistantMessage(
            content="I'll create that for you",
            tool_calls=[
                ToolCall(
                    id="tool_call_1",
                    tool="txt2img",
                    args={"prompt": "A cute orange cat", "width": 512, "height": 512},
                )
            ],
        ),
    ]

    # Test message limit functionality
    for i in range(10):
        thread.messages.append(UserMessage(content=f"Message {i}"))

    # Verify message limit is enforced
    assert len(thread.messages) > 5  # Before applying limit
    messages = thread.get_messages()
    assert len(messages) == 5  # After applying limit

    # Test hidden messages
    thread.messages = [
        UserMessage(content="First message"),
        UserMessage(content="Hidden message", hidden=True),
        AssistantMessage(content="Response"),
    ]

    # Check that hidden messages are excluded (except the last one)
    visible_messages = thread.get_messages()
    assert len(visible_messages) == 2
    assert visible_messages[0].content == "First message"
    assert visible_messages[1].content == "Response"

    # Test with hidden message as the last message
    thread.messages.append(UserMessage(content="Trigger message", hidden=True))
    visible_messages = thread.get_messages()
    assert len(visible_messages) == 3
    assert visible_messages[2].content == "Trigger message"

    # Test updating tool call
    thread.messages = [
        UserMessage(content="Make an image"),
        AssistantMessage(
            id=ObjectId(),
            content="Processing",
            tool_calls=[
                ToolCall(
                    id="tool_id_1",
                    tool="txt2img",
                    args={"prompt": "A cat"},
                    status="pending",
                )
            ],
        ),
    ]

    # Update the tool call in memory
    tool_call = thread.messages[1].tool_calls[0]
    tool_call.status = "completed"
    tool_call.result = [{"output": [{"url": "https://example.com/cat.jpg"}]}]

    # Verify update worked
    assert thread.messages[1].tool_calls[0].status == "completed"
    assert "example.com/cat.jpg" in str(thread.messages[1].tool_calls[0].result)


def test_prompt_thread():
    user = get_my_eden_user()

    agent = Agent.load("eve")
    tools = agent.get_tools()
    thread = agent.request_thread(key="test")

    messages = [
        UserMessage(
            content="eve, make another picture of a fancy dog using flux_schnell"
        )
    ]

    for update in prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        model="gpt-4o-mini",
    ):
        assert update.type != UpdateType.ERROR
