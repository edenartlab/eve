"""
This is experimental. Not actually imported anywhere.
Will be integrated in the future.
"""

import os
import asyncio
from eve.llm import *
from eve.tool import get_tools_from_mongo
from eve.app.auth.auth import get_my_eden_user

db = "STAGE"
user = get_my_eden_user(db=db)
thread_name_think = "test_soc23_think"
thread_name_act = "test_soc23_act"
tools = get_tools_from_mongo(db=db)

thread_think = Thread.from_name(name=thread_name_think, user=user.id, db=db)
thread_act = Thread.from_name(name=thread_name_act, user=user.id, db=db)

thread_act.push("messages", UserMessage(content="I am your inner voice, guiding you as you attempt to creare digital art for me to critique. As you create art, I will give you feedback and instruct you on ways to improve your work. Please refrain from restating what I tell you or being verbose. Just make the art, tell me concisely what you are doing, and show me the results."))


user_message = UserMessage(content="I am an artist who is trying to create an experimental great work of art. You are my inner voice, guiding me as I create art for you to critique. Start by making a suggestion for some kind of visual artwork for me to create. I will make something and show it to you, and then you will give me feedback and instruct me how to make changes to it. Make sure to keep me focused on making things, using my digital tools, and showing you images. And don't just keep iterating on the same thing. Try asking me to change things up, make totally different things occasionally. Go off the board. Now first begin by suggesting some image content to start with. Note: please do not be so verbose. 2-3 sentences max.")



def convert_assistant_messages(messages: List[AssistantMessage]):
    all_user_messages = []
    for message in messages:
        attachments = []
        for tool_call in message.tool_calls or []:
            result = tool_call.get_result()
            if result.get("status") == "completed":
                attachments.extend(
                    output["url"]
                    for result_item in result.get("result", [])
                    for output in result_item.get("output", [])
                    if "url" in output
                )
        user_message = UserMessage(
            content=message.content,
            attachments=attachments
        )
        all_user_messages.append(user_message)
    return all_user_messages



# def prompt_thread2(db, user_id, thread_name, user_message, tools):
#     async def run():
        
#         return all_messages
#     return asyncio.run(run())

from eve.llm import print_message

async def main():

    input_message = user_message
    runs = 0
    
    while True:
        think_messages = []
        async for message in async_prompt_thread(db, user_id, thread_name_think, input_message, {}):
            print_message(message, name="Eve 1")
            think_messages.append(message)
        
        think_messages = convert_assistant_messages(think_messages)

        act_messages = []
        async for message in async_prompt_thread(db, user_id, thread_name_act, think_messages, tools):
            print_message(message, name="Eve 2")
            act_messages.append(message)

        act_messages = convert_assistant_messages(act_messages)

        input_message = act_messages
        runs += 1

        if runs == 10:
            break


    # prompt_thread(db, user_id, thread_name_act, think_message, tools)
    # act_message = Thread.from_name(name=thread_name_act, user=ObjectId(user_id), db=db).messages[-1]

    # print("==========")
    # print(act_message)


asyncio.run(main())