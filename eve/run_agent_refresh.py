import random
import openai
import instructor
from time import time
from typing import List
from datetime import datetime, timezone, timedelta
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from eve.agent import Agent
from eve.mongo import get_collection
from eve.thread import Thread
# from eve.agent import AgentText

agents = get_collection(Agent.collection_name)

# beepler
# example_agent
# bombay_beach
# cyberswami
# vj
# desci
# anime
# photo
# gamekeeper

class KnowledgeDescription(BaseModel):
    """Defines when and why a reference document should be consulted to enhance responses."""

    summary: str = Field(..., description="A precise, content-focused summary of the document, detailing what information it contains without unnecessary adjectives or filler words.")
    retrieval_criteria: str = Field(..., description="A clear, specific description of when the reference document is needed to answer a user query. This should specify what topics, types of questions, or gaps in the assistant’s knowledge require consulting the document.")


class Suggestion(BaseModel):
    """A prompt suggestion for an Agent in two parts: a concise tagline, and a longer prompt for an LLM. The prompt should correspond to the agent's personality."""

    label: str = Field(..., description="A short and catchy tagline, no more than 7 words, to go into a home page button. Shorten, omit stop words (the, a, an, etc) when possible.")
    prompt: str = Field(..., description="A longer version of the tagline, a prompt to be sent to the agent following its greeting. The prompt should be no more than one sentence or 30 words.")


class AgentText(BaseModel):
    """A text prompt for an Agent in two parts: a concise tagline, and a longer prompt for an LLM. The prompt should correspond to the agent's personality."""

    suggestions: List[Suggestion] = Field(..., description="A list of prompt suggestions and corresponding taglines for the agent. Should be appropriate to the agent's description.")
    greeting: str = Field(..., description="A very short greeting for the agent to use as a conversation starter with a new user. Should be no more than 10 words.")

    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
                {
                    "greeting": "I'm your personal creative assistant! How can I help you?",
                    "suggestions": [
                        {
                            "tagline": "What tools can you use?",
                            "prompt": "Give me a list of all of your tools, and explain your capabilities.",
                        },
                        {
                            "tagline": "Help me make live visuals",
                            "prompt": "I'm making live visuals for an upcoming event. Can you help me?",
                        },
                        {
                            "tagline": "Turn a sketch into a painting",
                            "prompt": "I'm making sketches and doodles in my notebook, and I want to transform them into a digital painting.",
                        },
                        {
                            "tagline": "Draft a character",
                            "prompt": "Help me write out a character description for a video game I am producing.",
                        }
                    ]
                },
                {
                    "greeting": "What kind of a story would you like to write together?",
                    "suggestions": [
                        {
                            "tagline": "Make a romantic story",
                            "prompt": "I want to write a romantic comedy about a couple who meet at a party. Help me write it.",
                        },
                        {
                            "tagline": "Imagine a character",
                            "prompt": "I would like to draft a protagonist for a novel I'm writing about the sea.",
                        },
                        {
                            "tagline": "What have you written before?",
                            "prompt": "Tell me about some of the previous stories you've written.",
                        },
                        {
                            "tagline": "Revise the style of my essay",
                            "prompt": "I've made an essay about the history of the internet, but I'm not sure if it's written in the style I want. Help me revise it.",
                        }
                    ]
                }            
            ]
        }
    )



knowledge_template = """<Agent Description>
Name: {{name}}
Description: {{agent_description}}
</Agent Description>

<Reference>
This is {{name}}'s full reference document or knowledge:
---
{{knowledge_base}}
---
</Reference>
<Task>
Your task is to generate a KnowledgeDescription for a reference document. Given a description of yourself and access to the document, analyze its contents and produce the following:

summary – A concise, detailed description of what information is contained in the reference document. Focus on subjects, topics, facts, and structure rather than adjectives or generalizations. Be specific about what kind of knowledge is present.

retrieval_criteria – A structured, single-instruction paragraph that clearly defines when the reference document should be consulted. Identify the subjects, topics, types of questions, or knowledge gaps that require retrieving the document’s contents. This should help the assistant determine whether the document is necessary to accurately respond to a user message. Avoid overly broad conditions to prevent unnecessary retrievals, but ensure all relevant cases are covered.
</Task>"""


async def generate_agent_knowledge_description(agent: Agent):
    system_message = "You receive a description of an agent, along with a large document of information the agent must memorize, and you come up with instructions for the agent on when they should consult the reference document."

    prompt = Template(knowledge_template).render(
        name=agent["username"],
        agent_description=agent["persona"],
        knowledge_base=agent["knowledge"]
    )

    client = instructor.from_openai(openai.AsyncOpenAI())
    
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=KnowledgeDescription,
    )

    return result


async def generate_agent_text(agent: Agent):
    system_message = "You receive a description of an agent and come up with a greeting and suggestions for those agents' example prompts and taglines."

    prompt = f"""Come up with exactly FOUR (4, no more, no less) suggestions for sample prompts for the agent {agent["username"]}, as well as a simple greeting for the agent to begin a conversation with. Make sure all of the text is especially unique to or appropriate to {agent["username"]}, given their description. Do not use exclamation marks. Here is the description of {agent["username"]}:\n\n{agent["persona"]}."""

    client = instructor.from_openai(openai.AsyncOpenAI())
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=AgentText,
    )

    return result


async def rotate_agent_suggestions(since_hours=6):
    """
    Rotate agent suggestions, greetings, and knowledge descriptions for agents whose updatedAt is younger than 6 hours or null (new agents)
    """

    filter = {}
    if since_hours:
        filter["type"] = "agent"
        filter["$or"] = [
            {"refreshed_at": None},
            {"updatedAt": None},
            {"updatedAt": {"$gt": datetime.now(timezone.utc) - timedelta(hours=since_hours)}}
        ]

    for agent in agents.find(filter):
        updated_at = (agent.get("updatedAt") or agent["createdAt"]).replace(tzinfo=timezone.utc)
        refreshed_at = agent.get("refreshed_at")
        if refreshed_at:
            refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
        
        if refreshed_at and (updated_at - refreshed_at).total_seconds() <= 0:
           continue

        print("Refresh agent", agent["username"])
        
        agent_text = await generate_agent_text(agent)
        knowledge_description = await generate_agent_knowledge_description(agent)
        time = datetime.now(timezone.utc)

        update = {
            "knowledge_description": f"Summary: {knowledge_description.summary}. Retrieval Criteria: {knowledge_description.retrieval_criteria}",
            "greeting": agent_text.greeting,
            "suggestions": [s.model_dump() for s in agent_text.suggestions],
            "refreshed_at": time, 
            "updatedAt": time,
        }

        print(update)

        agents.update_one({"_id": agent["_id"]}, {"$set": update})


import asyncio
asyncio.run(rotate_agent_suggestions())