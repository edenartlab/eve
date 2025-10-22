from eve.tool import ToolContext
import openai
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List

from ....agent import Agent
from ....mongo import get_collection
# from ... import utils


agent_template = Template("""<Agent>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <username>{{username}}</username>
  {% if description is not none %}<description>{{description[:150]}}</description>{% endif %}
  {% if persona is not none %}<persona>{{persona[:150]}}</persona>{% endif %}
  <created_at>{{createdAt}}</created_at>
</Agent>""")


search_agents_template = Template("""<Agents>
<Owned>
The following agents are created and managed by the user.
{{docs_owned}}
</Owned>
<Public>
The following agents are public and owned by other users.
{{docs_public}}
</Public>
</Agents>

<Query>
{{query}}
</Query>

<Task>
Analyze these agents and return only the ones that are relevant to this search query. You should prefer the agents made by the user, and if none are found, expand your search to public agents. Do not return more than 10 agents.

Explain why each result matches the query criteria.
</Task>""")


class SearchResult(BaseModel):
    """A matching result from the database search."""

    id: str = Field(..., description="The MongoDB ID of the result")
    name: str = Field(..., description="The name/title of the result")
    description: str = Field(..., description="A brief description of the result")
    relevance: str = Field(
        ...,
        description="A brief explanation of why this result matches the search query",
    )


class SearchResults(BaseModel):
    """Results from searching the database."""

    results: List[SearchResult] = Field(
        ...,
        description="The matching results, ordered by relevance. Include only truly relevant results.",
    )


async def handler(context: ToolContext):
    searcher = context.user
    query = context.args.get("query")

    # Get all documents
    counter = 1
    docs = {}
    collection = get_collection(Agent.collection_name)
    for doc in collection.find({"type": "agent", "deleted": {"$ne": True}}):
        id = str(doc["_id"])
        doc["_id"] = counter
        docs[str(counter)] = {
            "id": id,
            "owned": str(searcher) == str(doc["owner"]),
            "public": doc["public"],
            "used": doc.get("creationCount", 0),
            "doc": doc,
        }
        counter += 1

    docs_owned = sorted(
        (doc for doc in docs.values() if doc["owned"]),
        key=lambda x: x["used"],
        reverse=True,
    )
    docs_public = sorted(
        (doc for doc in docs.values() if not doc["owned"] and doc["public"]),
        key=lambda x: x["used"],
        reverse=True,
    )

    docs_owned = "\n".join(agent_template.render(doc["doc"]) for doc in docs_owned)
    docs_public = "\n".join(agent_template.render(doc["doc"]) for doc in docs_public)

    # Create context for LLM
    prompt = search_agents_template.render(
        docs_owned=docs_owned, docs_public=docs_public, query=query
    )

    # Make LLM call
    system_message = f"""You are a search assistant that helps find relevant Agents based on natural language queries. Analyze the provided items and return only the most relevant matches for the query.
    Be selective - only return items that truly match the query's intent."""

    client = instructor.from_openai(openai.AsyncOpenAI())
    results = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_model=SearchResults,
    )

    # Map the counter ID back to actual doc ID
    matches = []
    for result in results.results:
        doc = docs.get(result.id)
        if doc:
            r = {
                "agent_id": doc["id"],
                "name": result.name,
                "username": doc["doc"].get("username"),
                "relevance": result.relevance,
            }
            if image := doc["doc"].get("userImage"):
                r["filename"] = image.split("/")[-1]
                r["mediaAttributes"] = {
                    "type": "image/jpg",
                    "width": 1024,
                    "height": 1024,
                    "aspectRatio": 1,
                }
            matches.append(r)

    return {"output": matches}
