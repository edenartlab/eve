from eve.tool import ToolContext
import openai
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List

from ....task import CreationsCollection, Creation
# from ... import utils


collection_template = Template("""<Agent>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <coverCreation>{{coverCreation}}</coverCreation>
  <created_at>{{createdAt}}</created_at>
</Agent>""")


search_collections_template = Template("""<Collections>
<Owned>
The following collections are created and managed by the user.
{{docs_owned}}
</Owned>
<Public>
The following collections are public and owned by other users.
{{docs_public}}
</Public>
</Collections>

<Query>
{{query}}
</Query>

<Task>
Analyze these collections and return only the ones that are relevant to this search query. You should prefer the collections made by the user, and if none are found, expand your search to public collections. Important: Do *not* return more than {{num_results}} collections. Only the most relevant.

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
    num_results = context.args.get("results", 5)

    # Get all documents
    counter = 1
    docs = {}
    collection = CreationsCollection.get_collection()
    for doc in collection.find({"deleted": {"$ne": True}}):
        id = str(doc["_id"])
        doc["_id"] = counter
        docs[str(counter)] = {
            "id": id,
            "owned": str(searcher) == str(doc["user"]),
            "public": doc["public"],
            "used": len(doc.get("creations", [])),
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

    docs_owned = "\n".join(collection_template.render(doc["doc"]) for doc in docs_owned)
    docs_public = "\n".join(
        collection_template.render(doc["doc"]) for doc in docs_public
    )

    # Create context for LLM
    prompt = search_collections_template.render(
        docs_owned=docs_owned,
        docs_public=docs_public,
        query=query,
        num_results=num_results,
    )

    # Make LLM call
    system_message = f"""You are a search assistant that helps find relevant Collections based on natural language queries. Analyze the provided items and return only the most relevant matches for the query. Be selective - only return items that truly match the query's intent."""

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
        cover = Creation.find_one({"_id": doc["doc"]["coverCreation"]})
        if doc:
            r = {
                "collection_id": doc["id"],
                "name": doc["doc"]["name"],
                "relevance": result.relevance,
            }
            if cover:
                r["filename"] = cover.filename
                r["mediaAttributes"] = {
                    "type": "image/jpg",
                    "width": 1024,
                    "height": 1024,
                    "aspectRatio": 1,
                }
            matches.append(r)

    return {"output": matches}
