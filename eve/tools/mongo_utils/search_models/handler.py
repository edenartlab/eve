import openai
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List

from ....utils import load_template
from ....models import Model
from ....mongo import get_collection




model_template = load_template("model_doc")

search_models_template = """<Models>
<Owned>
The following models are trained and managed by the user.
{{docs_owned}}
</Owned>
<Public>
The following models were trained by others but are public.
{{docs_public}}
</Public>
</Models>

<Query>
{{query}}
</Query>

<Task>
Analyze these models and return only the ones that are relevant to this search query. You should prefer models trained by the user, and if none are found, expand your search to public models. Do not return more than 10 models.

Explain why each result matches the query criteria.
</Task>"""

search_models_template = Template(search_models_template)



class SearchResult(BaseModel):
    """A matching result from the database search."""
    
    id: str = Field(..., description="The MongoDB ID of the result")
    name: str = Field(..., description="The name/title of the result")
    description: str = Field(..., description="A brief description of the result")
    relevance: str = Field(
        ..., 
        description="A brief explanation of why this result matches the search query"
    )

class SearchResults(BaseModel):
    """Results from searching the database."""
    
    results: List[SearchResult] = Field(
        ...,
        description="The matching results, ordered by relevance. Include only truly relevant results."
    )


async def handler(args: dict, user: str = None, agent: str = None):
    searcher = user
    query = args.get("query")
    
    # Get all documents
    counter = 1
    docs = {}
    collection = get_collection(Model.collection_name)
    # for doc in collection.find({"base_model": "flux-dev", "deleted": {"$ne": True}}):
    for doc in collection.find({"deleted": {"$ne": True}}):
        id = str(doc["_id"])
        doc["_id"] = counter        
        docs[str(counter)] = {
            "id": id,
            "owned": str(searcher) == str(doc["user"]),
            "public": doc["public"],
            "used": doc.get("creationCount", 0),
            "doc": doc,
        }
        counter += 1
    
    docs_owned = sorted(
        (doc for doc in docs.values() if doc["owned"]),
        key=lambda x: x["used"], 
        reverse=True
    )
    docs_public = sorted(
        (doc for doc in docs.values() if not doc["owned"] and doc["public"]),
        key=lambda x: x["used"], 
        reverse=True
    )[:10]  # limit to only 10 public docs

    docs_owned = "\n".join(model_template.render(doc["doc"]) for doc in docs_owned)
    docs_public = "\n".join(model_template.render(doc["doc"]) for doc in docs_public)

    # Create context for LLM
    prompt = search_models_template.render(
        docs_owned=docs_owned, 
        docs_public=docs_public, 
        query=query
    )

    print("--------------------------------")
    print(prompt[:500])
    print("--------------------------------")

    # Make LLM call
    system_message = f"""You are a search assistant that helps find relevant Models based on natural language queries. Analyze the provided items and return only the most relevant matches for the query.
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
                "model_id": doc["id"],
                "name": result.name,
                "username": doc["doc"].get("username"),
                "relevance": result.relevance,
            }
            if image := doc["doc"].get("thumbnail"):
                r["filename"] = image.split("/")[-1].replace(".jpg", "_768.webp")
                r["mediaAttributes"] = {
                    "type": "image/png",
                    "width": 2048,
                    "height": 2048,
                    "aspectRatio": 1,
                }            
            matches.append(r)

    return {
        "output": matches
    }
