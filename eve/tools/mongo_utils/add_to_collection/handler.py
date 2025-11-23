from eve.tool import ToolContext

from ....task import Creation, CreationsCollection


async def handler(context: ToolContext):
    # query = context.args.get("query")
    # tool = Tool.load("search_collections")
    # results = await tool.async_run({
    #     "query": query,
    #     "results": 1
    # })
    # results = results.get("output", [])

    # assert len(results) >= 1, "No collections found for this query"

    # collection_id = results[0]["collection_id"]
    collection_id = context.args.get("collection_id")
    creation_id = context.args.get("creation_id")

    collection = CreationsCollection.from_mongo(collection_id)
    collection.add_creation(creation_id)
    collection.save()

    creation = Creation.from_mongo(creation_id)

    return {
        "output": {
            "filename": creation.filename,
            "mediaAttributes": creation.mediaAttributes,
        }
    }
