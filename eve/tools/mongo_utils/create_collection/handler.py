from eve.tool import ToolContext

from ....task import CreationsCollection


async def handler(context: ToolContext):
    name = context.args.get("name")
    description = context.args.get("description")
    public = context.args.get("public", True)

    collection = CreationsCollection(
        user=context.user,
        name=name,
        description=description,
        public=public,
    )
    collection.save()

    return {
        "output": {
            "collection_id": str(collection.id),
            "name": collection.name,
            "description": collection.description,
            "public": collection.public,
        }
    }
