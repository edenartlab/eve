"""Handler for artifact_get tool."""

from bson import ObjectId

from eve.artifact import Artifact
from eve.tool import ToolContext


async def handler(context: ToolContext):
    """Get an artifact's current state."""
    args = context.args

    artifact_id = args["artifact_id"]
    view = args.get("view", "full")
    include_history = args.get("include_history", False)

    try:
        artifact = Artifact.from_mongo(ObjectId(artifact_id))
    except Exception as e:
        return {"output": {"error": f"Artifact not found: {str(e)}"}}

    # Build response based on view mode
    if view == "summary":
        result = {
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "name": artifact.name,
            "description": artifact.description,
            "version": artifact.version,
            "summary": artifact.get_summary(),
            "updated_at": artifact.updatedAt.isoformat() if artifact.updatedAt else None,
            "created_at": artifact.createdAt.isoformat() if artifact.createdAt else None,
        }
    else:
        # Full view
        result = {
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "name": artifact.name,
            "description": artifact.description,
            "version": artifact.version,
            "data": artifact.data,
            "updated_at": artifact.updatedAt.isoformat() if artifact.updatedAt else None,
            "created_at": artifact.createdAt.isoformat() if artifact.createdAt else None,
        }

    # Include history if requested
    if include_history and artifact.versions:
        result["history"] = [
            {
                "version": v.version,
                "timestamp": v.timestamp.isoformat(),
                "actor_type": v.actor_type,
                "message": v.message,
                "operations_count": len(v.operations),
            }
            for v in artifact.versions[-10:]  # Last 10 versions
        ]

    return {"output": result}
