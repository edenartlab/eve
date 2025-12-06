"""Handler for artifact_update tool."""

from bson import ObjectId

from eve.artifact import Artifact
from eve.tool import ToolContext


async def handler(context: ToolContext):
    """Apply structured updates to an artifact."""
    args = context.args

    artifact_id = args["artifact_id"]
    operations = args["operations"]
    message = args.get("message")

    # Determine actor from context
    actor_type = "agent" if context.agent else "user"
    actor_id = ObjectId(context.agent) if context.agent else (
        ObjectId(context.user) if context.user else None
    )

    try:
        artifact = Artifact.from_mongo(ObjectId(artifact_id))
    except Exception as e:
        return {"output": {"error": f"Artifact not found: {str(e)}"}}

    # Verify ownership - user must own the artifact to update it
    if context.user:
        user_id = ObjectId(context.user)
        if artifact.owner != user_id:
            return {"output": {"error": "Unauthorized: You do not own this artifact"}}

    # Store previous version for comparison
    previous_version = artifact.version

    # Apply operations
    try:
        artifact.apply_operations(
            operations=operations,
            actor_type=actor_type,
            actor_id=actor_id,
            message=message,
            save=True,
        )
    except Exception as e:
        return {"output": {"error": f"Failed to apply operations: {str(e)}"}}

    # Build response
    result = {
        "artifact_id": str(artifact.id),
        "name": artifact.name,
        "previous_version": previous_version,
        "new_version": artifact.version,
        "operations_applied": len(operations),
        "message": message or "Update applied successfully",
        "data": artifact.data,  # Return updated data
    }

    return {"output": result}
