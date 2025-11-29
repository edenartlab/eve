"""Handler for artifact_create tool."""

from bson import ObjectId

from eve.artifact import Artifact
from eve.tool import ToolContext


async def handler(context: ToolContext):
    """Create a new artifact."""
    args = context.args

    # Extract parameters
    artifact_type = args["type"]
    name = args["name"]
    description = args.get("description")
    data = args["data"]
    link_to_session = args.get("link_to_session", True)

    # Get owner from context
    if not context.user:
        return {"output": {"error": "User context required to create artifacts"}}

    owner_id = ObjectId(context.user)
    session_id = ObjectId(context.session) if context.session else None

    # Create the artifact
    artifact = Artifact(
        type=artifact_type,
        name=name,
        description=description,
        data=data,
        owner=owner_id,
        session=session_id if link_to_session else None,
        sessions=[session_id] if session_id and link_to_session else [],
    )

    artifact.save()

    # Return artifact info
    result = {
        "artifact_id": str(artifact.id),
        "type": artifact.type,
        "name": artifact.name,
        "version": artifact.version,
        "message": f"Artifact '{name}' created successfully",
    }

    return {"output": result}
