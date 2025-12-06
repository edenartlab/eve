"""Handler for artifact_list tool."""

from bson import ObjectId

from eve.artifact import Artifact
from eve.tool import ToolContext


async def handler(context: ToolContext):
    """List artifacts accessible in the current context."""
    args = context.args

    session_only = args.get("session_only", True)
    artifact_type = args.get("type")
    include_archived = args.get("include_archived", False)
    limit = args.get("limit", 20)

    artifacts = []

    if session_only and context.session:
        # List artifacts linked to the current session
        session_id = ObjectId(context.session)
        artifacts = Artifact.find_for_session(
            session_id, include_archived=include_archived
        )

        # Filter by type if specified
        if artifact_type:
            artifacts = [a for a in artifacts if a.type == artifact_type]

        # Apply limit
        artifacts = artifacts[:limit]

    elif context.user:
        # List all artifacts owned by the user
        user_id = ObjectId(context.user)
        artifacts = Artifact.find_for_user(
            user_id,
            artifact_type=artifact_type,
            include_archived=include_archived,
            limit=limit,
        )
    else:
        return {"output": {"error": "User or session context required"}}

    # Format results
    result = {
        "count": len(artifacts),
        "artifacts": [
            {
                "artifact_id": str(a.id),
                "type": a.type,
                "name": a.name,
                "description": a.description,
                "version": a.version,
                "summary": a.get_summary(max_length=100),
                "updated_at": a.updatedAt.isoformat() if a.updatedAt else None,
                "archived": a.archived,
            }
            for a in artifacts
        ],
    }

    return {"output": result}
