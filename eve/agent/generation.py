"""Consolidated generation-model access resolution.

Single source of truth for three questions that were previously answered by
divergent copies of the same logic (create/handler.py's nano_banana_enabled /
veo3_enabled blocks vs the billing path in tool.py):

1. Who pays for a generation (owner_pays / subsession semantics)?
2. What is that payer entitled to (subscription / feature flags)?
3. What generation posture and model preferences apply (request arg >
   agent settings > paying-user preferences > defaults)?

The premium tier requires BOTH keys: the agent owner's opt-in
(Agent.generation_settings.premium_models_enabled) and the paying user's
entitlement. A trigger user must never be able to point someone else's wallet
at premium models, and a paying subscriber must never be routed premium by an
agent whose owner didn't sanction it.
"""

from dataclasses import dataclass, field
from typing import Optional

from bson import ObjectId
from loguru import logger

# Flags that entitle a paying user to the premium generation tier, in addition
# to any paid subscription (subscriptionTier > 0). Kept as named constants so
# tightening (e.g. tier >= 2) is a one-line change.
PREMIUM_GEN_FLAGS = {"eden_admin", "free_tools", "preview", "tool_access_premium_models"}

# The pre-existing veo3 / nano-banana-pro subscriber gate, preserved verbatim.
SUBSCRIBER_FLAGS = {"eden_admin", "free_tools", "preview", "tool_access_veo3"}


@dataclass
class GenerationAccess:
    paying_user: Optional[object] = None  # User doc or None (internal/unauthenticated)
    subscriber: bool = False              # legacy veo3/nano-banana-pro gate
    premium_entitled: bool = False        # paying user may use the premium tier
    premium_enabled: bool = False         # premium_entitled AND agent opted in
    default_quality: str = "standard"
    image_model_preference: Optional[str] = None
    video_model_preference: Optional[str] = None
    notices: list = field(default_factory=list)


def _load(doc_or_id, cls):
    if doc_or_id is None or isinstance(doc_or_id, cls):
        return doc_or_id
    try:
        return cls.from_mongo(ObjectId(str(doc_or_id)))
    except Exception as e:
        logger.warning(f"generation: could not load {cls.__name__} {doc_or_id}: {e}")
        return None


def resolve_paying_user(
    user=None,
    agent=None,
    session=None,
    is_client_platform: bool = True,
):
    """Resolve who pays, following the billing path's semantics (tool.py
    handle_start_task): owner pays when owner_pays == "full", or when
    owner_pays == "deployments" AND the request came from a client platform;
    for subsessions, the acting agent is the root session's agent.

    Accepts documents or ids. Returns (paying_user_doc, agent_doc) — either
    may be None.
    """
    from eve.agent import Agent
    from eve.user import User

    user_doc = _load(user, User)
    agent_doc = _load(agent, Agent)
    paying = user_doc

    if agent_doc is not None:
        acting_agent = agent_doc
        if session is not None:
            try:
                from eve.agent.session.models import Session

                session_doc = _load(session, Session)
                while session_doc is not None and session_doc.parent_session:
                    session_doc = Session.from_mongo(session_doc.parent_session)
                if (
                    session_doc is not None
                    and session_doc.agents
                    and str(session_doc.agents[0]) != str(acting_agent.id)
                ):
                    acting_agent = (
                        _load(session_doc.agents[0], Agent) or acting_agent
                    )
            except Exception as e:
                logger.warning(f"generation: subsession walk failed: {e}")

        if acting_agent.owner_pays == "full" or (
            acting_agent.owner_pays == "deployments" and is_client_platform
        ):
            owner = _load(acting_agent.owner, User)
            if owner is not None:
                paying = owner
        agent_doc = acting_agent

    return paying, agent_doc


def _entitlements(paying_user):
    if paying_user is None:
        return False, False
    flags = set(paying_user.featureFlags or [])
    tier = paying_user.subscriptionTier or 0
    subscriber = bool(flags & SUBSCRIBER_FLAGS) or tier > 0
    premium_entitled = bool(flags & PREMIUM_GEN_FLAGS) or tier > 0
    return subscriber, premium_entitled


def resolve_generation_access(
    user=None,
    agent=None,
    session=None,
    is_client_platform: bool = True,
    paying_user=None,
) -> GenerationAccess:
    """Full access + posture resolution.

    Pass a pre-resolved paying_user (e.g. from the Task doc) when available —
    it is authoritative because billing already used it; re-derivation is the
    fallback for legacy/direct call sites.
    """
    from eve.agent import Agent
    from eve.user import User

    agent_doc = _load(agent, Agent)
    if paying_user is not None:
        paying = _load(paying_user, User)
    else:
        paying, agent_doc = resolve_paying_user(
            user=user, agent=agent_doc, session=session,
            is_client_platform=is_client_platform,
        )

    subscriber, premium_entitled = _entitlements(paying)

    gen = getattr(agent_doc, "generation_settings", None)
    premium_enabled = premium_entitled and (
        bool(getattr(gen, "premium_models_enabled", False))
        if agent_doc is not None
        # No agent (direct human tool use): the user sees the price in the UI;
        # entitlement alone gates.
        else True
    )

    # Posture: agent pro posture requires entitlement; user pro posture is
    # honored only when the preference-holder IS the paying user.
    default_quality = "standard"
    if agent_doc is not None and gen is not None:
        if gen.default_quality == "pro" and (subscriber or premium_entitled):
            default_quality = "pro"
    user_doc = _load(user, User)
    if (
        default_quality == "standard"
        and user_doc is not None
        and paying is not None
        and str(user_doc.id) == str(paying.id)
    ):
        user_gen = (user_doc.preferences or {}).get("generation") or {}
        if user_gen.get("default_quality") == "pro":
            default_quality = "pro"

    def _pref(kind: str) -> Optional[str]:
        agent_val = getattr(gen, f"{kind}_model_preference", None) if gen else None
        if agent_val:
            return agent_val
        if user_doc is not None and paying is not None and str(user_doc.id) == str(paying.id):
            return ((user_doc.preferences or {}).get("generation") or {}).get(
                f"{kind}_model_preference"
            )
        return None

    return GenerationAccess(
        paying_user=paying,
        subscriber=subscriber,
        premium_entitled=premium_entitled,
        premium_enabled=premium_enabled,
        default_quality=default_quality,
        image_model_preference=_pref("image"),
        video_model_preference=_pref("video"),
    )


def resolve_model_preference(
    args: dict, access: GenerationAccess, kind: str
) -> Optional[str]:
    """Precedence: request arg > agent setting > paying-user preference > None."""
    request_val = (args or {}).get("model_preference")
    if request_val:
        return str(request_val).lower()
    return getattr(access, f"{kind}_model_preference", None)
