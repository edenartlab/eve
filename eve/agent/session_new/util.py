from eve.agent.session.models import Session, PromptSessionContext
from eve.agent.session_new.budget import check_session_budget


def validate_prompt_session(session: Session, context: PromptSessionContext):
    if session.status == "archived":
        raise ValueError("Session is archived")
    has_budget = session.budget and (
        session.budget.token_budget
        or session.budget.manna_budget
        or session.budget.turn_budget
    )
    if has_budget:
        check_session_budget(session)
