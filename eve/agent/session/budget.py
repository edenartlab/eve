from typing import Optional

from eve.agent.session.models import Session, SessionBudget


def check_session_budget(session: Session):
    if session.budget:
        if session.budget.token_budget:
            if session.budget.tokens_spent >= session.budget.token_budget:
                raise ValueError("Session token budget exceeded")
        if session.budget.manna_budget:
            if session.budget.manna_spent >= session.budget.manna_budget:
                raise ValueError("Session manna budget exceeded")
        if session.budget.turn_budget:
            if session.budget.turns_spent >= session.budget.turn_budget:
                raise ValueError("Session turn budget exceeded")


def update_session_budget(
    session: Session,
    tokens_spent: Optional[int] = None,
    manna_spent: Optional[float] = None,
    turns_spent: Optional[int] = None,
):
    if session.budget:
        budget = session.budget
        if tokens_spent:
            budget.tokens_spent += tokens_spent
        if manna_spent:
            budget.manna_spent += manna_spent
        if turns_spent:
            budget.turns_spent += turns_spent
        session.update(budget=budget.model_dump())
        session.budget = SessionBudget(**session.budget)
