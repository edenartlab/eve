from eve.auth import get_my_eden_user
from eve.agent.session.session import Session, SessionMessage
from eve.agent.dispatch import dispatch


def test_dispatch():
    user = get_my_eden_user()
    session = Session.load("test-session")
    new_message = SessionMessage(
        sender_id=user.id,
        content="Who can tell me more about Juicebox?"
    )
    result = dispatch(session, new_message)
    print(result)


test_dispatch()
