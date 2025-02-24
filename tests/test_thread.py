import json
from eve.agent.thread import Thread
from eve.agent.llm import title_thread

def test_title_thread():
    """Pytest entry point"""

    thread = Thread.from_mongo("67660123b01077573b056e33")
    results = title_thread(thread=thread)

    print(json.dumps(results, indent=2))
