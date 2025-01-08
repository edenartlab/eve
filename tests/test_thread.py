import json
from eve.thread import Thread
from eve.llm import title_thread

def test_title_thread():
    """Pytest entry point"""

    thread = Thread.from_mongo("67660123b01077573b056e33")
    results = title_thread(thread=thread)

    print(json.dumps(results, indent=2))
