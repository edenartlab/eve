from eve.agent import Agent

def test_agent():
    agent = Agent.load("verdelis")
    tools = agent.get_tools()
    print("tools", tools.keys())

    tool = tools["flux_dev_lora"]
    result = tool.run({
        "prompt": "verdelis is in a library"
    }, mock=True)
    print(result)

    assert result["status"] == "completed"