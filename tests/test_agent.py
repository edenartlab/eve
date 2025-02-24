from eve.agent.agent import Agent

def test_agent():
    """Pytest entry point"""

    agent = Agent.load("banny")
    tools = agent.get_tools()
    print("the agent's tools are: ", tools.keys())

    tool = tools["flux_dev_lora"]
    result = tool.run({
        "prompt": "banny in a hotel"
    })
    print(result)