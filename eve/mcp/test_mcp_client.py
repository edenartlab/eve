import asyncio
from fastmcp import Client


# HTTP server
client = Client("http://localhost:8888/mcp")


async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        for tool in tools:
            print(f"Tool: {tool.name}, description: {tool.description}")


asyncio.run(main())
