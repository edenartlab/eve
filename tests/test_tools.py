import asyncio
import os
import json
from eve.tool import get_tools_from_mongo, get_tools_from_api_files
from eve.app.auth.auth import get_my_eden_user


async def async_run_tool(tool, api: bool, db: str, mock: bool):
    """Run a single tool test"""
    if api:
        user = get_my_eden_user(db=db)
        task = await tool.async_start_task(user.id, user.id, tool.test_args, db=db, mock=mock)
        return await tool.async_wait(task)
    return await tool.async_run(tool.test_args, db=db, mock=mock)

async def async_run_all_tools(
    tools: list[str],
    yaml: bool = False,
    db: str = "STAGE",
    api: bool = False,
    parallel: bool = True,
    mock: bool = True
):
    """Test multiple tools with their test args"""
    # Get tools from either yaml files or mongo
    tool_dict = get_tools_from_api_files(tools=tools, include_inactive=True) if yaml else get_tools_from_mongo(db=db, tools=tools)
    
    # Create and run tasks
    tasks = [async_run_tool(tool, api, db, mock) for tool in tool_dict.values()]
    results = await asyncio.gather(*tasks) if parallel else [await task for task in tasks]
    
    # Collect errors
    errors = [
        f"{tool}: {result['error']}" 
        for tool, result in zip(tool_dict.keys(), results) 
        if "error" in result
    ]
    
    print(f"\n\nTested {len(tool_dict)} tools with {len(errors)} errors:")
    if errors:
        print("\t" + "\n\t".join(errors))
    
    return results

def test_tools():
    """Pytest entry point"""

    # Test from mongo
    results = asyncio.run(async_run_all_tools(
        tools=[
            "flux_schnell", 
            "txt2img", 
            "legacy_create", 
            "example_tool", 
            "elevenlabs"
        ],
        yaml=False,
        db="STAGE",
        api=False,
        parallel=True,
        mock=True
    ))
    print(json.dumps(results, indent=2))

    # Test from yaml
    results = asyncio.run(async_run_all_tools(
        tools=["legacy_create"],
        yaml=True,
        db="STAGE",
        api=False,
        parallel=True,
        mock=True
    ))
    print(json.dumps(results, indent=2))


