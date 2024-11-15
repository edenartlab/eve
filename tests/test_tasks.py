from test_tools import *

async def run_test(tool):
    user_id = os.getenv("EDEN_TEST_USER_STAGE")
    task = await tool.async_start_task(user_id, tool.test_args, "STAGE")
    result = await tool.async_wait(task)
    if "error" in result:
        eden_utils.pprint(f"Tool: {tool.key}: ERROR {result['error']}", color="red")
    else:
        eden_utils.pprint(f"Tool: {tool.key}:", result, color="green")
    return result

async def run_all_tests():
    tools = get_tools_from_dir("tools", env="STAGE")
    tools.update(get_tools_from_dir("../../workflows", env="STAGE"))
    # tools.update(get_tools("../../private_workflows"))
    
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    print(f"Testing tools: {', '.join(tools.keys())}")

    results = await asyncio.gather(*[run_test(tool) for tool in tools.values()])    
    
    if args.save:
        save_results(tools, results)

    return results

if __name__ == "__main__":
    asyncio.run(run_all_tests())
