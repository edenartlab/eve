import eve
import json
import litellm

def get_current_weather(location, unit="fahrenheit"):
    """Mock weather function"""
    # This is a mock function - in reality you'd call a weather API
    weather_data = {
        "San Francisco": {"temperature": 68, "condition": "sunny"},
        "Tokyo": {"temperature": 75, "condition": "cloudy"},
        "Paris": {"temperature": 65, "condition": "rainy"}
    }
    
    city_weather = weather_data.get(location, {"temperature": 70, "condition": "unknown"})
    return json.dumps({
        "location": location,
        "temperature": city_weather["temperature"],
        "unit": unit,
        "condition": city_weather["condition"]
    })

def main():
    # Enable debug mode and parameter modification
    litellm._turn_on_debug()
    litellm.modify_params = True
    
    model = "anthropic/claude-haiku-4-5"  # works across Anthropic, Bedrock, Vertex AI
    
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        }
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    print("=== Making first completion call ===")
    response = litellm.completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
        reasoning_effort="low",
    )
    
    print("Response\n", response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Check for thinking blocks
    if hasattr(response_message, 'reasoning_content') and response_message.reasoning_content:
        print(f"\n=== REASONING CONTENT ===")
        print(response_message.reasoning_content)
        print("========================\n")
    
    if hasattr(response_message, 'thinking_blocks') and response_message.thinking_blocks:
        print(f"\n=== THINKING BLOCKS ===")
        for i, block in enumerate(response_message.thinking_blocks):
            print(f"Block {i+1}:")
            print(f"  Type: {block.get('type', 'unknown')}")
            if 'thinking' in block:
                print(f"  Thinking: {block['thinking'][:200]}..." if len(block['thinking']) > 200 else f"  Thinking: {block['thinking']}")
        print("=======================\n")
    
    print("Expecting there to be tool calls")
    if not tool_calls:
        print("No tool calls found!")
        return
    
    assert len(tool_calls) > 0  # this has to call the function for SF, Tokyo and paris
    
    # Step 2: check if the model wanted to call a function
    print(f"tool_calls: {tool_calls}")
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        
        messages.append(response_message)  # extend conversation with assistant's reply
        print("Response message\n", response_message)
        
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name not in available_functions:
                # the model called a function that does not exist in available_functions - don't try calling anything
                print(f"Function {function_name} not available!")
                return
            
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        
        print(f"messages: {messages}")
        
        print("\n=== Making second completion call ===")
        second_response = litellm.completion(
            model=model,
            messages=messages,
            seed=22,
            reasoning_effort="low",
            # tools=tools,  # commenting out tools for final response
            drop_params=True,
        )  # get a new response from the model where it can see the function response
        
        print("second response\n", second_response)
        
        # Check for thinking blocks in second response too
        second_message = second_response.choices[0].message
        if hasattr(second_message, 'reasoning_content') and second_message.reasoning_content:
            print(f"\n=== SECOND RESPONSE REASONING CONTENT ===")
            print(second_message.reasoning_content)
            print("========================================\n")
        
        if hasattr(second_message, 'thinking_blocks') and second_message.thinking_blocks:
            print(f"\n=== SECOND RESPONSE THINKING BLOCKS ===")
            for i, block in enumerate(second_message.thinking_blocks):
                print(f"Block {i+1}:")
                print(f"  Type: {block.get('type', 'unknown')}")
                if 'thinking' in block:
                    print(f"  Thinking: {block['thinking'][:200]}..." if len(block['thinking']) > 200 else f"  Thinking: {block['thinking']}")
            print("======================================\n")

if __name__ == "__main__":
    main()