import logging
import os
import time
import uuid
import json
import litellm
from litellm import acompletion, aresponses
from typing import Callable, List, AsyncGenerator, Optional

from eve.agent.thread import ChatMessage
from eve.agent.session.models import (
    LLMContext,
    LLMConfig,
    LLMContextMetadata,
    LLMResponse,
    LLMTraceMetadata,
    ToolCall,
)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


if os.getenv("LANGFUSE_TRACING_ENVIRONMENT"):
    litellm.success_callback = ["langfuse"]

supported_models = [
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",


    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]


class ToolMetadataBuilder:
    def __init__(
        self,
        tool_name: str,
        litellm_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.litellm_session_id = litellm_session_id
        self.tool_name = tool_name
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id

    def __call__(self) -> LLMContextMetadata:
        return LLMContextMetadata(
            session_id=self.litellm_session_id,
            trace_name=f"TOOL_{self.tool_name}",
            generation_name=f"TOOL_{self.tool_name}",
            trace_metadata=LLMTraceMetadata(
                user_id=str(self.user_id),
                agent_id=str(self.agent_id),
                session_id=str(self.session_id),
            ),
        )


def validate_input(context: LLMContext) -> None:
    if context.config.model not in supported_models:
        raise ValueError(f"Model {context.config.model} is not supported")


def construct_observability_metadata(context: LLMContext):
    if not context.metadata:
        return {}
    metadata = {
        "session_id": context.metadata.session_id,
        "trace_id": context.metadata.trace_id,
        "trace_name": context.metadata.trace_name,
        "generation_name": context.metadata.generation_name,
        "generation_id": context.metadata.generation_id,
    }
    if context.metadata.trace_metadata:
        metadata["trace_metadata"] = context.metadata.trace_metadata.model_dump()
        metadata["trace_user_id"] = context.metadata.trace_metadata.user_id
    return metadata


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    tools = context.tools or {}

    tools = [
        tool.openai_schema(exclude_hidden=True) 
        for tool in tools.values()
    ]

    # Gemini/Vertex: enum values must be strings and parameter type must be "string"
    if "gemini" in context.config.model or "vertex" in context.config.model:
        for tool in tools:
            params = (
                tool.get("function", {}).get("parameters", {}).get("properties", {})
            )
            for param_def in params.values():
                if "enum" in param_def:
                    param_def["enum"] = [str(val) for val in param_def["enum"]]
                    param_def["type"] = "string"

    return tools


async def async_run_tool_call(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
):
    tool = llm_context.tools[tool_call.tool]
    task = await tool.async_start_task(
        user_id=user_id,
        agent_id=agent_id,
        args=tool_call.args,
        mock=False,
        public=public,
        is_client_platform=is_client_platform,
    )

    result = await tool.async_wait(task)

    # Add task.cost and task.id to the result object
    if isinstance(result, dict):
        result["cost"] = getattr(task, "cost", None)
        result["task"] = getattr(task, "id", None)

    return result


def add_anthropic_cache_control(messages: List[dict]) -> List[dict]:
    """
    Add cache control for Anthropic models to optimize multi-turn conversations.

    - System messages are cached as static prefix
    - Second-to-last user message is cached as checkpoint
    - Final message is cached for continuation in follow-ups
    """
    # Add cache control to system message (static prefix)
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i]["cache_control"] = {"type": "ephemeral"}
            break

    # Find the last user message and second-to-last user message
    user_message_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_message_indices.append(i)

    # Mark second-to-last user message for caching (checkpoint)
    if len(user_message_indices) >= 2:
        messages[user_message_indices[-2]]["cache_control"] = {"type": "ephemeral"}

    # Mark the final message for continuing in followups
    if len(messages) > 0:
        messages[-1]["cache_control"] = {"type": "ephemeral"}

    return messages


def prepare_messages(
    messages: List[ChatMessage], model: Optional[str] = None
) -> List[dict]:
    messages = [schema for msg in messages for schema in msg.openai_schema()]

    # Add Anthropic cache control for models that support it
    if model and ("claude" in model or "anthropic" in model):
        messages = add_anthropic_cache_control(messages)

    return messages



async def async_prompt_litellm(
    context: LLMContext,
) -> LLMResponse:
    if "openai/" in context.config.model and context.config.reasoning_effort:
        return await async_prompt_litellm_responses(context)
    else:
        return await async_prompt_litellm_completion(context)



async def async_prompt_litellm_responses(
    context: LLMContext,
) -> LLMResponse:
    print(f"🧠 [DEBUG] RESPONSES !!!! Context CONFIG: {context.config}")
    
    messages = prepare_messages(context.messages, context.config.model)
    tools = construct_tools(context)

    response_kwargs = {
        "model": context.config.model,
        "input": messages,
        # "metadata": construct_observability_metadata(context),
        # "tools": tools,
        # "response_format": context.config.response_format,
        "reasoning": {"effort": context.config.reasoning_effort, "summary": "detailed"},
        # "fallbacks": context.config.fallback_models,    
        # "drop_params": True,
        # "num_retries": 2,
        # "timeout": 600,
        # "context_window_fallback_dict": {
        #     "gpt-5-mini": "gpt-5",
        #     "gpt-5-nano": "gpt-5",
        # },
    }

    print(f"🧠 [DEBUG] RESPONSE_KWARGS: {response_kwargs}")
    
    if context.config.reasoning_effort:
        response_kwargs["reasoning"] = {"effort": context.config.reasoning_effort, "summary": "detailed"}
    
    print(f"🧠 [DEBUG] RESPONSE_KWARGS2: {response_kwargs}")

    # Use finalized reasoning_effort from config if available
    if context.config.reasoning_effort:
        response_kwargs["reasoning"] = {
            "effort": context.config.reasoning_effort, 
            "summary": "detailed"
        }
        
        # Check if model supports reasoning
        supports_reasoning = litellm.supports_reasoning(model=context.config.model)
        print(f"🧠 [REASONING] Model {context.config.model} supports reasoning: {supports_reasoning}")
    
    logging.info(f"Attempting responses with model: {context.config.model}, fallbacks: {context.config.fallback_models}, reasoning_effort: {context.config.reasoning_effort}")
    
    try:
        t0 = time.time()
        print("start...", response_kwargs.get("reasoning"))
        print(".... ok 1")
        response = await aresponses(**response_kwargs)        
        print(".... ok 2")
        t1 = time.time()
        print(f"response done in {t1-t0} seconds")
        
        actual_model = getattr(response, "model", context.config.model)
        
        if actual_model != context.config.model and context.config.fallback_models:
            logging.info("Actual model used: %s", actual_model)
            
    except Exception as e:
        logging.error(f"All models failed. Error: {str(e)}")
        raise

    print(f"🧠 [DEBUG] RESPONSE!!!!: {response}")


    # tool_calls = []
    
    # # add web search as a tool call
    # psf = getattr(response.choices[0].message, "provider_specific_fields", None)
    # if psf:
    #     citations = psf.get("citations") or []
    #     sources = []
    #     for citation_block in citations:
    #         for citation in citation_block:
    #             source = {
    #                 "title": citation.get('title'),
    #                 "url": citation.get('url'),
    #             }
    #             if not source in sources:  # avoid duplicates
    #                 sources.append(source)
    #     if sources:
    #         tool_calls.append(
    #             ToolCall(
    #                 id=f"toolu_{uuid.uuid4()}",
    #                 tool="web_search",
    #                 args={},
    #                 result=sources,
    #                 status="completed",
    #             )
    #         )

    # # add regular tool calls
    # if response.choices[0].message.tool_calls:
    #     tool_calls.extend([
    #         ToolCall(
    #             id=tool_call.id,
    #             tool=tool_call.function.name,
    #             args=json.loads(tool_call.function.arguments),
    #             status="pending",
    #         )
    #         for tool_call in response.choices[0].message.tool_calls
    #     ])

    # # Extract thinking blocks if present
    # # Handle both thinking_blocks (Anthropic) and reasoning_content (other providers)
    # thought = None
    # message = response.choices[0].message
    

    # print(f"🧠 HERE IS THE MESSAGE@!!: {message}")
    # print(f"🧠 [DEBUG] Message attributes: {dir(message)}")
    # print(f"🧠 [DEBUG] Has reasoning_content: {hasattr(message, 'reasoning_content')}")
    # print(f"🧠 [DEBUG] Has thinking_blocks: {hasattr(message, 'thinking_blocks')}")
    
    # # Check raw response for debugging
    # if hasattr(response, '_raw') or hasattr(response, 'raw'):
    #     print(f"🧠 [DEBUG] Raw response available for inspection")
    
    # # Check for any other reasoning-related attributes
    # reasoning_attrs = [attr for attr in dir(message) if 'reason' in attr.lower() or 'think' in attr.lower()]
    # if reasoning_attrs:
    #     print(f"🧠 [DEBUG] Reasoning-related attributes found: {reasoning_attrs}")


    # # Check for Anthropic thinking_blocks first
    # if hasattr(message, 'thinking_blocks') and message.thinking_blocks and len(message.thinking_blocks) > 0:
    #     thought = message.thinking_blocks
    #     print(f"🧠 [THINKING] Anthropic thinking_blocks found: {len(message.thinking_blocks)} blocks")
    #     for i, block in enumerate(message.thinking_blocks):
    #         print(f"🧠 [THINKING] Block {i+1}: {block.get('thinking', '')[:200]}...")
        
    #     seconds = t1 - t0
    #     if seconds < 60:
    #         thought[0]["title"] = f"Thought for {seconds:.0f} seconds"
    #     else:
    #         thought[0]["title"] = f"Thought for {round(seconds/60)} minutes"
    
    # # Check for reasoning_content from other providers
    # elif hasattr(message, 'reasoning_content') and message.reasoning_content:
    #     print(f"🧠 [REASONING] reasoning_content found ({len(message.reasoning_content)} chars)")
    #     print(f"🧠 [REASONING] Content preview: {message.reasoning_content[:500]}...")
        
    #     # Convert reasoning_content to thinking_blocks format for consistency
    #     seconds = t1 - t0
    #     title = f"Reasoning for {seconds:.0f} seconds" if seconds < 60 else f"Reasoning for {round(seconds/60)} minutes"
        
    #     thought = [{
    #         "type": "reasoning",
    #         "thinking": message.reasoning_content,
    #         "title": title
    #     }]

    return LLMResponse(
        content=response.choices[0].message.content or "",
        tool_calls=None, #tool_calls or None,
        stop=response.choices[0].finish_reason,
        tokens_spent=response.usage.total_tokens,
        thought=None #thought,
    )




async def async_prompt_litellm_completion(
    context: LLMContext,
) -> LLMResponse:
    
    print(f"🧠 [DEBUG] COMPLETION !!!! Context CONFIG: {context.config}")
    
    messages = prepare_messages(context.messages, context.config.model)
    tools = construct_tools(context)

    completion_kwargs = {
        "model": context.config.model,
        "messages": messages,
        "metadata": construct_observability_metadata(context),
        "tools": tools,
        "response_format": context.config.response_format,
        "fallbacks": context.config.fallback_models,
        "drop_params": True,
        "num_retries": 2,
        "timeout": 600,
        "context_window_fallback_dict": {
            "gpt-4o-mini": "gpt-4o",
            "gpt-5-mini": "gpt-5",
            "gpt-5-nano": "gpt-5",
            "gemini-2.5-flash": "gemini-2.5-pro",
            "gemini-2.5-flash-lite": "gemini-2.5-flash",
        },
    }

    # add web search options for Anthropic models
    # todo: does this fail in fallback models?
    if "claude" in context.config.model:
        completion_kwargs["web_search_options"] = {
            "search_context_size": "medium"
        }
    
    # Use finalized reasoning_effort from config if available
    if context.config.reasoning_effort:
        completion_kwargs["reasoning_effort"] = context.config.reasoning_effort
        
        # Check if model supports reasoning
        supports_reasoning = litellm.supports_reasoning(model=context.config.model)
        print(f"🧠 [REASONING] Model {context.config.model} supports reasoning: {supports_reasoning}")
    
    logging.info(f"Attempting completion with model: {context.config.model}, fallbacks: {context.config.fallback_models}, reasoning_effort: {context.config.reasoning_effort}")
    
    try:
        t0 = time.time()
        print("start...", completion_kwargs.get("reasoning_effort"))
        response = await acompletion(**completion_kwargs)        
        t1 = time.time()
        print(f"response done in {t1-t0} seconds")
        
        actual_model = getattr(response, "model", context.config.model)
        
        if actual_model != context.config.model and context.config.fallback_models:
            logging.info("Actual model used: %s", actual_model)
            
    except Exception as e:
        logging.error(f"All models failed. Error: {str(e)}")
        raise

    tool_calls = []
    
    # add web search as a tool call
    psf = getattr(response.choices[0].message, "provider_specific_fields", None)
    if psf:
        citations = psf.get("citations") or []
        sources = []
        for citation_block in citations:
            for citation in citation_block:
                source = {
                    "title": citation.get('title'),
                    "url": citation.get('url'),
                }
                if not source in sources:  # avoid duplicates
                    sources.append(source)
        if sources:
            tool_calls.append(
                ToolCall(
                    id=f"toolu_{uuid.uuid4()}",
                    tool="web_search",
                    args={},
                    result=sources,
                    status="completed",
                )
            )

    # add regular tool calls
    if response.choices[0].message.tool_calls:
        tool_calls.extend([
            ToolCall(
                id=tool_call.id,
                tool=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
                status="pending",
            )
            for tool_call in response.choices[0].message.tool_calls
        ])

    # Extract thinking blocks if present
    # Handle both thinking_blocks (Anthropic) and reasoning_content (other providers)
    thought = None
    message = response.choices[0].message
    

    print(f"🧠 HERE IS THE MESSAGE@!!: {message}")
    print(f"🧠 [DEBUG] Message attributes: {dir(message)}")
    print(f"🧠 [DEBUG] Has reasoning_content: {hasattr(message, 'reasoning_content')}")
    print(f"🧠 [DEBUG] Has thinking_blocks: {hasattr(message, 'thinking_blocks')}")
    
    # Check raw response for debugging
    if hasattr(response, '_raw') or hasattr(response, 'raw'):
        print(f"🧠 [DEBUG] Raw response available for inspection")
    
    # Check for any other reasoning-related attributes
    reasoning_attrs = [attr for attr in dir(message) if 'reason' in attr.lower() or 'think' in attr.lower()]
    if reasoning_attrs:
        print(f"🧠 [DEBUG] Reasoning-related attributes found: {reasoning_attrs}")


    # Check for Anthropic thinking_blocks first
    if hasattr(message, 'thinking_blocks') and message.thinking_blocks and len(message.thinking_blocks) > 0:
        thought = message.thinking_blocks
        print(f"🧠 [THINKING] Anthropic thinking_blocks found: {len(message.thinking_blocks)} blocks")
        for i, block in enumerate(message.thinking_blocks):
            print(f"🧠 [THINKING] Block {i+1}: {block.get('thinking', '')[:200]}...")
        
        seconds = t1 - t0
        if seconds < 60:
            thought[0]["title"] = f"Thought for {seconds:.0f} seconds"
        else:
            thought[0]["title"] = f"Thought for {round(seconds/60)} minutes"
    
    # Check for reasoning_content from other providers
    elif hasattr(message, 'reasoning_content') and message.reasoning_content:
        print(f"🧠 [REASONING] reasoning_content found ({len(message.reasoning_content)} chars)")
        print(f"🧠 [REASONING] Content preview: {message.reasoning_content[:500]}...")
        
        # Convert reasoning_content to thinking_blocks format for consistency
        seconds = t1 - t0
        title = f"Reasoning for {seconds:.0f} seconds" if seconds < 60 else f"Reasoning for {round(seconds/60)} minutes"
        
        thought = [{
            "type": "reasoning",
            "thinking": message.reasoning_content,
            "title": title
        }]

    return LLMResponse(
        content=response.choices[0].message.content or "",
        tool_calls=tool_calls or None,
        stop=response.choices[0].finish_reason,
        tokens_spent=response.usage.total_tokens,
        thought=thought,
    )


async def async_prompt_stream_litellm(
    context: LLMContext,
) -> AsyncGenerator[str, None]:
    # Todo: if we use stream again, add web search here like we do in async_prompt_litellm
    response = await litellm.acompletion(
        model=context.config.model,
        messages=prepare_messages(context.messages, context.config.model),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
        stream=True,
        response_format=context.config.response_format,
    )
    async for part in response:
        yield part


DEFAULT_LLM_HANDLER = async_prompt_litellm
DEFAULT_LLM_STREAM_HANDLER = async_prompt_stream_litellm


async def async_prompt(
    context: LLMContext,
    handler: Optional[Callable[[LLMContext], str]] = DEFAULT_LLM_HANDLER,
) -> LLMResponse:
    validate_input(context)
    response = await handler(context)
    return response


async def async_prompt_stream(
    context: LLMContext,
    handler: Optional[
        Callable[[LLMContext, LLMConfig], AsyncGenerator[str, None]]
    ] = DEFAULT_LLM_STREAM_HANDLER,
) -> AsyncGenerator[str, None]:
    validate_input(context)
    async for chunk in handler(context):
        yield chunk
