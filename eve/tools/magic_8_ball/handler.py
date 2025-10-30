import random
from ...agent import Agent
from ...agent.session.models import ChatMessage, LLMContext, LLMConfig
from ...agent.session.session_llm import async_prompt
from ...tool import ToolContext
from loguru import logger


async def handler(context: ToolContext):
    """
    Magic Eightball handler that randomly samples from categories and generates constrained content.

    This tool implements the Magic Eightball prompting system to combat diversity collapse
    by forcing external randomization outside the LLM's probability distributions.
    """

    # Extract parameters
    sample_dict = context.args.get("sample_dict", {})
    prompt_context = context.args.get("context", "")
    instruction = context.args.get("instruction", "")
    agent_override = context.args.get("agent")  # Optional agent from argument

    if not sample_dict:
        return {"error": "sample_dict is required"}
    if not prompt_context:
        return {"error": "context is required"}
    if not instruction:
        return {"error": "instruction is required"}

    # Step 1: Random sampling from each category
    sampled_values = {}
    for category, options in sample_dict.items():
        if options and isinstance(options, list):
            sampled_values[category] = random.choice(options)

    if not sampled_values:
        return {"error": "No valid categories with options found in sample_dict"}

    # Step 2: Create formatted output of selections
    samples_text = "\n".join(
        [f"• {category}: {value}" for category, value in sampled_values.items()]
    )

    # Step 3: Set up system prompt (with agent persona if available)
    system_prompt = """You are a creative assistant that generates content strictly constrained by mandatory elements. You must naturally incorporate ALL specified elements into your response."""

    # Use agent persona if provided (either from args or session context)
    agent = None
    if agent_override:
        try:
            agent = Agent.from_mongo(agent_override)
        except Exception as e:
            logger.error(
                f"Warning: Could not load agent from override '{agent_override}': {e}"
            )
    elif context.agent:
        try:
            agent = Agent.from_mongo(context.agent)
        except Exception as e:
            logger.error(
                f"Warning: Could not load agent from session '{context.agent}': {e}"
            )

    if agent:
        system_prompt = f"""You are {agent.name}. The following is a description of your persona.
        # Your persona

        {agent.persona}
        
        # Task
        
        You are a creative assistant that generates content strictly constrained by mandatory elements. You must naturally incorporate ALL specified elements into your response. Let your persona influence your creative direction, voice, and style. Stay true to your character while fulfilling this creative task."""

    # Step 4: Create inner prompt with constraints
    inner_user = f"""
CONTEXT:
{prompt_context}

MANDATORY ELEMENTS (you MUST incorporate ALL of these features into your response):
{samples_text}

INSTRUCTIONS:
{instruction}

The mandatory elements should feel integral to the story, not just mentioned in passing.
"""

    # Step 5: Call LLM with constraints
    try:
        ctx = LLMContext(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=inner_user),
            ],
            config=LLMConfig(
                model="claude-haiku-4-5",
                fallback_models=["claude-haiku-4-5", "gpt-4o"],
                reasoning_effort="medium",
            ),
        )

        response = await async_prompt(ctx)
        generated_content = response.content.strip()

    except Exception as e:
        return {"error": f"Content generation failed: {str(e)}"}

    # Step 6: Format final response
    result = f"""🎱 EIGHTBALL SELECTIONS:
{samples_text}

📝 GENERATED CONTENT:
{generated_content}"""

    return {
        "output": result,
        "intermediate_outputs": {
            "sampled_values": sampled_values,
            "raw_content": generated_content,
        },
    }
