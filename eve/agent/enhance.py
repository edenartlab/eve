import wave
from jinja2 import Template
from eve.agent.llm import async_prompt
from eve.agent.thread import UserMessage
from pydantic import BaseModel, Field



style_guide_prompt = Template("""<PromptGuide>
The following is a guide on how to structure and write good text prompts.

{{prompt_guide}}
                                 
</PromptGuide>""")

user_prompt = Template("""<UserPrompt>
Here is the user's prompt:
                               
{{prompt}}

Now convert this prompt to fit the style guide.
</UserPrompt>""")


async def enhance_prompt(prompt_guide: str, prompt: str):
    try:
        class PromptText(BaseModel):
            """An enhanced prompt."""

            prompt: str = Field(
                ...,
                description="Generated prompt conforming to prompting guide.",
            )

        messages = [
            UserMessage(
                content=style_guide_prompt.render(
                    prompt_guide=prompt_guide
                )
            ),
            UserMessage(
                content=user_prompt.render(
                    prompt=prompt
                )
            ),
        ]

        result = await async_prompt(
            messages=messages,
            system_message=f"You are a prompt engineer who enhances prompts. Given a style guide, you convert possibly faulty prompts into better prompts.",
            model="gpt-4o-mini",
            # model="claude-3-7-sonnet-latest",
            response_model=PromptText,
        )
        print("--d 3124234sff", result.prompt)

        return result.prompt
    
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return prompt
