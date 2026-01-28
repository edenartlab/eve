import os
import random
from typing import Literal

import instructor
from elevenlabs.client import ElevenLabs
from openai import OpenAI

from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))


def get_short_name(full_name: str) -> str:
    """Extract the short name from a full voice name like 'George - Warm, Captivating Storyteller'."""
    if " - " in full_name:
        return full_name.split(" - ")[0]
    return full_name


async def handler(context: ToolContext):
    args = context.args
    description = args.get("description")
    gender = args.get("gender")
    exclude = args.get("exclude", [])

    response = eleven.voices.get_all()
    voices = list(response.voices)
    random.shuffle(voices)

    if gender:
        if gender not in ["male", "female"]:
            raise ValueError("Gender must be either 'male' or 'female'")
        voices = [v for v in voices if v.labels and v.labels.get("gender") == gender]

    if exclude:
        exclude_lower = [name.lower() for name in exclude]
        voices = [v for v in voices if v.name and v.name.lower() not in exclude_lower]

    if not voices:
        raise ValueError("No voices found matching the specified criteria")

    if not description:
        selected_voice = random.choice(voices)
        return {"output": get_short_name(selected_voice.name or "")}

    client = instructor.from_openai(OpenAI())
    voice_names = [v.name for v in voices]

    def format_voice(v):
        parts = [v.name]
        if v.labels:
            label_str = ", ".join(f"{k}={val}" for k, val in v.labels.items() if val)
            if label_str:
                parts.append(f"[{label_str}]")
        if v.description:
            parts.append(f'"{v.description}"')
        return " ".join(parts)

    voice_descriptions = "\n".join([format_voice(v) for v in voices])

    prompt = f"""You are given the following list of voices and their descriptions.

---
{voice_descriptions}
---

You are given the following description of a desired character:

---
{description}
---

Select the voice that best matches the description of the character."""

    selected_voice = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Literal[tuple(voice_names)],  # type: ignore
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at selecting the right voice for a character.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return {"output": get_short_name(selected_voice)}
