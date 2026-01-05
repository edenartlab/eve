import asyncio
import os
from tempfile import NamedTemporaryFile
from typing import Iterator

from elevenlabs.client import ElevenLabs
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from eve import utils
from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))


async def handler(context: ToolContext):
    prompt = context.args["prompt"]

    if context.args.get("enhance_prompt"):
        try:
            prompt = await asyncio.to_thread(enhance_prompt, prompt)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")

    async def generate_with_params():
        def _generate():
            audio_generator = eleven.text_to_sound_effects.convert(
                text=prompt,
                duration_seconds=context.args["duration"],
            )
            if isinstance(audio_generator, Iterator):
                return b"".join(audio_generator)
            return audio_generator

        return await asyncio.to_thread(_generate)

    audio_generator = await utils.async_exponential_backoff(
        generate_with_params, max_attempts=3, initial_delay=1
    )

    audio = audio_generator

    # save to file
    audio_file = NamedTemporaryFile(delete=False)
    audio_file.write(audio)
    audio_file.close()

    return {
        "output": audio_file.name,
    }


enhancement_prompt = """Your job is to transform a user’s raw request into an optimal prompt for Eleven Music—maximizing musicality and control while staying concise.

GENERAL PRINCIPLES
- Preserve the user’s intent; add missing musical details that make the output controllable.
- Keep the main prompt to 1–3 compact sentences. Add separate timing/lyrics lines only if needed.
- Favor clear, evocative descriptors over verbosity. Simple, high‑impact words are welcome when creative freedom is desired.
- Default behavior: Eleven Music includes vocals unless told otherwise. If the user wants no vocals, explicitly add “instrumental only”.
- Do NOT ask questions; infer sensible defaults.

CONTENT TO INCLUDE (as applicable)
1) Genre & Tone
   - Name the genre(s) and emotional/mood intent (e.g., “eerie, foreboding”, “uplifting, triumphant”).
   - You may fuse genres when requested (e.g., indie rock + soul).

2) Tempo & Key (Musical Control)
   - Include precise tempo or a narrow range (e.g., “130 BPM” or “130–140 BPM”).
   - Include a key if relevant (e.g., “in A minor”). You may omit if not important to the user’s intent.

3) Instrumentation & Stems Strategy
   - Use “solo” before instruments to isolate (e.g., “solo electric guitar”, “solo piano in C minor”).
   - If vocals only: use “a cappella” (e.g., “a cappella female vocals”).
   - If the user wants no vocals: add “instrumental only”.
   - Be musically descriptive (e.g., “dissonant violin screeches over pulsing sub‑bass”, “driving synth arpeggios, punchy drums, distorted bass, glitch effects”).

4) Vocals & Delivery
   - Specify singer count/roles if relevant (e.g., “two singers harmonizing in C”).
   - Describe delivery/timbre (e.g., “raw,” “live,” “breathy,” “gritty,” “aggressive,” “glitching”).
   - If it’s an ad or narration: say “voiceover only” and include the opening script if provided.

5) Structure & Timing
   - You may shape the form (e.g., “rising tension, quick transitions, dynamic energy bursts”).
   - Use explicit timing cues when needed:
     • “lyrics begin at 15 seconds”
     • “instrumental only after 1:45”
   - If the user supplies or requests lyrics, include them under a separate “Lyrics:” block.
   - Do not embed duration unless the user mentioned it; duration is typically controlled by the calling tool.

6) Language & Multilingual
   - If a language is specified or implied, state it (e.g., “Japanese lyrics,” “Spanish voiceover”). Otherwise leave unspecified (defaults to English behavior).

7) Creative Brevity vs. Control
   - When the user wants surprise/creativity, keep descriptors evocative and minimal.
   - When they want precision, include BPM, key, and explicit arrangement/timing.

OUTPUT FORMAT
- Return ONLY the optimized prompt text. If lyrics are present or requested, add a “Lyrics:” block on new lines after the main prompt.
- Examples of formatting patterns you may produce (illustrative only—do not output examples verbatim):
  • “Intense, fast‑paced electronic track for a high‑adrenaline game; driving synth arpeggios, punchy drums, distorted bass, glitch textures; 135–145 BPM in A minor; rising tension with quick transitions and explosive bursts; two singers doing aggressive call‑and‑response.”
  • “High‑end mascara ad: voiceover only; sleek, modern, confident; sparse, glossy electronic bed with subtle percussion; 100 BPM in F minor. Voiceover opens: ‘We bring you the most volumizing mascara yet.’ Brand name at the end.”
  • “Live‑feeling indie rock that fuses alt R&B, gritty soul, and folk; raw, one‑take energy; female lead enters at 0:15; 92 BPM in C major; roomy drums, overdriven guitars, warm bass; chorus blooms, then intimate outro.”

SPECIAL CASES
- If the user asks for stems or isolated parts, use “solo” for instruments and “a cappella” for vocals; do NOT mention ‘stems extraction’.
- If the user requests “no vocals”, include “instrumental only”.
- Respect any provided script/lyrics; do not rewrite brand names or required copy.

FAIL‑SAFES
- Never include meta commentary, apologies, or instructions—only the finished prompt (plus optional “Lyrics:” block).
- Avoid unnecessary jargon; keep language production‑ready."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def enhance_prompt(
    prompt: str = None,
):
    client = OpenAI()
    enhanced_prompt = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": enhancement_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return enhanced_prompt.choices[0].message.content
