import openai
import os
import json
import tempfile
from ... import utils

async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    """
    Handles audio transcription using the OpenAI API.
    """
    # OpenAI client will automatically use the OPENAI_API_KEY environment variable.
    # If not set, an openai.AuthenticationError will be raised upon API call.
    client = openai.AsyncOpenAI()

    temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    file_path_str = utils.download_file(args["audio"], temp_audio.name, overwrite=True)
    selected_model_arg = args.get("model", "gpt-4o-transcribe")
    enable_timestamps = args.get("use_timestamps", False)
    prompt_text = args.get("prompt")
    api_call_params = {}

    if enable_timestamps:
        # Timestamps require whisper-1 and verbose_json format
        api_call_params["model"] = "whisper-1"
        api_call_params["response_format"] = "verbose_json"
        api_call_params["timestamp_granularities"] = [args.get("timestamp_granularity", "segment")]
    else:
        api_call_params["model"] = selected_model_arg
        # "json" format is supported by gpt-4o models and whisper-1,
        # and returns {"text": "..."}
        api_call_params["response_format"] = "json"

    if prompt_text:
        api_call_params["prompt"] = prompt_text
    
    # The 'file' parameter must be a file object opened in binary read mode.
    with open(file_path_str, "rb") as audio_file_obj:
        transcription_response = await client.audio.transcriptions.create(
            file=audio_file_obj,
            **api_call_params
        )

    # transcription_response is a Pydantic model from the OpenAI library.
    # .model_dump() converts it to a dictionary.
    # - For response_format="json", this dict is like {"text": "Transcription text..."}
    # - For response_format="verbose_json", this dict contains segments, words with timestamps, etc.
    output_data = transcription_response.model_dump() 
    
    return {
        "output": json.dumps(output_data) # Return the full JSON structure as a string
    } 