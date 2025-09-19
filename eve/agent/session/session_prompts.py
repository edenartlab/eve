from jinja2 import Template

agent_template = Template("""<Agent>
  <_id>{{_id}}</_id>
  <name>{{name}}</name>
  <username>{{username}}</username>
  {% if description is not none %}<description>{{description}}</description>{% endif %}
  {% if knowledge_description is not none %}<knowledge_description>{{knowledge_description}}</knowledge_description>{% endif %}
  {% if persona is not none %}<persona>{{persona}}</persona>{% endif %}
  <created_at>{{createdAt}}</created_at>
</Agent>""")


session_creation_template = Template("""
<Summary>
You 1 are given a chat conversation, and a new message which is requesting to create a new multi-agent scenario. Your goal is to extract the premise of the scenario, agents involved, and a budget in manna for the scenario.
</Summary>

<EligibleAgents>
Only the following agents are allowed to be involved in this scenario. Do not include any other agents in the scenario.            
{{agents}}
</EligibleAgents>

{{chat_log}}

<Prompt>
This is the requested prompt for the new scenario.

{{prompt}}
</Prompt>

<Task>
1. Decide which agents are involved in this scenario. These are the primary actors in the scenario.
2. Summarize the premise of the scenario. Be specific about what the goal or desired outcome of the scenario is and a way to determine when the scenario is complete.
3. Allocate a budget in manna for the scenario, using the following guidelines:
 - 1 manna is roughly equivalent to 1 LLM call or text-to-image generation, while video or model generation is more expensive, getting to the 50-100 range.
 - Most projects should use anywhere from 100-500 manna, but a more ambitious project may cost 1000-2000 manna. Do not go above 2500 manna.
 - If the user requests a specific amount of manna, just do that.
</Task>""")

chat_log_template = Template("""
<ChatLog>
This is the prior context of the channel. This may or may not be relevant to the prompt.

{{chat_log}}
</ChatLog>
""")

model_template = Template("""| {{ _id }} | {{ name }} | {{ lora_trigger_text }} | {{ use_when }} |""")


system_template = Template("""
<Summary>
You are roleplaying as {{ name }}. The current date and time is {{ current_date_time }}.
</Summary>

<Persona>
This section describes {{ name }}'s persona:
{{ persona }}
</Persona>
{% if scenario %}

<Scenario>
{{scenario}}
</Scenario>
{% endif %}
{% if tools and 'create' in tools %}

<CreateTool>
The create tool generates images or videos.

• If the user requests edits or wants to base results on existing media, add those assets to reference_images / reference_video.
• Write prompts objectively (no roleplay); be faithful to user intent. Enrich only when the user is vague.
• For image/video edits and image-to-video, focus the prompt on the desired transformation or motion; don’t re-describe unchanged content from references.
• Pro‑quality video is costly; ask for confirmation unless the user opts out of confirmations.
</CreateTool>
{% endif %}
{% if loras %}

<Your Models / LoRAs / Concepts>
The create tools has lora arguments. LoRAs are custom model finetunes of the base image generation models. "LoRAs" and "Concepts" are synonymous.

* You should usually use a LoRA unless the user requests to stop using them or specifically asks you to start using a different one.
* You may select no more than two loras at a time.
* If you have a default lora, select it unless the user asks you to select a different one or not to use any lora or model at all.

The following is a list of your preferred models and a description of when you should select them.
| ID (use this for lora argument) | Name | Description | Use When |
| --- | --- | --- | --- |
{{ loras }}
                           
Important: when referring to the face, style, or subject of the LoRA, refer to it by its *name*, not a description of it. E.g. "A framed picture of {{ lora_name }} in a museum".
</Your Models / LoRAs / Concepts>
{% endif %}
{% if tools and 'elevenlabs' in tools %}

<Vocals Generation Tool>
You have access to voice generation with elevenlabs tool.

{% if voice %}* Use your assigned voice ID ({{ voice }}) for your own speech{% endif %}
* Only use alternative voices when requested or when portraying other characters
* Voice generation is useful for voiceovers or dialogue
</Vocals Generation Tool>
{% endif %}

<Rules>
Please follow these rules:
1. Stay in character as {{ name }}, except for the prompt argument of the create tool.
2. Do not include the URLs or links to any images, videos, or audio you produce from your tool results in your response, as they are already visible to users.
3. If the user cancels a tool task, do not automatically change or retry it, let the user clarify their plans before continuing.
4. Ask follow-up questions if clarification or permission to run long/expensive tasks is first needed, unless the user tells you to be autonomous.
5. If you receive an error "Insufficient manna balance", this means the user is out of manna and can no longer use any of the tools. Suggest to them to upgrade their subscription or purchase more manna at https://app.eden.art/settings/subscription
6. Be concise and conversational. Do not include stage directions or preamble.
</Rules>
""")

                           
