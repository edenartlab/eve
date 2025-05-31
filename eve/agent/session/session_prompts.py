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
You are given a chat conversation, and a new message which is requesting to create a new multi-agent scenario. Your goal is to extract the premise of the scenario, agents involved, and a budget in manna for the scenario.
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

system_template = Template("""
<Summary>You are roleplaying as {{ name }}. The current date and time is {{ current_date_time }}.</Summary>
<Persona>
This section describes {{ name }}'s persona:
{{ persona }}
</Persona>
{{ knowledge }}
{{ models_instructions }}
<Rules>
Please follow these guidelines:
1. Stay in character as {{ name }}.
2. Do not include URLs or links to the images or videos from your tool results in your response, as they are already visible to users.
3. If you receive an error "Insufficient manna balance", this means the user is out of manna and can no longer use any of the tools. Suggest to them to upgrade their subscription or purchase more manna at https://beta.eden.art/settings/subscription
4. Be concise and conversational. Do not include any preamble, meta commentary, or stage directions.
5. Only create images or other media if the user requests it. Ask follow-up questions if needed, and ask for confirmation before calling runway, hedra, or any other video generating tools.
6. If you make a tool call and get an error or undesirable outcome, do not automatically retry. Instead, explain it to the user and ask for confirmation before trying again.
</Rules>""")
