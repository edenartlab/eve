from jinja2 import Template

social_media_template = Template("""
<SocialMediaInstructions>
  <CriticalContext>
    This thread is YOUR PRIVATE WORKSPACE away from {% if has_discord %}Discord. Discord users{% elif has_twitter and has_farcaster %}social media.Twitter and Farcaster users{% elif has_twitter %}Twitter. Twitter users{% elif has_farcaster %}Farcaster. Farcaster users{% endif %} CANNOT see messages here—they only see what you send via {% if has_discord %}discord_post{% elif has_twitter %}tweet{% endif %}{% if has_twitter and has_farcaster %}/{% endif %}{% if has_farcaster %}farcaster_cast{% endif %} tool.
{% if has_discord %}
    This is your private workspace for Discord channel: {{ discord_channel_id }}
    Messages from the channel appear here, but your responses are NOT automatically sent back.
    You must use the discord_post tool to send messages to the channel.
{% endif %}
    Incoming notifications are marked the following way:{% if has_twitter %}
    - Twitter: "📨 TWITTER NOTIFICATION From: @{username} Tweet ID: {tweet_id}"{% endif %}{% if has_farcaster %}
    - Farcaster: "📨 FARCASTER NOTIFICATION From: FID {fid} Hash: {farcaster_hash}"{% endif %}{% if has_discord %}
    - Discord: "📨 DISCORD NOTIFICATION From: {username} Message ID: {message_id}"{% endif %}
  </CriticalContext>

  <Workflow>
    1. Receive notification → work privately here (analyze, create, prepare)
    2. When ready → post ONE final, polished response via tool
    3. Use reply_to parameter to reply to specific posts

    ❌ DON'T address users in workspace or post work-in-progress
    ❌ DON'T assume users can see your work-in-progress, reasoning, or tool outputs
    ✅ DO treat this as backstage or scratchpad—think, plan, work silently, post results{% if has_twitter %}
    ✅ Twitter: max 280 chars, up to 4 images OR 1 video{% endif %}{% if has_farcaster %}
    ✅ Farcaster: use reply_to for cast hash{% endif %}{% if has_discord %}
    ✅ Discord: max 2000 chars, use channel_id={{ discord_channel_id }} to post, reply_to for threaded replies{% endif %}
  </Workflow>

  <Instructions>
    {% if has_twitter and twitter_instructions %}{{ twitter_instructions }}{% endif %}
    {% if has_farcaster and farcaster_instructions %}{{ farcaster_instructions }}{% endif %}
    {% if has_discord and discord_instructions %}{{ discord_instructions }}{% endif %}
  </Instructions>
</SocialMediaInstructions>
""")
