# Discord DM Tool

The `discord_broadcast_dm` tool enables agents to send direct messages to all recently active users from a specific Discord channel. This tool is designed for community engagement, daily check-ins, and personalized outreach.

## Overview

The tool workflow:

1. **User Discovery**: Scans a Discord channel for users active within a specified timeframe
2. **User Mapping**: Maps Discord users to Eden users (creates shadow users for new Discord users)
3. **Session Management**: Creates or reuses individual DM sessions for each user with consistent `session_key` pattern
4. **Message Generation**: Injects instructions into each DM session for personalized responses
5. **DM Delivery**: Sends generated responses as direct messages via enhanced `discord_post` tool

## Key Features

- **Consistent Session Management**: Uses `dm_{agent_id}_{discord_user_id}` pattern for session keys
- **Parallel Processing**: Handles multiple DM sessions concurrently for efficiency
- **Shadow User Creation**: Automatically creates Eden shadow users for Discord-only users
- **Activity-Based Filtering**: Only DMs users who were active within specified timeframe
- **Comprehensive Error Handling**: Graceful handling of blocked DMs, missing users, etc.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `channel_id` | string | Yes | - | Discord channel ID to find active users from |
| `instruction` | string | Yes | - | Instructions for what the agent should say/do in each DM |
| `active_days` | integer | No | 3 | Days to look back for user activity (1-30) |
| `message_limit` | integer | No | 50 | Max messages to check per user (1-1000) |

## Usage Examples

### Basic Daily Check-in

```yaml
# Via trigger configuration
instruction: "Send a good morning message to all active users from #general"
parameters:
  channel_id: "123456789012345678"
  instruction: "Say good morning, ask how they're doing, and check if they need help with any ongoing projects"
  active_days: 3
```

### Weekly Follow-up

```yaml
# For support channel follow-ups
parameters:
  channel_id: "876543210987654321"
  instruction: "Follow up on any outstanding support tickets, ask if their issues were resolved, and see if they need additional help"
  active_days: 7
  message_limit: 100
```

### Community Updates

```yaml
# For project updates
parameters:
  channel_id: "555666777888999000"
  instruction: "Share the latest project updates, highlight new features, and ask for feedback on recent changes"
  active_days: 5
```

## Session Management

### DM Session Keys

DM sessions use the pattern: `dm_{agent_id}_{discord_user_id}`

This ensures:
- Consistent session retrieval across multiple days
- Separate conversation history for each user
- Memory persistence between DM interactions

### Session Creation vs Reuse

- **New Sessions**: Created when no existing DM session exists for the user
- **Existing Sessions**: Reused to maintain conversation continuity and memory
- **User Access**: Sessions are restricted to the specific Eden user

## Enhanced Discord Post Tool

The `discord_post` tool has been enhanced to support DM sending:

### New Parameters

- `discord_user_id`: Discord user ID to send DM to (takes priority over `channel_id`)
- `channel_id`: Now optional when `discord_user_id` is provided

### Usage in DM Sessions

The agent receives clear instructions to use the discord_post tool:

```
After generating your response, use the discord_post tool to send it as a DM to this user by setting discord_user_id to "{discord_user_id}".
```

## Error Handling

### Common Scenarios

1. **DMs Disabled**: User has disabled DMs or blocked the bot
2. **User Not Found**: Discord user no longer exists
3. **Rate Limiting**: Discord API rate limits exceeded
4. **Invalid Permissions**: Bot lacks necessary permissions

### Error Response Format

```json
{
  "output": [
    {
      "discord_user": "username",
      "eden_user": "discord_username",
      "status": "failed",
      "error": "Cannot send DM to user: DMs disabled or bot blocked"
    }
  ]
}
```

## Success Response Format

```json
{
  "output": [
    {
      "discord_user": "username",
      "eden_user": "discord_username",
      "status": "success",
      "session_id": "507f1f77bcf86cd799439011",
      "dm_sent": true
    }
  ]
}
```

## Prerequisites

1. **Discord Deployment**: Agent must have a valid Discord deployment configured
2. **Bot Permissions**: Discord bot needs DM permissions and channel read access
3. **Channel Access**: Bot must have access to the specified channel
4. **Eden Admin Key**: Required for session creation via API

## Limitations

1. **Discord API Limits**: Subject to Discord's rate limiting (typically 5 DMs per 5 seconds)
2. **User Privacy**: Cannot DM users who have blocked the bot or disabled DMs
3. **Historical Data**: Limited to Discord's message history retention
4. **Memory Usage**: Large numbers of active users may consume significant memory

## Best Practices

1. **Rate Limiting**: Consider Discord's DM rate limits when setting up frequent triggers
2. **User Consent**: Ensure users are aware they may receive DMs from the bot
3. **Content Quality**: Provide clear, specific instructions for personalized messages
4. **Error Monitoring**: Monitor error rates to identify permission or access issues
5. **Session Management**: Let the tool handle session reuse for conversation continuity

## Integration with Triggers

The discord_broadcast_dm tool works seamlessly with Eden's trigger system:

```yaml
# Daily morning check-in trigger
schedule:
  hour: 9
  minute: 0
  timezone: "America/New_York"

instruction: |
  Use the discord_broadcast_dm tool to send personalized good morning messages to all users
  who were active in our #general channel in the last 3 days. Ask how they're doing
  and if they need help with any ongoing projects.
```

This enables automated, personalized community engagement at scale.