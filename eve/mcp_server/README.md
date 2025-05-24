# Eden MCP Server

This module provides a Model Context Protocol (MCP) server that exposes Eden's tools to external clients. It allows third-party applications to use Eden's creative AI tools through a standardized interface.

## Overview

The MCP server provides:
- **Tool Discovery**: List all available Eden tools with their schemas
- **Tool Execution**: Execute tools with proper authentication and manna management
- **Task Monitoring**: Check the status of running tasks
- **Cost Management**: Track manna costs for tool usage

## Architecture

### Components

1. **`server.py`** - Core MCP server implementation using FastMCP
2. **`modal_app.py`** - Modal deployment configuration for cloud hosting
3. **`client_example.py`** - Example client demonstrating usage
4. **`README.md`** - This documentation

### Flow

```
External Client → MCP Server → Eden Tools → Modal Apps → Results
                     ↓
               API Key Validation
                     ↓
               Manna Balance Check
                     ↓
               Tool Execution
```

## Features

### Authentication & Authorization
- API key validation using Eve's existing auth system
- User manna balance checking before tool execution
- Rate limiting through existing Eve infrastructure

### Tool Management
- Automatic discovery of available tools from YAML configs and MongoDB
- Schema generation for tool parameters
- Support for all Eve tool types (ComfyUI, Modal, Replicate, etc.)

### Execution Models
- **Synchronous**: Execute tool and wait for completion
- **Asynchronous**: Start tool and poll for status
- **Streaming**: Real-time updates (future enhancement)

## Available MCP Tools

### `execute_tool(tool_name, arguments, api_key)`
Execute an Eden tool with the provided arguments.

**Parameters:**
- `tool_name`: Name of the tool to execute (e.g., "flux_schnell", "websearch")
- `arguments`: Tool-specific parameters as a dictionary
- `api_key`: Valid Eden API key

**Returns:**
```json
{
  "status": "completed|failed|running",
  "result": {...},
  "task_id": "task_id_string",
  "cost": 50
}
```

### `list_available_tools(api_key)`
Get a list of all available tools.

**Returns:**
```json
{
  "status": "success",
  "tools": [
    {
      "key": "tool_name",
      "name": "Display Name",
      "description": "Tool description",
      "output_type": "image|video|text|audio",
      "cost_estimate": "50",
      "parameters": {...}
    }
  ],
  "count": 42
}
```

### `get_tool_schema(tool_name, api_key)`
Get the complete schema for a specific tool.

**Returns:**
```json
{
  "status": "success",
  "tool_name": "tool_name",
  "schema": {
    "name": "tool_name",
    "description": "Tool description",
    "input_schema": {
      "type": "object",
      "properties": {...},
      "required": [...]
    }
  }
}
```

### `check_task_status(task_id, api_key)`
Check the status of a previously submitted task.

**Returns:**
```json
{
  "status": "success",
  "task_id": "task_id",
  "task_status": "completed|failed|running",
  "result": {...},
  "cost": 50,
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Usage Examples

### Basic Usage

```python
from eve.mcp_server.client_example import EdenMCPClient

# Initialize client
client = EdenMCPClient("your-api-key")

# List available tools
tools = await client.list_tools()

# Execute a tool
result = await client.execute_tool("websearch", {
    "query": "latest AI news",
    "num_results": 5
})
```

### Direct Modal Usage

```python
from eve.mcp_server.modal_app import execute_tool_call

# Execute tool via Modal
result = await execute_tool_call.remote(
    "flux_schnell",
    {"prompt": "a beautiful sunset", "aspect_ratio": "16:9"},
    "your-api-key"
)
```

## Deployment

### Local Development

1. Install dependencies:
```bash
pip install "mcp[cli]"
```

2. Run the server:
```bash
python -m eve.mcp_server.server
```

### Modal Deployment

1. Deploy to Modal:
```bash
modal deploy eve/mcp_server/modal_app.py
```

2. The server will be available at the Modal endpoint URL.

### Integration with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "eden-tools": {
      "command": "python",
      "args": ["-m", "eve.mcp_server.server"],
      "env": {
        "EDEN_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Security Considerations

1. **API Key Protection**: Never expose API keys in logs or error messages
2. **Rate Limiting**: Existing Eve rate limits apply to MCP requests
3. **Input Validation**: All tool parameters are validated before execution
4. **User Isolation**: Users can only access their own tasks and results

## Error Handling

The MCP server handles various error conditions:

- **Invalid API Key**: Returns authentication error
- **Insufficient Manna**: Returns balance error before execution
- **Tool Not Found**: Returns list of available tools
- **Invalid Parameters**: Returns parameter validation errors
- **Tool Execution Errors**: Returns detailed error information

## Monitoring & Logging

- All tool executions are logged with user context
- Sentry integration for error tracking
- Langfuse integration for request tracing
- Modal metrics for performance monitoring

## Future Enhancements

1. **Streaming Support**: Real-time updates for long-running tools
2. **Batch Operations**: Execute multiple tools in parallel
3. **Tool Composition**: Chain multiple tools together
4. **Custom Tool Registration**: Allow users to register custom tools
5. **WebSocket Support**: Real-time bidirectional communication

## Contributing

When adding new tools or modifying the MCP server:

1. Ensure tools are properly exposed via `visible` and `active` flags
2. Add appropriate parameter validation
3. Test with the example client
4. Update this documentation

## Support

For issues or questions:
- Check the Eve documentation
- Review existing tool configurations
- Test with the provided example client
- Contact the Eden team for API key issues