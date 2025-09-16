import json
import requests
import html as html_module
from datetime import datetime
from pathlib import Path
from bson import ObjectId

from ..mongo import get_collection
from ..agent import Agent
from ..user import User
from ..task import Task
from . import data_utils


def export_user_data(
    username: str = None, 
    agentname: str = None,
    export_dir: Path = Path(".")
):
    """Export user data to a folder with JSON and HTML files
    
    Args:
        username: The username to export data for (if None, uses current user)
        agentname: Optional agent name to filter sessions by (only export sessions containing this agent)
    """
    
    # Get current user if username not provided
    if not username:
        from ..auth import get_my_eden_user
        user = get_my_eden_user()
        username = user.username
    
    # Load user by username to get user_id
    user = User.load(username)
    if not user:
        raise ValueError(f"User '{username}' not found")
    user_id = str(user.id)
    
    # Load agent by name if provided
    agent_id = None
    if agentname:
        agent = Agent.load(agentname)
        if not agent:
            raise ValueError(f"Agent '{agentname}' not found")
        agent_id = str(agent.id)

    print("Exporting user data for", username, "with", agentname if agentname else "all agents")
    
    # Create export directory
    agent_suffix = f"_agent_{agentname}" if agentname else ""
    export_dir = export_dir / f"export_{username}{agent_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    sessions_collection = get_collection("sessions")
    messages_collection = get_collection("messages")
    
    # Build query to get sessions
    query = {"owner": ObjectId(user_id)}
    if agent_id:
        # Only get sessions where the agent appears in the agents list
        query["agents"] = ObjectId(agent_id)
    
    # Get filtered sessions
    all_sessions = list(sessions_collection.find(query))
    
    # Prepare data for JSON export
    export_data = {
        "username": username,
        "user_id": user_id,
        "export_date": datetime.now().isoformat(),
        "sessions": []
    }
    
    if agentname:
        export_data["agentname"] = agentname
        export_data["agent_id"] = agent_id
        export_data["filter"] = f"Sessions containing agent {agentname}"
    
    # Cache for agent names
    agent_names_cache = {}
    
    def get_agent_name(agent_id):
        """Get agent name from cache or load from DB"""
        if agent_id not in agent_names_cache:
            try:
                agent = Agent.from_mongo(agent_id)
                agent_names_cache[agent_id] = agent.name if agent else "Unknown Agent"
            except:
                agent_names_cache[agent_id] = "Unknown Agent"
        return agent_names_cache[agent_id]
    
    # Prepare HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eden Export - {username}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .session {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .session-header {{
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .session-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .session-meta {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .message {{
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            background-color: #f9f9f9;
        }}
        .message-user {{
            background-color: #e3f2fd;
        }}
        .message-assistant {{
            background-color: #f5f5f5;
        }}
        .message-system {{
            background-color: #fff3e0;
        }}
        .message-header {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #555;
        }}
        .message-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .attachments {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e0e0e0;
        }}
        .attachment-link {{
            display: inline-block;
            margin-right: 10px;
            margin-bottom: 5px;
            color: #1976d2;
            text-decoration: none;
        }}
        .attachment-link:hover {{
            text-decoration: underline;
        }}
        .tool-calls {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .tool-call {{
            margin-bottom: 10px;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
        }}
        .tool-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .tool-status {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        .status-completed {{
            background-color: #c8e6c9;
            color: #2e7d32;
        }}
        .status-failed {{
            background-color: #ffcdd2;
            color: #c62828;
        }}
        .status-pending {{
            background-color: #fff9c4;
            color: #f57c00;
        }}
        .tool-result {{
            margin-top: 8px;
            padding: 8px;
            background-color: #f5f5f5;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <h1>{export_title}</h1>
    <h3>Compiled {export_date}</h3>
    <hr>
""".format(
        username=username,
        export_title=f"Export of {username} sessions with {agentname}" if agentname else f"Export of all {username} sessions",
        export_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Process each session
    for session in all_sessions:
        session_id = session["_id"]
        
        # Get all messages for this session (excluding eden and tool roles)
        messages = list(messages_collection.find({
            "session": session_id,
            "role": {"$nin": ["eden", "tool"]}
        }).sort("createdAt", 1))
        
        # Get tool messages separately for merging with tool calls
        tool_messages = list(messages_collection.find({
            "session": session_id,
            "role": "tool"
        }))
        
        # Create a map of tool messages by their name (tool call id)
        tool_message_map = {}
        for tm in tool_messages:
            if tm.get("name"):
                tool_message_map[tm["name"]] = tm
        
        # Skip sessions with no messages after filtering
        if not messages:
            continue
        
        # Prepare session data for JSON
        session_data = {
            "session_id": str(session_id),
            "title": session.get("title", "Untitled Session"),
            "status": session.get("status", "active"),
            "created_at": session.get("createdAt", "").isoformat() if session.get("createdAt") else None,
            "updated_at": session.get("updatedAt", "").isoformat() if session.get("updatedAt") else None,
            "agents": [str(agent_id) for agent_id in session.get("agents", [])],
            "messages": []
        }
        
        # Add session to HTML
        escaped_title = html_module.escape(session.get("title") or "Untitled Session")
        html_content += f"""
    <div class="session">
        <div class="session-header">
            <div class="session-title"><a href="https://staging.app.eden.art/sessions/{str(session_id)}" target="_blank">{escaped_title}</a></div>
            <div class="session-meta">
                Created: {session.get("createdAt", "").strftime("%Y-%m-%d %H:%M:%S") if session.get("createdAt") else "Unknown"}<br>
            </div>
        </div>
"""
        
        # Process each message
        for message in messages:
            # Prepare message data for JSON
            message_data = {
                "message_id": str(message["_id"]),
                "role": message.get("role", ""),
                "content": message.get("content", ""),
                "sender": str(message.get("sender")) if message.get("sender") else None,
                "sender_name": message.get("sender_name", ""),
                "created_at": message.get("createdAt", "").isoformat() if message.get("createdAt") else None,
                "attachments": message.get("attachments", []),
                "tool_calls": []
            }
            
            # Process tool calls
            tool_calls = message.get("tool_calls") or []
            for tc in tool_calls:
                result = tc.get("result", [])
                result = data_utils.prepare_result(result)
                tool_call_data = {
                    "id": tc.get("id", ""),
                    "tool": tc.get("tool", ""),
                    "args": tc.get("args", {}),
                    "status": tc.get("status", ""),
                    "cost": tc.get("cost"),
                    "error": tc.get("error"),
                    "result": result  # Include full results
                }
                
                # Merge tool message content if available
                if tc.get("id") in tool_message_map:
                    tool_msg = tool_message_map[tc.get("id")]
                    if tool_msg.get("content"):
                        try:
                            tool_content = json.loads(tool_msg["content"])
                            # Override with tool message content if it has more detailed results
                            if isinstance(tool_content, dict) and tool_content.get("result"):
                                tool_call_data["result"] = tool_content.get("result", [])
                                tool_call_data["status"] = tool_content.get("status", tc.get("status", ""))
                        except:
                            pass  # Keep original if parsing fails
                
                message_data["tool_calls"].append(tool_call_data)
            
            session_data["messages"].append(message_data)
            
            # Add message to HTML
            role_class = f"message-{message.get('role', 'user')}"
            escaped_content = html_module.escape(message.get("content", ""))
            escaped_sender_name = html_module.escape(message.get('sender_name', '')) if message.get('sender_name') else ""
            
            # Get display name for the message
            role = message.get("role", "user")
            if role == "assistant" and message.get("sender"):
                display_role = get_agent_name(message.get("sender"))
            elif role == "user":
                display_role = username
            else:
                display_role = role.upper()
            
            html_content += f"""
        <div class="message {role_class}">
            <div class="message-header">
                {display_role} {f"- {escaped_sender_name}" if escaped_sender_name else ""}
                <span class="timestamp">{message.get("createdAt", "").strftime("%Y-%m-%d %H:%M:%S") if message.get("createdAt") else ""}</span>
            </div>
            <div class="message-content">{escaped_content}</div>
"""
            
            # Add attachments if any
            if message.get("attachments"):
                html_content += '            <div class="attachments"><strong>Attachments:</strong><br>'
                for attachment in message.get("attachments", []):
                    html_content += f'                <a href="{attachment}" class="attachment-link">{attachment.split("/")[-1]}</a><br>'
                html_content += "            </div>\n"
            
            # Add tool calls if any
            if tool_calls:
                html_content += '            <div class="tool-calls"><strong>Tool Calls:</strong>'
                for tc in tool_calls:
                    status_class = f"status-{tc.get('status', 'pending')}"
                    html_content += f"""
                <div class="tool-call">
                    <span class="tool-name">{tc.get("tool", "")}</span>
                    <span class="tool-status {status_class}">{tc.get("status", "pending")}</span>
"""
                    if tc.get("args"):
                        html_content += f"                    <div>Args: <code>{json.dumps(tc.get('args', {}), indent=2, default=str)}</code></div>\n"
                    if tc.get("error"):
                        html_content += f'                    <div style="color: red;">Error: {tc.get("error", "")}</div>\n'
                    if tc.get("result") or tc.get("id") in tool_message_map:
                        html_content += '                    <div class="tool-result">'
                        
                        # Check if there's a corresponding tool message
                        if tc.get("id") in tool_message_map:
                            tool_msg = tool_message_map[tc.get("id")]
                            if tool_msg.get("content"):
                                try:
                                    # Parse the tool message content (it's JSON)
                                    tool_content = json.loads(tool_msg["content"])
                                    
                                    # Display the parsed content nicely
                                    if isinstance(tool_content, dict):
                                        if tool_content.get("status") == "completed" and tool_content.get("result"):
                                            # Display the actual result as formatted JSON
                                            results = tool_content.get("result", [])
                                            html_content += '<pre style="background: #f5f5f5; padding: 8px; border-radius: 4px; overflow-x: auto;">'
                                            html_content += html_module.escape(json.dumps(results, indent=2, default=str))
                                            html_content += '</pre>'
                                        elif tool_content.get("error"):
                                            html_content += f'<div style="color: red;">Error: {html_module.escape(tool_content.get("error", ""))}</div>'
                                        else:
                                            # Show the full tool content as JSON
                                            html_content += '<pre style="background: #f5f5f5; padding: 8px; border-radius: 4px; overflow-x: auto;">'
                                            html_content += html_module.escape(json.dumps(tool_content, indent=2, default=str))
                                            html_content += '</pre>'
                                    else:
                                        # Fallback for non-dict content
                                        html_content += f'<div>{html_module.escape(str(tool_content))}</div>'
                                except:
                                    # If parsing fails, show raw content
                                    html_content += f'<div>{html_module.escape(tool_msg.get("content", ""))}</div>'
                        else:
                            # Fallback to showing result data if no tool message
                            result_data = tc.get("result", [])
                            if result_data:
                                # Display the JSON result prettily
                                html_content += '<pre style="background: #f5f5f5; padding: 8px; border-radius: 4px; overflow-x: auto;">'
                                result_data_html = json.dumps(result_data, indent=2, default=str)
                                html_content += html_module.escape(result_data_html)
                                html_content += '</pre>'
                            else:
                                html_content += 'No results'
                        
                        html_content += '</div>\n'
                    html_content += "                </div>"
                html_content += "            </div>\n"
            
            html_content += "        </div>\n"
        
        html_content += "    </div>\n"
        export_data["sessions"].append(session_data)
    
    html_content += """
</body>
</html>
"""
    
    # Write JSON file
    json_path = export_dir / "user_data_export.json"
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)
    
    # Write HTML file
    html_path = export_dir / "user_data_export.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Export completed successfully!")
    print(f"Export directory: {export_dir}")
    print(f"- JSON file: {json_path}")
    print(f"- HTML file: {html_path}")
    
    return export_dir


def export_agent_creations(
    username: str = None,
    agentname: str = None,
    export_dir: Path = Path(".")
):
    """Export agent creations

    Args:
        agentname: The agent name to export creations for
        export_dir: The directory to export the creations to
    """

    if not agentname:
        return

    # Get current user if username not provided
    if not username:
        from ..auth import get_my_eden_user
        user = get_my_eden_user()
        username = user.username
    
    # Load user by username to get user_id
    user = User.load(username)
    if not user:
        raise ValueError(f"User '{username}' not found")
    user_id = str(user.id)
    
    # Load agent by name if provided
    agent = Agent.load(agentname)
    if not agent:
        raise ValueError(f"Agent '{agentname}' not found")
    
    print(f"Exporting Agent creations for {agentname}")
    
    # Create export directory
    export_dir = export_dir / f"{agentname}_creations"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Get filtered creatiobs
    creations_collection = get_collection("creations3")
    query = {"agent": agent.id, "user": ObjectId(user_id)}

    all_creations = list(creations_collection.find(query))

    for creation in all_creations:
        id = str(creation["_id"])
        created_at = creation["createdAt"].strftime("%Y-%m-%d_%H%M")
        url = data_utils.prepare_result(creation).get("url")
        task = Task.from_mongo(creation["task"])
        creation["task"] = task.model_dump()
        ext = creation["filename"].split(".")[-1]
        if url:
            response = requests.get(url)
            with open(export_dir / f"{created_at}_{id}.{ext}", "wb") as f:
                f.write(response.content)
            with open(export_dir / f"{created_at}_{id}.json", "w") as f:
                json.dump(creation, f, indent=2, default=str)

    print(f"Export completed successfully!")
    print(f"Export directory: {export_dir}")