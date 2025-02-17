import json
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def analyze_conversations(input_file, output_file, plot_file):
    # Initialize tool usage counter
    tool_usage = defaultdict(int)
    
    # Read and parse JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Process each conversation thread
        for thread in conversations:
            # Write thread separator and title only
            out.write('\n' + '='*5 + '\n')
            out.write(f"Title: {thread.get('title', 'Untitled')}\n")
            out.write('-'*5 + '\n\n')
            
            # Process messages in the thread
            for msg in thread['messages']:
                # Write message content with just the role, skip empty assistant messages
                role = msg['role'].upper()
                content = msg['content']
                if role == 'ASSISTANT' and not content.strip():
                    continue
                
                # Start with the role
                out.write(f"{role}:\n")
                
                # If it's an assistant message with tool calls, add tool information
                if role == 'ASSISTANT' and 'tool_calls' in msg and msg['tool_calls']:
                    for tool_call in msg['tool_calls']:
                        if tool_call.get('tool'):
                            out.write(f"tool_call: {tool_call['tool']}\n")
                    out.write("\n")  # Add separator line after tool calls
                
                # Write the actual message content
                out.write(f"{content}\n\n")
                
                # Count tool usage if present
                if 'tool_calls' in msg and msg['tool_calls']:
                    for tool_call in msg['tool_calls']:
                        if tool_call.get('tool'):
                            tool_usage[tool_call['tool']] += 1

    # Create sorted bar chart of tool usage
    if tool_usage:
        # Sort tools by usage count
        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
        tools, counts = zip(*sorted_tools)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(tools)), counts)
        
        # Customize the chart
        plt.xticks(range(len(tools)), tools, rotation=45, ha='right')
        plt.xlabel('Tool Name')
        plt.ylabel('Number of Uses')
        plt.title('Distribution of Tool Usage in Conversations')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        
        # Print tool usage summary
        print("\nTool Usage Summary:")
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"{tool}: {count} uses")

if __name__ == "__main__":
    input_file = "eden-prod.threads3.json"
    output_file = "conversations.txt"
    plot_file = "tools.png"
    
    try:
        analyze_conversations(input_file, output_file, plot_file)
        print(f"\nAnalysis complete! Check {output_file} for conversation logs and {plot_file} for tool usage visualization.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")