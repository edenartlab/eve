#!/usr/bin/env python3
"""
Generate Mintlify documentation for Eden tools from their api.yaml files.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def get_param_type_display(param: Dict[str, Any]) -> str:
    """Get the display type for a parameter."""
    param_type = param.get("type", "string")

    # Handle array types
    if param_type == "array":
        item_type = param.get("items", {}).get("type", "string")
        return f"array<{item_type}>"

    return param_type


def format_choices(choices: List[Any]) -> str:
    """Format choices list for display."""
    if isinstance(choices, list):
        return ", ".join([f"`{choice}`" for choice in choices])
    return str(choices)


def generate_param_field(
    name: str, param: Dict[str, Any], is_required: bool = False
) -> str:
    """Generate a ParamField MDX component for a parameter."""
    param_type = get_param_type_display(param)
    lines = []

    # Build the opening tag with attributes
    attrs = [f'body="{name}"', f'type="{param_type}"']

    if is_required:
        attrs.append("required")

    default = param.get("default")
    if default is not None:
        if isinstance(default, bool):
            attrs.append(f"default={{{str(default).lower()}}}")
        elif isinstance(default, (int, float)):
            attrs.append(f"default={{{default}}}")
        else:
            attrs.append(f'default="{default}"')

    lines.append(f"<ParamField {' '.join(attrs)}>")

    # Add description
    description = param.get("description", "")
    if description:
        lines.append(f"  {description}")

    # Add additional details
    details = []

    # Add label if different from name
    label = param.get("label")
    if label and label != name:
        details.append(f"Label: {label}")

    # Add choices
    choices = param.get("choices")
    if choices:
        details.append(f"Options: {format_choices(choices)}")

    # Add min/max for numeric types
    if param.get("minimum") is not None:
        details.append(f"Minimum: {param['minimum']}")
    if param.get("maximum") is not None:
        details.append(f"Maximum: {param['maximum']}")

    # Add tip if present
    tip = param.get("tip")
    if tip:
        lines.append("")
        lines.append("  <Tip>")
        for tip_line in tip.strip().split("\n"):
            lines.append(f"    {tip_line}")
        lines.append("  </Tip>")

    # Add visible_if condition
    visible_if = param.get("visible_if")
    if visible_if:
        lines.append("")
        lines.append("  <Note>")
        lines.append(f"    Only visible when `{visible_if}`")
        lines.append("  </Note>")

    # Add any additional details as bullet points
    if details:
        lines.append("")
        for detail in details:
            lines.append(f"  - {detail}")

    lines.append("</ParamField>")

    return "\n".join(lines)


def format_tool_name(tool_name: str) -> str:
    """Convert snake_case folder name to Title Case."""
    # Replace underscores with spaces and capitalize each word
    words = tool_name.replace("_", " ").split()
    formatted_words = []
    for word in words:
        # Handle special cases and acronyms
        if word.lower() in ["api", "url", "id", "ai", "tts", "3d", "sdxl", "lora"]:
            formatted_words.append(word.upper())
        elif word.lower() == "openai":
            formatted_words.append("OpenAI")
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def generate_tool_doc(tool_name: str, api_data: Dict[str, Any]) -> str:
    """Generate MDX documentation for a single tool."""
    lines = []

    # Use formatted folder name for title
    display_name = format_tool_name(tool_name)

    # Front matter
    lines.append("---")
    lines.append(f'title: "{display_name}"')

    description = api_data.get("description", "")
    if description:
        # Escape quotes in description
        description = description.replace('"', '\\"')
        lines.append(f'description: "{description}"')

    lines.append("---")
    lines.append("")

    # Overview section
    lines.append("## Overview")
    lines.append("")
    if description:
        lines.append(description)
        lines.append("")

    # Tool details
    details = []

    if api_data.get("output_type"):
        details.append(f"- **Output Type**: {api_data['output_type']}")

    if api_data.get("base_model"):
        details.append(f"- **Base Model**: {api_data['base_model']}")

    if api_data.get("cost_estimate"):
        details.append(f"- **Estimated Cost**: {api_data['cost_estimate']} credits")

    if api_data.get("handler"):
        details.append(f"- **Handler**: {api_data['handler']}")

    if details:
        lines.extend(details)
        lines.append("")

    # Parameters section
    parameters = api_data.get("parameters", {})
    if parameters:
        lines.append("## Parameters")
        lines.append("")

        # Separate required and optional parameters
        required_params = []
        optional_params = []

        for name, param in parameters.items():
            if param.get("required", False):
                required_params.append((name, param))
            else:
                optional_params.append((name, param))

        # Required parameters
        if required_params:
            lines.append("### Required Parameters")
            lines.append("")
            for name, param in required_params:
                lines.append(generate_param_field(name, param, is_required=True))
                lines.append("")

        # Optional parameters
        if optional_params:
            lines.append("### Optional Parameters")
            lines.append("")
            for name, param in optional_params:
                lines.append(generate_param_field(name, param, is_required=False))
                lines.append("")

    # Additional metadata
    metadata = []

    if not api_data.get("active", True):
        metadata.append("**Status**: Inactive")

    if not api_data.get("visible", True):
        metadata.append("**Visibility**: Hidden")

    if metadata:
        lines.append("## Metadata")
        lines.append("")
        lines.extend(metadata)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mintlify documentation for Eden tools"
    )
    parser.add_argument(
        "tools", nargs="*", help="List of tool names to generate docs for"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate documentation for all tools"
    )
    parser.add_argument(
        "--eve-dir", default="../eve", help="Path to eve repository (default: ../eve)"
    )
    parser.add_argument(
        "--docs-dir",
        default="../mintlify-docs",
        help="Path to mintlify-docs repository (default: ../mintlify-docs)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.absolute()
    eve_dir = (script_dir / args.eve_dir).resolve()
    docs_dir = (script_dir / args.docs_dir).resolve()

    tools_dir = eve_dir / "eve" / "tools"
    docs_tools_dir = docs_dir / "tools"

    # Validate directories exist
    if not tools_dir.exists():
        print(f"Error: Tools directory not found: {tools_dir}", file=sys.stderr)
        sys.exit(1)

    if not docs_dir.exists():
        print(f"Error: Docs directory not found: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    # Create tools documentation directory if it doesn't exist
    docs_tools_dir.mkdir(exist_ok=True)

    # Determine which tools to process
    if args.tools:
        # Process specified tools
        tool_names = args.tools
    elif args.all or (not args.tools):
        # Process all tools with api.yaml files (default behavior or --all flag)
        tool_names = []
        for tool_path in tools_dir.iterdir():
            if tool_path.is_dir():
                api_file = tool_path / "api.yaml"
                if api_file.exists():
                    tool_names.append(tool_path.name)
        print(f"Found {len(tool_names)} tools with api.yaml files")
    else:
        print(
            "Error: No tools specified. Use --all to generate all tools or provide tool names.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Process each tool
    generated_docs = []
    skipped_tools = []
    for tool_name in tool_names:
        api_file = tools_dir / tool_name / "api.yaml"

        if not api_file.exists():
            print(f"Warning: No api.yaml found for tool '{tool_name}'", file=sys.stderr)
            continue

        try:
            with open(api_file, "r") as f:
                api_data = yaml.safe_load(f)

            # Skip tools that are inactive or invisible
            if not api_data.get("active", True):
                skipped_tools.append(f"{tool_name} (inactive)")
                continue

            if not api_data.get("visible", True):
                skipped_tools.append(f"{tool_name} (invisible)")
                continue

            # Generate documentation
            doc_content = generate_tool_doc(tool_name, api_data)

            # Write to file
            doc_file = docs_tools_dir / f"{tool_name}.mdx"
            with open(doc_file, "w") as f:
                f.write(doc_content)

            print(f"Generated documentation for '{tool_name}'")
            generated_docs.append(tool_name)

        except Exception as e:
            print(f"Error processing tool '{tool_name}': {e}", file=sys.stderr)

    # Show skipped tools if any
    if skipped_tools:
        print(f"\nSkipped {len(skipped_tools)} tool(s):")
        for tool in skipped_tools:
            print(f"  - {tool}")

    # Update docs.json navigation if docs were generated
    if generated_docs:
        docs_json_file = docs_dir / "docs.json"

        try:
            with open(docs_json_file, "r") as f:
                docs_config = yaml.safe_load(f)

            # Group tools by output type
            tools_by_type = {}
            for tool_name in generated_docs:
                api_file = tools_dir / tool_name / "api.yaml"
                try:
                    with open(api_file, "r") as f:
                        api_data = yaml.safe_load(f)
                    output_type = api_data.get("output_type", "other")
                    if output_type not in tools_by_type:
                        tools_by_type[output_type] = []
                    tools_by_type[output_type].append(tool_name)
                except Exception:
                    # If we can't read the file, put it in 'other'
                    if "other" not in tools_by_type:
                        tools_by_type["other"] = []
                    tools_by_type["other"].append(tool_name)

            # Sort tools within each type
            for output_type in tools_by_type:
                tools_by_type[output_type].sort()

            # Define the order and display names for output types
            type_order = [
                ("image", "Image Generation"),
                ("video", "Video Generation"),
                ("audio", "Audio Generation"),
                ("text", "Text Processing"),
                ("string", "Text & Data Tools"),
                ("lora", "Model Training"),
                ("model", "Model Training"),
                ("3d", "3D Generation"),
                ("other", "Other Tools"),
            ]

            # Build groups for the Tools tab
            groups = []
            for output_type, display_name in type_order:
                if output_type in tools_by_type:
                    groups.append(
                        {
                            "group": display_name,
                            "pages": [
                                f"tools/{name}" for name in tools_by_type[output_type]
                            ],
                        }
                    )

            # Add any remaining output types not in our predefined order
            remaining_types = set(tools_by_type.keys()) - set(
                [t[0] for t in type_order]
            )
            for output_type in sorted(remaining_types):
                # Capitalize and format the output type for display
                display_name = output_type.replace("_", " ").title()
                groups.append(
                    {
                        "group": display_name,
                        "pages": [
                            f"tools/{name}" for name in tools_by_type[output_type]
                        ],
                    }
                )

            # Find or create Tools tab
            tools_tab = None
            for tab in docs_config["navigation"]["tabs"]:
                if tab.get("tab") == "Tools":
                    tools_tab = tab
                    break

            if not tools_tab:
                # Create new Tools tab
                tools_tab = {"tab": "Tools", "groups": groups}
                # Insert before API reference
                api_index = next(
                    (
                        i
                        for i, tab in enumerate(docs_config["navigation"]["tabs"])
                        if tab.get("tab") == "API reference"
                    ),
                    len(docs_config["navigation"]["tabs"]),
                )
                docs_config["navigation"]["tabs"].insert(api_index, tools_tab)
            else:
                # Update existing tab with new groups
                tools_tab["groups"] = groups

            # Write back to docs.json
            import json

            with open(docs_json_file, "w") as f:
                json.dump(docs_config, f, indent=2)
                f.write("\n")

            print(f"\nUpdated docs.json with {len(generated_docs)} tool(s)")

        except Exception as e:
            print(f"Warning: Could not update docs.json: {e}", file=sys.stderr)

    print(f"\nSuccessfully generated documentation for {len(generated_docs)} tool(s)")


if __name__ == "__main__":
    main()
