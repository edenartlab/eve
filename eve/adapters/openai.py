import json
import os
import magic
from eve.utils import (
    prepare_result, 
    dumps_json, 
    download_file, 
    image_to_base64
)


def get_tool_schema(tool_call):
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.tool, 
            "arguments": json.dumps(tool_call.args)
        },
    }


def get_tool_result_schema(tool_call, truncate_images=False):
    content = {"status": tool_call.status}

    if tool_call.status == "failed":
        content["error"] = tool_call.error
    else:
        content["result"] = prepare_result(tool_call.result)

    return {
        "role": "tool",
        "name": tool_call.tool,
        "content": dumps_json(content),
        "tool_call_id": tool_call.id,
    }


def get_message_schema(message, truncate_images=False):
    # System Message
    if message.role == "system":
        return [
            {
                "role": "system",
                "content": message.content,
            }
        ]

    # User Message
    elif message.role == "user":
        return [
            {
                "role": "user",
                "content": _get_content_block(message, truncate_images=truncate_images),
                **({"name": message.name} if message.name else {}),
            }
        ]

    # Assistant Message
    else:
        schema = [
            {
                "role": "assistant",
                "content": message.content,
                "function_call": None,
                "tool_calls": None,
            }
        ]
        if message.tool_calls:
            schema[0]["tool_calls"] = [
                get_tool_schema(t) for t in message.tool_calls
            ]
            schema.extend(
                [
                    get_tool_result_schema(t, truncate_images=truncate_images)
                    for t in message.tool_calls
                ]
            )

            image_blocks = []
            image_urls = []

            for tool_call in message.tool_calls:
                if tool_call.status == "completed" and tool_call.result:
                    result = prepare_result(tool_call.result)
                    file_outputs = [
                        o["url"]
                        for r in result
                        for o in r.get("output", [])
                        if isinstance(o, dict) and o.get("url")
                    ]
                    image_outputs = [
                        o
                        for o in file_outputs
                        if o
                        and o.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm")
                        )
                    ]

                    for image_url in image_outputs:
                        try:
                            image_path = download_file(
                                image_url,
                                os.path.join(
                                    "/tmp/eden_file_cache/",
                                    image_url.split("/")[-1],
                                ),
                                overwrite=False,
                            )

                            image_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_to_base64(image_path, max_size=512, quality=95, truncate=truncate_images)}"
                                    },
                                }
                            )

                            if image_url.lower().endswith(("webm", "mp4")):
                                image_urls.append(
                                    f"{image_url} (This url is a video, the corresponding image attachment is its first frame.)"
                                )
                            else:
                                image_urls.append(image_url)

                        except Exception as e:
                            print(f"Error processing image {image_url}: {e}")
                            continue

            # Create single synthetic user message if we have any images
            if image_blocks:
                if len(image_blocks) == 1:
                    content = f"The attached image corresponds to the tool result with url {image_urls[0]}"
                else:
                    content = "The order of the attached images corresponds to the tool results whose URLs are: \n"
                    content += "\n* ".join(image_urls)

                schema.append(
                    {
                        "role": "user",
                        "name": "system_tool_result",
                        "content": [{"type": "text", "text": content}]
                        + image_blocks,
                    }
                )

        return schema


def _get_content_block(message, truncate_images=False):
    """Assemble user message content block"""

    # start with original message content
    content = message.content or ""

    if message.attachments:
        # append attachments info (url and type) to content
        attachment_lines = []
        attachment_files = []
        attachment_errors = []
        for attachment in message.attachments:
            try:
                attachment_file = download_file(
                    attachment,
                    os.path.join(
                        "/tmp/eden_file_cache/", attachment.split("/")[-1]
                    ),
                    overwrite=False,
                )
                mime_type = magic.from_file(attachment_file, mime=True)
                if "video" in mime_type:
                    attachment_lines.append(
                        f"* {attachment} (The asset is a video, the corresponding image attachment is its first frame.)"
                    )
                    attachment_files.append(attachment_file)
                elif "image" in mime_type:
                    attachment_lines.append(f"* {attachment}")
                    attachment_files.append(attachment_file)
                else:
                    attachment_lines.append(
                        f"* {attachment}: (Mime type: {mime_type})"
                    )
            except Exception as e:
                print("error downloading attachment", e)
                attachment_errors.append(f"* {attachment}: {str(e)}")

        attachments = ""
        if attachment_lines:
            attachments += "The attached images correspond to the following urls:\n"
            attachments += "\n".join(attachment_lines)
        if attachment_errors:
            attachments += "The following files failed to attach:\n"
            attachments += "\n".join(attachment_errors)
        attachments = f"<attachments>\n{attachments}\n</attachments>"
        content += f"\n{attachments}"

        # add image blocks
        block = []
        for file_path in attachment_files:
            try:
                block.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"""data:image/jpeg;base64,{
                                image_to_base64(
                                    file_path,
                                    max_size=512,
                                    quality=95,
                                    truncate=truncate_images,
                                )
                            }"""
                        },
                    }
                )
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                # Skip this image and continue with others
                continue

        if content:
            block.extend([{"type": "text", "text": content.strip()}])

        content = block

    return content

