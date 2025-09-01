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
        "type": "tool_use",
        "id": tool_call.id,
        "name": tool_call.tool,
        "input": tool_call.args,
    }


def get_tool_result_schema(tool_call, truncate_images=False):    
    content = {"status": tool_call.status}

    if tool_call.status == "completed":
        content["result"] = prepare_result(tool_call.result)
        file_outputs = [
            o["url"]
            for r in content["result"]
            for o in r.get("output", [])
            if isinstance(o, dict) and o.get("url")
        ]
        file_outputs = [
            o
            for o in file_outputs
            if o
            and o.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm")
            )
        ]
        try:
            files = [
                download_file(
                    url,
                    os.path.join("/tmp/eden_file_cache/", url.split("/")[-1]),
                    overwrite=False,
                )
                for url in file_outputs
            ]
            image_block = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_to_base64(
                            file_path,
                            max_size=512,
                            quality=95,
                            truncate=truncate_images,
                        ),
                    },
                }
                for file_path in files
            ]

            if image_block:
                image_block_content = dumps_json(content["result"])
                text_block = [{"type": "text", "text": image_block_content}]
                content = text_block + image_block
            else:
                content = dumps_json(content)

        except Exception as e:
            print("Warning: Can not inject image results:", e)
            content = dumps_json(content)

    # For Anthropic: if content is a list (text + images), use it directly
    # If content is a dict/object, JSON encode it
    if isinstance(content, dict):
        content = dumps_json(content)

    result = {"type": "tool_result", "tool_use_id": tool_call.id, "content": content}

    if tool_call.status == "failed":
        result["is_error"] = True

    return result


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
    if message.role == "user":
        content = _get_content_block(message, truncate_images=truncate_images)
        return [{"role": "user", "content": content}] if content else []

    # Assistant Message
    else:
        if not message.content and not message.tool_calls:
            return []
        schema = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": message.content}]
                if message.content
                else [],
            }
        ]
        if message.tool_calls:
            schema[0]["content"].extend(
                [get_tool_schema(t) for t in message.tool_calls]
            )
            schema.append(
                {
                    "role": "user",
                    "content": [
                        get_tool_result_schema(t, truncate_images=truncate_images)
                        for t in message.tool_calls
                    ],
                }
            )
        return schema


def _get_content_block(message, truncate_images=False):
    """Assemble user message content block"""

    # start with original message content
    content = message.content or ""

    # let claude see names
    if message.name:
        content = f"<User>{message.name}</User>\n\n{content}"

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
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_base64(
                                file_path,
                                max_size=512,
                                quality=95,
                                truncate=truncate_images,
                            ),
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
