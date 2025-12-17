from __future__ import annotations

import json
from datetime import date, datetime

from bson import ObjectId

from .. import s3
from .validation_utils import is_downloadable_file


def prepare_result(result, summarize=False):
    if isinstance(result, dict):
        if "error" in result and result["error"] is not None:
            return result
        if "mediaAttributes" in result:
            result["mediaAttributes"].pop("blurhash", None)
        if "filename" in result:
            filename = result.pop("filename")
            url = s3.get_full_url(filename)
            if summarize:
                return url
            else:
                result["url"] = url
        return {k: prepare_result(v, summarize) for k, v in result.items()}
    elif isinstance(result, list):
        return [prepare_result(item, summarize) for item in result]
    else:
        return result


def upload_result(result, save_thumbnails=False, save_blurhash=False, tool_key=None):
    from ..tool_constants import SKIP_UPLOAD_PROCESSING_TOOLS
    from .media_utils import upload_media

    if tool_key and tool_key in SKIP_UPLOAD_PROCESSING_TOOLS:
        return result

    if isinstance(result, dict):
        exlude_result_processing_keys = ["subtool_calls"]
        return {
            k: upload_result(
                v, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
            )
            if k not in exlude_result_processing_keys
            else v
            for k, v in result.items()
        }
    elif isinstance(result, list):
        return [
            upload_result(
                item, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
            )
            for item in result
        ]
    elif is_downloadable_file(result):
        return upload_media(
            result, save_thumbnails=save_thumbnails, save_blurhash=save_blurhash
        )
    else:
        return result


def deep_filter(current, changes):
    if not isinstance(current, dict) or not isinstance(changes, dict):
        return changes if changes != current else None
    result = {}
    for key, value in changes.items():
        if key in current:
            if isinstance(current[key], dict) and isinstance(value, dict):
                filtered = deep_filter(current[key], value)
                if filtered:
                    result[key] = filtered
            elif current[key] != value:
                result[key] = value
        else:
            result[key] = value
    return result if result else None


def deep_update(data, changes):
    if not isinstance(data, dict) or not isinstance(changes, dict):
        return changes
    for key, value in changes.items():
        if key in data:
            if isinstance(data[key], dict) and isinstance(value, dict):
                deep_update(data[key], value)
            elif data[key] != value:
                data[key] = value
        else:
            data[key] = value
    return data


def serialize_json(obj, *, indent=None, exclude=None):
    """Return *obj* as a JSON string."""

    def scrub(value):
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            pruned = {k: v for k, v in value.items() if not exclude or k not in exclude}
            return {k: scrub(v) for k, v in pruned.items()}
        if isinstance(value, (list, tuple, set)):
            return [scrub(item) for item in value]
        return value

    if obj is None:
        return ""

    cleaned = scrub(obj)
    return cleaned


def dumps_json(obj, *, indent=None, exclude=None):
    cleaned = serialize_json(obj, indent=indent, exclude=exclude)
    return json.dumps(cleaned, indent=indent)


def overwrite_dict(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            overwrite_dict(base[key], value)
        else:
            base[key] = value
