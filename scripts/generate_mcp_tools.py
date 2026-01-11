#!/usr/bin/env python3
"""Generate MCP tool api.yaml files from MCP server tool schemas."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ACRONYMS = {
    "id": "ID",
    "api": "API",
    "url": "URL",
    "uid": "UID",
    "uuid": "UUID",
    "ip": "IP",
    "mcp": "MCP",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def _labelize(name: str) -> str:
    if not name:
        return name
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", name)
    words = [w for w in name.split() if w]
    labeled = []
    for word in words:
        lower = word.lower()
        if lower in ACRONYMS:
            labeled.append(ACRONYMS[lower])
        else:
            labeled.append(word[:1].upper() + word[1:])
    return " ".join(labeled)


def _expand_env(value: str) -> str:
    return os.path.expandvars(value)


def _has_unresolved_env(value: str) -> bool:
    return bool(re.search(r"\$\{?[A-Z0-9_]+\}?", value))


def _build_url(server_url: str, env_params: dict | None) -> str:
    url = _expand_env(server_url)
    params = []
    if env_params:
        for param_name, env_var in env_params.items():
            env_value = os.getenv(env_var)
            if env_value:
                params.append(f"{param_name}={env_value}")
    if params:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{'&'.join(params)}"
    return url


def _build_headers(bearer_env: str | None) -> dict[str, str] | None:
    if not bearer_env:
        return None
    bearer = os.getenv(bearer_env)
    if not bearer:
        return None
    return {"Authorization": f"Bearer {bearer}"}


def _to_type(schema: dict) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = [t for t in schema_type if t != "null"]
        if len(schema_type) == 1:
            schema_type = schema_type[0]
    if schema_type == "number":
        return "float"
    return schema_type


def _convert_schema(schema: dict, required: Iterable[str] | None = None) -> dict:
    required_set = set(required or [])
    properties = schema.get("properties") or {}
    params: dict[str, Any] = {}
    for name, prop_schema in properties.items():
        params[name] = _convert_property(name, prop_schema, name in required_set)
    return params


def _convert_anyof(options: list[dict]) -> list[dict]:
    converted = []
    for option in options:
        if option.get("type") == "null":
            continue
        converted.append(_convert_property("", option, False, include_label=False))
    return converted


def _convert_items(schema: dict) -> dict:
    if schema.get("anyOf") or schema.get("oneOf"):
        return {
            "anyOf": _convert_anyof(schema.get("anyOf") or schema.get("oneOf") or [])
        }
    if schema.get("type") == "object":
        return {
            "type": "object",
            "parameters": _convert_schema(schema, schema.get("required")),
        }
    return _convert_property("", schema, False, include_label=False)


def _convert_property(
    name: str,
    schema: dict,
    required: bool,
    include_label: bool = True,
) -> dict:
    param: dict[str, Any] = {}

    any_of = schema.get("anyOf") or schema.get("oneOf")
    schema_type = _to_type(schema)

    if any_of:
        param["anyOf"] = _convert_anyof(any_of)
    else:
        if schema_type:
            param["type"] = schema_type

    title = schema.get("title")
    description = schema.get("description")

    if include_label and name:
        param["label"] = title or _labelize(name)

    if description:
        param["description"] = description
    elif include_label and title and name:
        param["description"] = title

    if required is not None and name:
        param["required"] = bool(required)

    if "default" in schema:
        param["default"] = schema.get("default")

    if "enum" in schema:
        param["choices"] = schema.get("enum")

    if schema_type in {"integer", "float"}:
        if "minimum" in schema:
            param["minimum"] = schema.get("minimum")
        if "maximum" in schema:
            param["maximum"] = schema.get("maximum")

    if schema_type == "array":
        if "minItems" in schema:
            param["min_length"] = schema.get("minItems")
        if "maxItems" in schema:
            param["max_length"] = schema.get("maxItems")
        items = schema.get("items") or {}
        if items:
            param["items"] = _convert_items(items)

    if schema_type == "object":
        param["parameters"] = _convert_schema(schema, schema.get("required"))

    if "examples" in schema:
        param["examples"] = schema.get("examples")

    return param


def _merge_param_overrides(
    params: dict[str, Any],
    overrides: dict[str, dict],
) -> dict[str, Any]:
    for name, override in overrides.items():
        if name not in params:
            params[name] = {}
        params[name].update(override or {})
    return params


def _rename_params(
    params: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    if not aliases:
        return params
    renamed: dict[str, Any] = {}
    for name, param in params.items():
        renamed_name = aliases.get(name, name)
        renamed[renamed_name] = param
    return renamed


def _tool_output_path(repo_root: Path, output_dir: str, tool_key: str) -> Path:
    return repo_root / output_dir / tool_key / "api.yaml"


def _build_tool_schema(
    defaults: dict,
    server_cfg: dict,
    tool_cfg: dict,
    tool_name: str,
    input_schema: dict,
    remote_tool: Any,
) -> dict:
    key = tool_cfg.get("key") or tool_name

    params = _convert_schema(input_schema, input_schema.get("required"))
    params = _rename_params(params, tool_cfg.get("param_aliases") or {})
    params = _merge_param_overrides(params, tool_cfg.get("param_overrides") or {})

    schema: dict[str, Any] = {}
    schema.update(defaults)
    schema.update(
        {
            "name": tool_cfg.get("name")
            or tool_cfg.get("title")
            or getattr(remote_tool, "title", None),
            "description": tool_cfg.get("description")
            or getattr(remote_tool, "description", None),
        }
    )

    for override_key in ["cost_estimate", "output_type", "active", "visible"]:
        if override_key in tool_cfg:
            schema[override_key] = tool_cfg[override_key]

    if not schema.get("name"):
        schema["name"] = _labelize(key)

    if not schema.get("description"):
        schema["description"] = ""

    if tool_cfg.get("tip"):
        schema["tip"] = tool_cfg.get("tip")

    if tool_cfg.get("thumbnail"):
        schema["thumbnail"] = tool_cfg.get("thumbnail")

    schema.update(
        {
            "handler": defaults.get("handler", "mcp"),
            "mcp_server_url": server_cfg["mcp_server_url"],
            "mcp_tool_name": tool_name,
        }
    )

    for key_name in [
        "mcp_server_name",
        "mcp_env_params",
        "mcp_bearer_env",
        "mcp_use_user_api_key",
        "mcp_timeout",
        "mcp_max_retries",
        "mcp_user_token_url",
        "mcp_user_token_api_key_env",
        "mcp_user_token_ttl_seconds",
    ]:
        if key_name in tool_cfg and tool_cfg[key_name] is not None:
            schema[key_name] = tool_cfg[key_name]
        elif key_name in server_cfg and server_cfg[key_name] is not None:
            schema[key_name] = server_cfg[key_name]

    schema["parameters"] = params

    return schema


def _schema_needs_env(schema: dict) -> list[str]:
    missing = []
    url = _expand_env(schema["mcp_server_url"])
    if _has_unresolved_env(url):
        missing.extend(re.findall(r"\$\{?([A-Z0-9_]+)\}?", url))
    token_url = schema.get("mcp_user_token_url")
    if token_url:
        token_url = _expand_env(token_url)
        if _has_unresolved_env(token_url):
            missing.extend(re.findall(r"\$\{?([A-Z0-9_]+)\}?", token_url))
    env_params = schema.get("mcp_env_params") or {}
    for env_var in env_params.values():
        if not os.getenv(env_var):
            missing.append(env_var)
    bearer_env = schema.get("mcp_bearer_env")
    if bearer_env and not os.getenv(bearer_env):
        missing.append(bearer_env)
    token_api_key_env = schema.get("mcp_user_token_api_key_env")
    if token_api_key_env and not os.getenv(token_api_key_env):
        missing.append(token_api_key_env)
    return missing


def _schema_for_codegen(server_cfg: dict) -> dict:
    schema = dict(server_cfg)
    codegen_url = server_cfg.get("codegen_server_url")
    if codegen_url:
        schema["mcp_server_url"] = codegen_url
    codegen_env_params = server_cfg.get("codegen_env_params")
    if codegen_env_params:
        schema["mcp_env_params"] = codegen_env_params
    codegen_bearer_env = server_cfg.get("codegen_bearer_env")
    if codegen_bearer_env:
        schema["mcp_bearer_env"] = codegen_bearer_env
    return schema


async def _fetch_tools(server_cfg: dict) -> dict[str, Any]:
    url = _build_url(server_cfg["mcp_server_url"], server_cfg.get("mcp_env_params"))
    headers = _build_headers(server_cfg.get("mcp_bearer_env"))
    tools: dict[str, Any] = {}

    async with streamablehttp_client(
        url, headers=headers, timeout=server_cfg.get("mcp_timeout", 30)
    ) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            cursor = None
            while True:
                result = await session.list_tools(cursor=cursor)
                for tool in result.tools:
                    tools[tool.name] = tool
                cursor = result.nextCursor
                if not cursor:
                    break
    return tools


async def _generate(
    config_path: Path,
    check: bool,
    skip_missing_auth: bool,
) -> int:
    config = _load_yaml(config_path)
    repo_root = _repo_root()
    defaults = config.get("defaults") or {}
    servers = config.get("servers") or {}

    changed: list[Path] = []
    for server_name, server_cfg in servers.items():
        server_schema = _schema_for_codegen(server_cfg)
        missing = _schema_needs_env(server_schema)
        if missing and skip_missing_auth:
            print(
                f"Skipping {server_name}: missing env {', '.join(sorted(set(missing)))}"
            )
            continue
        if missing:
            raise RuntimeError(
                f"Missing env for {server_name}: {', '.join(sorted(set(missing)))}"
            )

        tools = await _fetch_tools(server_schema)
        tool_cfgs = server_cfg.get("tools") or {}
        for remote_name, tool_cfg in tool_cfgs.items():
            if remote_name not in tools:
                raise RuntimeError(
                    f"Tool '{remote_name}' not found on MCP server '{server_name}'"
                )

            tool_key = tool_cfg.get("key") or remote_name
            output_path = _tool_output_path(
                repo_root, server_cfg["output_dir"], tool_key
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            tool = tools[remote_name]
            input_schema = tool.inputSchema

            tool_schema = _build_tool_schema(
                defaults,
                server_cfg,
                tool_cfg,
                remote_name,
                input_schema,
                tool,
            )

            existing = None
            if output_path.exists():
                existing = _load_yaml(output_path)

            if existing == tool_schema:
                continue

            changed.append(output_path)
            if not check:
                output_path.write_text(_dump_yaml(tool_schema))

    if changed:
        if check:
            print("MCP tool specs out of date:")
            for path in changed:
                print(f" - {path.relative_to(repo_root)}")
            return 1
        print("Updated MCP tool specs:")
        for path in changed:
            print(f" - {path.relative_to(repo_root)}")
    else:
        print("MCP tool specs up to date.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MCP tool api.yaml files from MCP server schemas."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "eve/tools/mcp/mcp_codegen.yaml",
        help="Path to MCP codegen config.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift without writing files.",
    )
    parser.add_argument(
        "--skip-missing-auth",
        action="store_true",
        help="Skip servers that are missing required auth env vars.",
    )
    args = parser.parse_args()

    return asyncio.run(_generate(args.config, args.check, args.skip_missing_auth))


if __name__ == "__main__":
    raise SystemExit(main())
