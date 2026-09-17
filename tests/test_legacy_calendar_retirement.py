"""Offline actual-source tests; never import eve bootstrap or read ~/.eve.

Load complete Calendar modules with only external imports stubbed. For the huge
API module, compile the exact selected handler ASTs with inert boundary objects.
No Mongo, Google, Modal, credentials or live agent state is accessed.
"""
import ast
import asyncio
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "Google Calendar is retired in old Eden. Use new Eden at https://dev.eden.art."

class APIError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

class PlatformClient:
    def remove_tools(self):
        pass

def module(**values):
    result = types.ModuleType("fixture")
    result.__dict__.update(values)
    return result

class Retirement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.io = MagicMock(side_effect=AssertionError("external I/O forbidden"))
        self.modules = {
            "eve.agent.deployments": module(PlatformClient=PlatformClient),
            "eve.agent.session.models": module(DeploymentConfig=object, DeploymentSecrets=object,
                DeploymentSecretsGoogleCalendar=object, Deployment=types.SimpleNamespace(load=self.io)),
            "eve.api.errors": module(APIError=APIError),
            "eve.agent.agent": module(Agent=types.SimpleNamespace(from_mongo=self.io)),
            "eve.tool": module(ToolContext=object),
            "google.auth.transport.requests": module(Request=self.io),
            "google.oauth2.credentials": module(Credentials=self.io),
            "google_auth_oauthlib.flow": module(Flow=types.SimpleNamespace(from_client_config=self.io)),
            "googleapiclient.discovery": module(build=self.io),
            "googleapiclient.errors": module(HttpError=type("HttpError", (Exception,), {})),
            "loguru": module(logger=MagicMock()),
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def load(self, relative):
        return runpy.run_path(str(ROOT / relative))

    async def denied(self, call):
        with self.assertRaises(APIError) as raised:
            result = call()
            if asyncio.iscoroutine(result):
                await result
        self.assertEqual(raised.exception.status_code, 410)
        self.assertEqual(str(raised.exception), MESSAGE)
        self.io.assert_not_called()

    async def test_oauth_credentials_and_service_are_retired(self):
        source = self.load("eve/agent/deployments/google_calendar.py")
        for name, args in [("get_google_client_config", ()), ("create_oauth_flow", ("https://fixture.invalid",)),
                           ("credentials_from_secrets", (object(),)), ("get_calendar_service", (object(),)),
                           ("list_user_calendars", (object(),))]:
            with self.subTest(name=name):
                await self.denied(lambda: source[name](*args))

    async def test_platform_hooks_cannot_activate_calendar(self):
        source = self.load("eve/agent/deployments/google_calendar.py")
        client = source["GoogleCalendarClient"]()
        for name, args in [("predeploy", (object(), object())), ("postdeploy", ()),
                           ("update", ()), ("interact", (object(),)),
                           ("handle_emission", (object(),)), ("add_tools", ())]:
            with self.subTest(name=name):
                await self.denied(lambda: getattr(client, name)(*args))

    async def test_all_registered_tools_refuse_before_reading_saved_credentials(self):
        platform = module(**self.load("eve/agent/deployments/google_calendar.py"))
        with patch.dict(sys.modules, {"eve.agent.deployments.google_calendar": platform}):
            utilities = module(**self.load("eve/tools/google_calendar/utils.py"))
            with patch.dict(sys.modules, {"eve.tools.google_calendar.utils": utilities}):
                for name, action in [("query", "list"), ("query", "get"), ("query", "find_free_slots"),
                                     ("edit", "create"), ("edit", "update"), ("delete", "delete")]:
                    with self.subTest(name=name, action=action):
                        handler = self.load(f"eve/tools/google_calendar/google_calendar_{name}/handler.py")["handler"]
                        await self.denied(lambda: handler(types.SimpleNamespace(agent="synthetic", args={"action": action})))

    async def test_generic_create_update_delete_refuse_without_mutating_deployment(self):
        tree = ast.parse((ROOT / "eve/api/handlers.py").read_text())
        for name in ["handle_v2_deployment_create", "handle_v2_deployment_update", "handle_v2_deployment_delete"]:
            with self.subTest(name=name):
                function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
                function.decorator_list = []
                code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
                deployment = types.SimpleNamespace(platform="google_calendar")
                namespace = {"APIError": APIError, "ClientType": types.SimpleNamespace(GOOGLE_CALENDAR="google_calendar"),
                    "Agent": types.SimpleNamespace(from_mongo=self.io), "ObjectId": lambda value: value,
                    "Deployment": types.SimpleNamespace(from_mongo=lambda value: deployment)}
                exec(compile(ast.fix_missing_locations(code), "eve/api/handlers.py", "exec"), namespace)
                request = types.SimpleNamespace(platform="google_calendar", agent="synthetic", deployment_id="synthetic")
                await self.denied(lambda: namespace[name](request))

if __name__ == "__main__":
    unittest.main()
