"""Gmail deployment client and utilities for handling inbound/outbound email.

Expected setup (see scripts/setup_gmail_deployment.sh for automation):
- A Google Workspace domain-wide delegated service account (JSON key) with
  Gmail scopes (`gmail.modify` and `gmail.send`) stored in deployment secrets.
- Gmail webhooks delivered via Pub/Sub push include either a base64 `raw` MIME
  message or a JSON envelope containing metadata about the inbound email.
- Deployments configure reply timing through `reply_delay_seconds` and
  `reply_variance_seconds`, and may override the reply alias/display name.

Inbound emails are converted into prompt sessions with Gmail-specific update
config so that assistant replies can be returned through the Gmail API after
the configured delay.
"""

import asyncio
import base64
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bson import ObjectId
from fastapi import Request
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from loguru import logger

import eve.mongo
from eve.agent.agent import Agent
from eve.agent.deployments import PlatformClient
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.models import (
    ChatMessageRequestInput,
    Deployment,
    DeploymentConfig,
    DeploymentSecrets,
    DeploymentSecretsGmail,
    DeploymentSettingsGmail,
    Session,
    SessionUpdateConfig,
    UpdateType,
)
from eve.api.api_requests import (
    DeploymentInteractRequest,
    PromptSessionRequest,
    SessionCreationArgs,
)
from eve.api.errors import APIError
from eve.user import User


@dataclass
class GmailInboundEmail:
    message_id: str
    thread_id: Optional[str]
    history_id: Optional[str]
    subject: Optional[str]
    from_address: str
    from_name: Optional[str]
    to_address: Optional[str]
    to_name: Optional[str]
    plain_body: Optional[str]
    html_body: Optional[str]
    snippet: Optional[str]
    references: List[str]
    in_reply_to: Optional[str]
    received_at: Optional[datetime] = None


def _decode_base64url(data: str | bytes) -> bytes:
    if not data:
        return b""
    if isinstance(data, str):
        data = data.encode("utf-8")
    padding = b"=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _strip_html(content: str) -> str:
    import re

    # Basic HTML tag removal for fallback text content
    text = re.sub(r"<head.*?>.*?</head>", "", content, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _decode_mime_header(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        decoded = str(make_header(decode_header(value)))
        return decoded
    except Exception:
        return value


def _collect_message_bodies(mime_message) -> Tuple[Optional[str], Optional[str]]:
    """Extract text/plain and text/html payloads."""
    plain_text = None
    html_text = None

    if mime_message.is_multipart():
        for part in mime_message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" and plain_text is None:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                try:
                    plain_text = payload.decode(charset, errors="replace")
                except Exception:
                    plain_text = payload.decode("utf-8", errors="replace")
            elif content_type == "text/html" and html_text is None:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                try:
                    html_text = payload.decode(charset, errors="replace")
                except Exception:
                    html_text = payload.decode("utf-8", errors="replace")
    else:
        payload = mime_message.get_payload(decode=True) or b""
        charset = mime_message.get_content_charset() or "utf-8"
        try:
            plain_text = payload.decode(charset, errors="replace")
        except Exception:
            plain_text = payload.decode("utf-8", errors="replace")

    return plain_text, html_text


def unwrap_pubsub_message(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return inner payload and attributes from Pub/Sub style body."""
    message = body.get("message") or {}
    attributes = message.get("attributes") or {}

    data_field = message.get("data")
    if data_field:
        try:
            decoded = _decode_base64url(data_field)
            inner = json.loads(decoded)
            return inner, attributes
        except json.JSONDecodeError:
            # If data is raw email rather than JSON, surface as payload with raw field
            return {"raw": data_field}, attributes
        except Exception as exc:
            logger.warning(f"Failed to decode Pub/Sub data: {exc}")

    return body, attributes


def parse_inbound_email(payload: Dict[str, Any]) -> GmailInboundEmail:
    """Normalise inbound webhook payload into GmailInboundEmail."""
    if "gmail_message" in payload and isinstance(payload["gmail_message"], dict):
        payload = payload["gmail_message"]

    raw_data = (
        payload.get("raw")
        or payload.get("raw_email")
        or payload.get("mime_raw")
        or payload.get("rawMessage")
    )

    message_id = (
        payload.get("message_id")
        or payload.get("messageId")
        or payload.get("id")
        or payload.get("gmail_message_id")
    )
    thread_id = payload.get("thread_id") or payload.get("threadId")
    history_id = payload.get("history_id") or payload.get("historyId")
    subject = payload.get("subject")
    snippet = payload.get("snippet")
    from_value = payload.get("from_address") or payload.get("from")
    to_value = payload.get("to_address") or payload.get("to")
    references = payload.get("references") or payload.get("References") or []
    in_reply_to = payload.get("in_reply_to") or payload.get("In-Reply-To")
    received_ts = payload.get("internal_date") or payload.get("received_ts")

    plain_body = payload.get("body_text") or payload.get("text_body")
    html_body = payload.get("body_html") or payload.get("html_body")

    from_name = None
    to_name = None

    if raw_data:
        try:
            mime_bytes = _decode_base64url(raw_data)
            mime_message = message_from_bytes(mime_bytes)

            subject = subject or _decode_mime_header(mime_message.get("Subject"))

            from_parsed = parseaddr(mime_message.get("From", ""))
            if from_parsed[1]:
                from_name = _decode_mime_header(from_parsed[0]) or from_name
                from_value = from_parsed[1]

            to_parsed = parseaddr(mime_message.get("To", ""))
            if to_parsed[1]:
                to_name = _decode_mime_header(to_parsed[0]) or to_name
                to_value = to_parsed[1]

            if not references:
                refs = mime_message.get_all("References") or []
                references = []
                for ref_header in refs:
                    references.extend(ref_header.split())

            in_reply_to = in_reply_to or mime_message.get("In-Reply-To")
            message_id = message_id or mime_message.get("Message-ID")

            extracted_plain, extracted_html = _collect_message_bodies(mime_message)
            plain_body = plain_body or extracted_plain
            html_body = html_body or extracted_html
        except Exception as exc:
            logger.warning(f"Failed to parse raw Gmail payload: {exc}")

    # Fallback parse for direct address strings
    if from_value and not from_name:
        parsed = parseaddr(from_value)
        if parsed[0]:
            from_name = parsed[0]
        if parsed[1]:
            from_value = parsed[1]

    if to_value and not to_name:
        parsed = parseaddr(to_value)
        if parsed[0]:
            to_name = parsed[0]
        if parsed[1]:
            to_value = parsed[1]

    if not message_id:
        raise ValueError("Inbound Gmail payload missing message_id")

    if isinstance(references, str):
        references = references.split()
    references = [ref.strip() for ref in references if ref]

    received_at = None
    if received_ts:
        try:
            # Accept epoch millis or RFC3339 strings
            if isinstance(received_ts, (int, float)):
                received_at = datetime.fromtimestamp(received_ts / 1000, tz=timezone.utc)
            else:
                received_at = datetime.fromisoformat(received_ts)
        except Exception:
            received_at = None

    return GmailInboundEmail(
        message_id=message_id,
        thread_id=thread_id,
        history_id=history_id,
        subject=subject,
        from_address=from_value,
        from_name=from_name,
        to_address=to_value,
        to_name=to_name,
        plain_body=plain_body,
        html_body=html_body,
        snippet=snippet,
        references=references or [],
        in_reply_to=in_reply_to,
        received_at=received_at,
    )


class GmailAPIClient:
    def __init__(
        self, secrets: DeploymentSecretsGmail, settings: DeploymentSettingsGmail
    ):
        scopes = secrets.token_scopes or [
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.send",
        ]
        self._credentials = service_account.Credentials.from_service_account_info(
            secrets.service_account_info,
            scopes=scopes,
            subject=secrets.delegated_user,
        )
        self._user = secrets.delegated_user
        self._token_lock = asyncio.Lock()
        self._settings = settings

    async def _refresh_if_needed(self):
        if self._credentials.valid and not self._credentials.expired:
            return
        async with self._token_lock:
            if self._credentials.valid and not self._credentials.expired:
                return
            await asyncio.to_thread(self._credentials.refresh, GoogleRequest())

    async def get_access_token(self) -> str:
        await self._refresh_if_needed()
        if not self._credentials.token:
            await asyncio.to_thread(self._credentials.refresh, GoogleRequest())
        return self._credentials.token

    async def get_profile(self) -> Dict[str, Any]:
        token = await self.get_access_token()
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}/profile"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            return response.json()

    async def send_email(
        self, message: EmailMessage, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        raw_bytes = message.as_bytes()
        encoded_message = base64.urlsafe_b64encode(raw_bytes).decode("utf-8")
        payload: Dict[str, Any] = {"raw": encoded_message}
        if thread_id:
            payload["threadId"] = thread_id

        token = await self.get_access_token()
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}/messages/send"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code >= 400:
                raise APIError(
                    f"Gmail send failed: {response.status_code} ({response.text})",
                    status_code=response.status_code,
                )
            return response.json()

    async def watch_mailbox(
        self,
        topic_name: str,
        label_ids: Optional[List[str]] = None,
        include_spam_and_trash: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"topicName": topic_name}
        if label_ids:
            body["labelIds"] = label_ids
        if include_spam_and_trash:
            body["labelFilterBehavior"] = "include"

        token = await self.get_access_token()
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}/watch"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code >= 400:
                raise APIError(
                    f"Gmail watch failed: {response.status_code} ({response.text})",
                    status_code=response.status_code,
                )
            return response.json()


class GmailClient(PlatformClient):
    TOOLS = []

    def __init__(self, agent: Optional[Agent] = None, deployment: Optional[Deployment] = None):
        super().__init__(agent=agent, deployment=deployment)
        self._gmail_api_client: Optional[GmailAPIClient] = None

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> Tuple[DeploymentSecrets, DeploymentConfig]:
        secrets = secrets or DeploymentSecrets()
        config = config or DeploymentConfig()

        if not secrets.gmail:
            raise APIError("Gmail secrets are required for Gmail deployment", 400)

        if not secrets.gmail.service_account_info:
            raise APIError("Gmail service_account_info is required", 400)
        if not secrets.gmail.delegated_user:
            raise APIError("Gmail delegated_user is required", 400)

        if config.gmail is None:
            config.gmail = DeploymentSettingsGmail()

        if not config.gmail.reply_from_address:
            config.gmail.reply_from_address = (
                secrets.gmail.reply_alias or secrets.gmail.delegated_user
            )

        # Validate credentials by fetching Gmail profile
        gmail_api = GmailAPIClient(secrets.gmail, config.gmail)
        try:
            profile = await gmail_api.get_profile()
            logger.info(
                f"[GMAIL-PREDEPLOY] Verified Gmail access for {profile.get('emailAddress')}"
            )
        except Exception as exc:
            logger.error(f"[GMAIL-PREDEPLOY] Unable to validate Gmail credentials: {exc}")
            raise APIError(
                f"Failed to validate Gmail credentials: {exc}",
                status_code=400,
            )

        return secrets, config

    async def postdeploy(self) -> None:
        logger.info("[GMAIL-POSTDEPLOY] Gmail deployment ready")

    async def stop(self) -> None:
        logger.info("[GMAIL-STOP] No teardown required for Gmail deployments yet")

    async def interact(self, request: Request | DeploymentInteractRequest) -> None:
        """Handle direct DeploymentInteractRequest or FastAPI request wrapper."""
        if isinstance(request, DeploymentInteractRequest):
            interact_request = request
        elif isinstance(request, Request):
            data = await request.json()
            interact_request = DeploymentInteractRequest(**data)
        else:
            raise TypeError("Unsupported request type for GmailClient.interact")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{get_api_url()}/sessions/prompt",
                json=interact_request.interaction.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                    "X-Client-Platform": "gmail",
                    "X-Client-Deployment-Id": str(self.deployment.id)
                    if self.deployment
                    else "",
                },
            )
            if response.status_code != 200:
                raise APIError(
                    f"Gmail interaction failed: {response.status_code} ({response.text})",
                    status_code=response.status_code,
                )

    async def handle_emission(self, emission) -> None:
        if not self.deployment:
            raise ValueError("Deployment is required for Gmail handle_emission")
        if emission.type not in (UpdateType.ASSISTANT_MESSAGE,):
            return

        message_text = emission.content or ""
        if not message_text.strip():
            return

        update_config = emission.update_config
        if not update_config:
            logger.warning("[GMAIL-EMISSION] Missing update_config; skipping send")
            return

        recipient = update_config.gmail_from_address
        if not recipient:
            logger.warning(
                "[GMAIL-EMISSION] Missing gmail_from_address in update_config; skipping send"
            )
            return

        settings = self._get_settings()
        delay_seconds = self._compute_reply_delay(settings)

        asyncio.create_task(
            self._send_email_after_delay(
                recipient=recipient,
                subject=update_config.gmail_subject,
                thread_id=update_config.gmail_thread_id,
                in_reply_to=update_config.gmail_message_id or update_config.gmail_thread_id,
                references=self._collect_reference_ids(update_config),
                body=message_text,
                delay_seconds=delay_seconds,
            )
        )

    async def process_inbound_email(self, email: GmailInboundEmail) -> None:
        if not self.deployment:
            raise ValueError("Deployment is required for processing Gmail email")

        # Load agent and ensure we have latest deployment context
        if not self.agent:
            self.agent = Agent.from_mongo(ObjectId(self.deployment.agent))

        session_key_basis = email.thread_id or email.message_id
        session_key = f"gmail-{session_key_basis}"

        # Resolve or create user from email
        user = User.from_email(
            email_address=email.from_address,
            fallback_username=email.from_name,
        )

        session = None
        try:
            session = Session.load(session_key=session_key)
            if session.deleted:
                session.deleted = False
                session.status = "active"
                session.save()
        except eve.mongo.MongoDocumentNotFound:
            session = None

        message_content = email.plain_body or (
            _strip_html(email.html_body) if email.html_body else None
        )
        if not message_content:
            message_content = email.snippet or ""

        api_url = get_api_url()
        prompt_request = PromptSessionRequest(
            user_id=str(user.id),
            actor_agent_ids=[str(self.deployment.agent)],
            message=ChatMessageRequestInput(
                content=message_content,
                sender_name=email.from_name or user.username,
            ),
            update_config=SessionUpdateConfig(
                update_endpoint=f"{api_url}/v2/deployments/emission",
                deployment_id=str(self.deployment.id),
                gmail_thread_id=email.thread_id or email.message_id,
                gmail_message_id=email.message_id,
                gmail_history_id=email.history_id,
                gmail_from_address=email.from_address,
                gmail_to_address=email.to_address,
                gmail_subject=email.subject,
            ),
        )

        if session:
            prompt_request.session_id = str(session.id)
        else:
            prompt_request.creation_args = SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(self.deployment.agent)],
                title=email.subject or f"Gmail thread {session_key_basis}",
                session_key=session_key,
                platform="gmail",
                extras={
                    "gmail_thread_id": email.thread_id,
                    "gmail_initial_message_id": email.message_id,
                },
            )

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{api_url}/sessions/prompt",
                json=prompt_request.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                    "X-Client-Platform": "gmail",
                    "X-Client-Deployment-Id": str(self.deployment.id),
                },
            )
            if response.status_code != 200:
                logger.error(
                    f"[GMAIL-INBOUND] Session prompt failed: {response.status_code} - {response.text}"
                )
                raise APIError(
                    f"Failed to process Gmail email: {response.status_code}",
                    status_code=response.status_code,
                )

        # Persist last_history_id if provided
        if email.history_id:
            self._update_history_id(email.history_id)

    async def ensure_watch(self) -> Optional[Dict[str, Any]]:
        """Ensure Gmail watch is active for this deployment."""
        if not self.deployment:
            raise ValueError("Deployment is required for ensure_watch")

        secrets = self._get_secrets()
        topic = secrets.pubsub_topic
        if not topic:
            logger.warning("[GMAIL-WATCH] No pubsub_topic configured; skipping watch")
            return None

        include_spam = os.getenv("GMAIL_WATCH_INCLUDE_SPAM", "false").lower() == "true"

        gmail_api = self._get_gmail_api_client()
        try:
            response = await gmail_api.watch_mailbox(
                topic_name=topic,
                label_ids=secrets.watch_label_ids,
                include_spam_and_trash=include_spam,
            )
        except APIError as exc:
            logger.error(f"[GMAIL-WATCH] Failed to refresh watch: {exc}")
            raise

        history_id = response.get("historyId")
        if history_id:
            self._update_history_id(str(history_id))

        expiration = response.get("expiration")
        if expiration:
            self._update_watch_expiration(expiration)

        logger.info(
            f"[GMAIL-WATCH] Watch refreshed for deployment {self.deployment.id}. "
            f"History: {history_id}, Expiration: {expiration}"
        )
        return response

    async def _send_email_after_delay(
        self,
        recipient: str,
        subject: Optional[str],
        thread_id: Optional[str],
        in_reply_to: Optional[str],
        references: List[str],
        body: str,
        delay_seconds: float,
    ):
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            settings = self._get_settings()
            secrets = self._get_secrets()
            gmail_api = self._get_gmail_api_client()

            reply_from = settings.reply_from_address or secrets.delegated_user
            display_name = settings.reply_display_name
            from_header = formataddr((display_name, reply_from)) if display_name else reply_from

            if subject:
                normalized_subject = subject
                if not subject.lower().startswith("re:"):
                    normalized_subject = f"Re: {subject}"
            else:
                normalized_subject = "Re: your message"

            email_message = EmailMessage()
            email_message["To"] = recipient
            email_message["From"] = from_header
            email_message["Subject"] = normalized_subject
            if in_reply_to:
                email_message["In-Reply-To"] = in_reply_to
            if references:
                email_message["References"] = " ".join(references)

            email_message.set_content(body)

            logger.info(
                f"[GMAIL-SEND] Sending reply to {recipient} (thread: {thread_id}, delay: {delay_seconds:.2f}s)"
            )
            await gmail_api.send_email(email_message, thread_id=thread_id)
        except Exception as exc:
            logger.error(f"[GMAIL-SEND] Failed to send Gmail response: {exc}", exc_info=True)

    def _get_secrets(self) -> DeploymentSecretsGmail:
        deployment_secrets = self.deployment.secrets
        if isinstance(deployment_secrets, dict):
            deployment_secrets = DeploymentSecrets(**deployment_secrets)
        if not deployment_secrets.gmail:
            raise ValueError("Gmail secrets missing on deployment")
        return deployment_secrets.gmail

    def _get_settings(self) -> DeploymentSettingsGmail:
        deployment_config = self.deployment.config
        if isinstance(deployment_config, dict):
            deployment_config = DeploymentConfig(**deployment_config)
        if not deployment_config or not deployment_config.gmail:
            return DeploymentSettingsGmail()
        return deployment_config.gmail

    def _get_gmail_api_client(self) -> GmailAPIClient:
        if not self._gmail_api_client:
            self._gmail_api_client = GmailAPIClient(
                self._get_secrets(),
                self._get_settings(),
            )
        return self._gmail_api_client

    def _compute_reply_delay(self, settings: DeploymentSettingsGmail) -> float:
        base = max(settings.reply_delay_seconds or 0, 0)
        variance = max(settings.reply_variance_seconds or 0, 0)
        return base + (random.uniform(0, variance) if variance > 0 else 0)

    def _collect_reference_ids(self, update_config: SessionUpdateConfig) -> List[str]:
        references = []
        if update_config.gmail_message_id:
            references.append(update_config.gmail_message_id)
        return references

    def _update_history_id(self, history_id: str):
        try:
            Deployment.get_collection().update_one(
                {"_id": self.deployment.id},
                {"$set": {"config.gmail.last_history_id": history_id}},
            )
            self.deployment.reload()
        except Exception as exc:
            logger.warning(f"[GMAIL] Failed to persist last history id: {exc}")

    def _update_watch_expiration(self, expiration_ms: int | str):
        try:
            expiration_int = int(expiration_ms)
            expiration_dt = datetime.fromtimestamp(expiration_int / 1000, tz=timezone.utc)
        except Exception as exc:
            logger.warning(f"[GMAIL] Invalid watch expiration value: {expiration_ms} ({exc})")
            return

        try:
            Deployment.get_collection().update_one(
                {"_id": self.deployment.id},
                {"$set": {"config.gmail.watch_expiration": expiration_dt.isoformat()}},
            )
            self.deployment.reload()
        except Exception as exc:
            logger.warning(f"[GMAIL] Failed to persist watch expiration: {exc}")
