"""Gmail deployment client and utilities for handling inbound/outbound email.

Expected setup (see scripts/setup_gmail_deployment.sh for automation):
- A Google Workspace domain-wide delegated service account (JSON key) with
  Gmail scopes (`gmail.modify` and `gmail.send`) stored in deployment secrets.
- Gmail webhooks delivered via Pub/Sub push include either a base64 `raw` MIME
  message or a JSON envelope containing metadata about the inbound email.
- Deployments configure reply timing through `reply_delay_minimum` and
  `reply_delay_variance`, and may override the reply alias/display name.

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
from datetime import datetime, timedelta, timezone
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
    PromptSessionContext,
    Session,
    SessionUpdateConfig,
    UpdateType,
)
from eve.api.api_requests import (
    DeploymentInteractRequest,
)
from eve.api.errors import APIError
from eve.user import User

MAX_LENGTH_FOR_FULL_VARIANCE = 1800
from eve.agent.session.context import add_chat_message
from eve.trigger import Trigger, calculate_next_scheduled_run


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


def unwrap_pubsub_message(
    body: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
                received_at = datetime.fromtimestamp(
                    received_ts / 1000, tz=timezone.utc
                )
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

    async def list_history(
        self, start_history_id: str, page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {
            "startHistoryId": start_history_id,
            "historyTypes": "messageAdded",
        }
        if page_token:
            params["pageToken"] = page_token

        token = await self.get_access_token()
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}/history"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return response.json()

    async def get_message_raw(self, message_id: str) -> Dict[str, Any]:
        token = await self.get_access_token()
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}/messages/{message_id}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                params={"format": "raw"},
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return response.json()


class GmailClient(PlatformClient):
    TOOLS = ["gmail_send"]

    def __init__(
        self, agent: Optional[Agent] = None, deployment: Optional[Deployment] = None
    ):
        super().__init__(agent=agent, deployment=deployment)
        self._gmail_api_client: Optional[GmailAPIClient] = None
        self._own_addresses: Optional[set[str]] = None

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
            logger.error(
                f"[GMAIL-PREDEPLOY] Unable to validate Gmail credentials: {exc}"
            )
            raise APIError(
                f"Failed to validate Gmail credentials: {exc}",
                status_code=400,
            )

        try:
            self.add_tools()
        except Exception as exc:
            raise APIError(f"Failed to add Gmail tools: {exc}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        logger.info("[GMAIL-POSTDEPLOY] Gmail deployment ready")

    async def stop(self) -> None:
        try:
            self.remove_tools()
        except Exception as exc:
            logger.error(f"[GMAIL-STOP] Failed to remove Gmail tools: {exc}")

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

        asyncio.create_task(
            self._send_email_after_delay(
                recipient=recipient,
                subject=update_config.gmail_subject,
                thread_id=update_config.gmail_thread_id,
                in_reply_to=update_config.gmail_message_id
                or update_config.gmail_thread_id,
                references=self._collect_reference_ids(update_config),
                body=message_text,
                delay_seconds=0,
            )
        )

    async def process_inbound_email(self, email: GmailInboundEmail) -> None:
        if not self.deployment:
            raise ValueError("Deployment is required for processing Gmail email")

        own_addresses = self._get_own_addresses()
        if email.from_address and email.from_address.lower() in own_addresses:
            logger.info(
                f"[GMAIL-INBOUND] Skipping self-sent message {email.message_id} from {email.from_address}"
            )
            return

        if not self.agent:
            self.agent = Agent.from_mongo(ObjectId(self.deployment.agent))

        settings = self._get_settings()
        secrets = self._get_secrets()

        session_key_basis = email.thread_id or email.message_id
        session_key = f"gmail-{session_key_basis}"

        user = User.from_gmail(
            email_address=email.from_address,
            fallback_username=email.from_name,
        )

        session: Optional[Session] = None
        try:
            session = Session.load(session_key=session_key)
            if hasattr(session, "deleted") and session.deleted:
                session.deleted = False
                session.status = "active"
                session.save()
        except eve.mongo.MongoDocumentNotFound:
            session = None

        if not session:
            session = self._create_gmail_session(
                user=user,
                session_key=session_key,
                session_key_basis=session_key_basis,
                email=email,
            )

        message_content = email.plain_body or (
            _strip_html(email.html_body) if email.html_body else None
        )
        if not message_content:
            message_content = email.snippet or ""

        inbound_context = PromptSessionContext(
            session=session,
            initiating_user_id=str(user.id),
            message=ChatMessageRequestInput(
                content=message_content,
                sender_name=email.from_name or user.username,
            ),
        )
        await add_chat_message(session, inbound_context)

        api_url = get_api_url()
        update_config = SessionUpdateConfig(
            update_endpoint=f"{api_url}/v2/deployments/emission",
            deployment_id=str(self.deployment.id),
            gmail_thread_id=email.thread_id or email.message_id,
            gmail_message_id=email.message_id,
            gmail_history_id=email.history_id,
            gmail_from_address=email.from_address,
            gmail_to_address=email.to_address
            or settings.reply_from_address
            or secrets.reply_alias
            or secrets.delegated_user,
            gmail_subject=email.subject,
        )

        delay_seconds = self._compute_reply_delay(settings, message_content)

        await self._schedule_reply_trigger(
            session=session,
            user=user,
            email=email,
            update_config=update_config,
            message_content=message_content,
            delay_seconds=delay_seconds,
            settings=settings,
        )

        if email.history_id:
            self._update_history_id(email.history_id)

    async def process_history_update(self, history_id: str) -> Dict[str, Any]:
        """Handle Gmail push notifications that provide only a historyId."""
        if not self.deployment:
            raise ValueError("Deployment is required for process_history_update")

        settings = self._get_settings()
        last_history_id = settings.last_history_id

        if not last_history_id:
            logger.info(
                "[GMAIL-HISTORY] No previous history_id stored; initializing baseline."
            )
            self._update_history_id(history_id)
            return {"processed": 0, "reason": "baseline_initialized"}

        gmail_api = self._get_gmail_api_client()

        processed = 0
        next_page: Optional[str] = None
        seen_messages: set[str] = set()

        while True:
            history_response = await gmail_api.list_history(
                start_history_id=last_history_id, page_token=next_page
            )

            for history_entry in history_response.get("history", []):
                messages_added = history_entry.get("messagesAdded") or []
                for message_info in messages_added:
                    message_meta = message_info.get("message") or {}
                    message_id = message_meta.get("id")
                    label_ids = message_meta.get("labelIds") or []

                    if not message_id or message_id in seen_messages:
                        continue

                    if any(label in label_ids for label in ["SENT", "DRAFT", "OUTBOX"]):
                        logger.info(
                            f"[GMAIL-HISTORY] Skipping message {message_id} due to labels {label_ids}"
                        )
                        continue

                    seen_messages.add(message_id)
                    try:
                        raw_message = await gmail_api.get_message_raw(message_id)
                    except Exception as exc:
                        logger.warning(
                            f"[GMAIL-HISTORY] Failed to fetch raw message {message_id}: {exc}"
                        )
                        continue

                    payload = {
                        "raw": raw_message.get("raw"),
                        "id": raw_message.get("id"),
                        "threadId": raw_message.get("threadId"),
                    }

                    try:
                        email = parse_inbound_email(payload)
                        email.history_id = history_id
                        await self.process_inbound_email(email)
                        processed += 1
                    except Exception as exc:
                        logger.error(
                            f"[GMAIL-HISTORY] Failed to process message {message_id}: {exc}",
                            exc_info=True,
                        )

            next_page = history_response.get("nextPageToken")
            if not next_page:
                break

        self._update_history_id(history_id)
        return {"processed": processed}

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
            from_header = (
                formataddr((display_name, reply_from)) if display_name else reply_from
            )

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
            logger.error(
                f"[GMAIL-SEND] Failed to send Gmail response: {exc}", exc_info=True
            )

    @classmethod
    async def send_tool_email(
        cls,
        *,
        agent_id: Optional[str],
        args: Optional[dict],
        user: Optional[str] = None,
    ) -> dict:
        if not agent_id:
            raise APIError("Agent identifier is required", status_code=400)

        agent = Agent.from_mongo(agent_id)
        if not agent:
            raise APIError("Agent not found", status_code=404)

        deployment = Deployment.load(agent=agent.id, platform="gmail")
        if not deployment or not deployment.valid:
            raise APIError(
                "Agent has no valid Gmail deployment configured",
                status_code=400,
            )

        client = cls(agent=agent, deployment=deployment)

        email_args = args or {}
        to_address = email_args.get("to")
        subject = email_args.get("subject")
        body = email_args.get("text") or email_args.get("body")
        thread_id = email_args.get("thread_id")
        in_reply_to = email_args.get("in_reply_to")
        references = email_args.get("references") or []
        if isinstance(references, str):
            references = [references]
        references = [str(ref) for ref in references if ref]
        delay_seconds = max(0.0, float(email_args.get("delay_seconds") or 0))

        if not to_address:
            raise APIError("Recipient email address is required", status_code=400)
        if not subject:
            raise APIError("Email subject is required", status_code=400)
        if not body:
            raise APIError("Email body is required", status_code=400)

        await client._send_email_after_delay(
            recipient=to_address,
            subject=subject,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            references=references,
            body=body,
            delay_seconds=delay_seconds,
        )

        return {
            "output": [
                {
                    "url": f"mailto:{to_address}",
                    "title": subject,
                    "status": "sent",
                }
            ]
        }

    async def _schedule_reply_trigger(
        self,
        *,
        session: Session,
        user: User,
        email: GmailInboundEmail,
        update_config: SessionUpdateConfig | Dict[str, Any],
        message_content: str,
        delay_seconds: float,
        settings: DeploymentSettingsGmail,
    ) -> None:
        """Create a one-time trigger that will prompt the agent to reply via Gmail."""

        if isinstance(update_config, SessionUpdateConfig):
            to_address_display = update_config.gmail_to_address
        else:
            to_address_display = update_config.get("gmail_to_address")

        trigger_collection = Trigger.get_collection()
        existing_trigger = trigger_collection.find_one(
            {
                "update_config.gmail_message_id": email.message_id,
                "deleted": {"$ne": True},
                "status": {"$in": ["active", "running"]},
            }
        )
        if existing_trigger:
            logger.info(
                f"[GMAIL-TRIGGER] Reply trigger already pending for {email.message_id}, skipping"
            )
            return

        scheduled_for = datetime.now(timezone.utc) + timedelta(
            seconds=max(delay_seconds, 0.0)
        )
        schedule = {
            "timezone": "UTC",
            "year": scheduled_for.year,
            "month": scheduled_for.month,
            "day": scheduled_for.day,
            "hour": scheduled_for.hour,
            "minute": scheduled_for.minute,
            "second": scheduled_for.second,
            "start_date": scheduled_for,
            "end_date": scheduled_for,
        }
        next_run = calculate_next_scheduled_run(schedule) or scheduled_for

        instructions = (settings.email_instructions or "").strip()
        context_parts = []
        if instructions:
            context_parts.append(
                f"<EmailInstructions>\n{instructions}\n</EmailInstructions>"
            )

        body_text = (message_content or email.snippet or "").strip()
        email_context = (
            "<InboundEmail>\n"
            f"From: {email.from_name or email.from_address}\n"
            f"FromAddress: {email.from_address}\n"
            f"To: {to_address_display or email.to_address or 'unknown'}\n"
            f"Subject: {email.subject or '(no subject)'}\n\n"
            f"{body_text or '(no body provided)'}\n"
            "</InboundEmail>"
        )
        context_parts.append(email_context)
        trigger_context = "\n\n".join(context_parts)

        base_prompt = (
            "Review the email in the trigger context and craft the reply you intend to send. "
            "Respond with the exact email body that should be delivered back to the sender. "
            "After composing the reply, you MUST call the `gmail_send` tool to deliver it."
        )
        trigger_prompt = (
            f"{base_prompt}\n\n<TriggerContext>\n{trigger_context}\n</TriggerContext>"
        )

        trigger_name = (
            email.subject or (email.from_name or email.from_address) or "Gmail reply"
        )

        update_config_payload = (
            update_config.model_dump(exclude_none=True)
            if hasattr(update_config, "model_dump")
            else update_config
        )

        api_url = get_api_url()
        headers = {
            "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
            "Content-Type": "application/json",
        }

        schedule_payload = {}
        for key, value in schedule.items():
            if isinstance(value, datetime):
                schedule_payload[key] = value.isoformat()
            else:
                schedule_payload[key] = value

        payload = {
            "agent": str(self.deployment.agent),
            "user": str(user.id),
            "name": trigger_name[:120],
            "context": trigger_context,
            "trigger_prompt": trigger_prompt,
            "posting_instructions": [],
            "schedule": schedule_payload,
            "session_type": "another",
            "session": str(session.id),
        }

        if update_config_payload:
            payload["update_config"] = update_config_payload

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{api_url}/triggers/create",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except Exception as exc:
            logger.error(
                f"[GMAIL-TRIGGER] Failed to create trigger via API: {exc}",
                exc_info=True,
            )
            raise

        logger.info(
            "[GMAIL-TRIGGER] Scheduled reply for %s at %s (delay=%.2fs)",
            email.message_id,
            next_run.isoformat(),
            delay_seconds,
        )

    def _create_gmail_session(
        self,
        *,
        user: User,
        session_key: str,
        session_key_basis: str,
        email: GmailInboundEmail,
    ) -> Session:
        if not self.deployment:
            raise ValueError("Deployment is required to create Gmail session")

        session = Session(
            owner=ObjectId(str(user.id)),
            agents=[ObjectId(self.deployment.agent)],
            title=email.subject or f"Gmail thread {session_key_basis}",
            session_key=session_key,
            platform="gmail",
            status="active",
            extras={
                "gmail_thread_id": email.thread_id,
                "gmail_initial_message_id": email.message_id,
            },
        )
        session.save()
        return session

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

    def _get_own_addresses(self) -> set[str]:
        if self._own_addresses is not None:
            return self._own_addresses

        secrets = self._get_secrets()
        settings = self._get_settings()
        addresses: set[str] = set()

        for value in (
            getattr(secrets, "reply_alias", None),
            secrets.delegated_user if secrets else None,
            settings.reply_from_address if settings else None,
        ):
            if value:
                addresses.add(value.lower())

        self._own_addresses = addresses
        return addresses

    def _compute_reply_delay(
        self,
        settings: DeploymentSettingsGmail,
        message_text: Optional[str],
    ) -> float:
        """Compute a delay using minimum/variance settings scaled by message length."""

        base = settings.reply_delay_minimum
        variance = settings.reply_delay_variance

        base = float(base or 0)
        variance = float(variance or 0)

        if variance <= 0:
            return max(0.0, base)

        length = len(message_text.strip()) if message_text else 0
        length_factor = min(1.0, length / MAX_LENGTH_FOR_FULL_VARIANCE)
        scaled_variance = variance * length_factor
        jitter = random.uniform(0, scaled_variance) if scaled_variance > 0 else 0
        return max(0.0, base + jitter)

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
            expiration_dt = datetime.fromtimestamp(
                expiration_int / 1000, tz=timezone.utc
            )
        except Exception as exc:
            logger.warning(
                f"[GMAIL] Invalid watch expiration value: {expiration_ms} ({exc})"
            )
            return

        try:
            Deployment.get_collection().update_one(
                {"_id": self.deployment.id},
                {"$set": {"config.gmail.watch_expiration": expiration_dt.isoformat()}},
            )
            self.deployment.reload()
        except Exception as exc:
            logger.warning(f"[GMAIL] Failed to persist watch expiration: {exc}")
