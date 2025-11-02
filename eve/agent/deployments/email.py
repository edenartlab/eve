import os
from typing import Optional

import aiohttp
from fastapi import Request
from loguru import logger

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import (
    DeploymentConfig,
    DeploymentSecrets,
    UpdateType,
)


class EmailClient(PlatformClient):
    TOOLS = ["email_send"]

    @staticmethod
    def _get_api_credentials() -> tuple[str, str]:
        base_url = os.getenv("MAILGUN_API_BASE_URL", "https://api.mailgun.net")
        api_key = os.getenv("MAILGUN_API_KEY")
        if not api_key:
            raise APIError("MAILGUN_API_KEY is not configured", status_code=500)
        return base_url.rstrip("/"), api_key

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        if not secrets or not secrets.email or not secrets.email.domain:
            raise APIError("Email domain is required", status_code=400)

        if not config or not config.email:
            raise APIError("Email configuration is required", status_code=400)

        sender_email = config.email.sender_email
        if not sender_email:
            raise APIError("Sender email is required", status_code=400)
        if sender_email and "@" in sender_email:
            sender_domain = sender_email.split("@")[-1].lower()
            configured_domain = secrets.email.domain.lower()
            if sender_domain != configured_domain:
                raise APIError(
                    "Sender email must use the verified domain",
                    status_code=400,
                )

        avg_delay = config.email.reply_delay_average_minutes
        variance_delay = config.email.reply_delay_variance_minutes

        if avg_delay is not None and avg_delay < 0:
            raise APIError("Average reply delay must be non-negative", status_code=400)

        if variance_delay is not None and variance_delay < 0:
            raise APIError("Reply delay variance must be non-negative", status_code=400)

        try:
            self.add_tools()
        except Exception as exc:
            raise APIError(f"Failed to add email tools: {exc}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        # Nothing to do post deploy for email currently
        return

    async def stop(self) -> None:
        try:
            self.remove_tools()
        except Exception as exc:
            logger.error(f"Failed to remove email tools: {exc}")

    async def interact(self, request: Request) -> None:
        raise NotImplementedError("Interact is not supported for email deployments")

    async def handle_emission(self, emission) -> None:
        if not self.deployment:
            raise ValueError("Deployment is required for handle_emission")

        if emission.type != UpdateType.ASSISTANT_MESSAGE:
            # Only handle final assistant messages for outbound email
            return

        update_config = emission.update_config
        if not update_config:
            logger.debug("Email emission missing update_config")
            return

        recipient = update_config.email_sender
        if not recipient:
            logger.debug("Email emission missing recipient")
            return

        content = emission.content
        if not content:
            logger.debug("Email emission missing content")
            return

        subject = update_config.email_subject or "Message from your Eden agent"
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        in_reply_to = update_config.email_message_id
        references = update_config.email_thread_id or in_reply_to

        headers = {}
        if in_reply_to:
            headers["In-Reply-To"] = in_reply_to
        if references:
            headers["References"] = references

        await self.send_email(
            to_address=recipient,
            subject=subject,
            text_content=content,
            custom_headers=headers,
        )

    def _build_from_address(self) -> str:
        config = self.deployment.config.email if self.deployment else None
        secrets = self.deployment.secrets.email if self.deployment else None
        agent_name = getattr(self.agent, "name", "Eden Agent")

        if config and config.sender_email:
            return f"{agent_name} <{config.sender_email}>"

        domain = secrets.domain if secrets else None
        fallback_email = f"no-reply@{domain}" if domain else "no-reply@eden.art"
        return f"{agent_name} <{fallback_email}>"

    async def send_email(
        self,
        to_address: str,
        subject: str,
        text_content: str,
        html_content: Optional[str] = None,
        reply_to: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
    ) -> dict:
        base_url, api_key = self._get_api_credentials()

        secrets = self.deployment.secrets.email
        if not secrets or not secrets.domain:
            raise APIError("Email domain is not configured", status_code=500)

        domain = secrets.domain
        from_address = self._build_from_address()

        data: dict[str, str] = {
            "from": from_address,
            "to": to_address,
            "subject": subject,
            "text": text_content,
        }

        if html_content:
            data["html"] = html_content

        if reply_to:
            data["h:Reply-To"] = reply_to

        if custom_headers:
            for header, value in custom_headers.items():
                data[f"h:{header}"] = value

        auth = aiohttp.BasicAuth("api", api_key)
        async with aiohttp.ClientSession(auth=auth) as session:
            url = f"{base_url}/v3/{domain}/messages"
            async with session.post(url, data=data) as response:
                response_text = await response.text()
                if response.status >= 400:
                    logger.error(
                        f"Email send failed ({response.status}): {response_text}"
                    )
                    raise APIError(
                        f"Failed to send email: {response_text}",
                        status_code=500,
                    )

                logger.info(
                    f"Sent email to {to_address} from deployment {self.deployment.id}"
                )

                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type.lower():
                    try:
                        return await response.json()
                    except Exception:
                        return {"message": response_text}

                return {"message": response_text}
