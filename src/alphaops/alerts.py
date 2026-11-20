"""Push alerts to Slack (incoming webhook) and/or Telegram (bot API).

Alerting is best-effort: a broken webhook must never crash the job it is reporting on,
so every failure is logged and swallowed. With no channel configured the alert still
appears in the structured log, so it is never lost silently.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from alphaops.config import Settings

log = logging.getLogger(__name__)

ICONS = {"critical": "🔴", "warning": "🟠", "info": "🟢"}


def _post(url: str, data: bytes, content_type: str, timeout: float = 10.0) -> None:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def send_alert(settings: Settings, title: str, body: str, severity: str = "warning") -> int:
    """Send to every configured channel. Returns the number of channels that accepted it."""
    text = f"{ICONS.get(severity, '')} [{severity.upper()}] {title}\n{body}".strip()
    log.log(
        logging.ERROR if severity == "critical" else logging.WARNING,
        "alert",
        extra={"alert_title": title, "severity": severity, "body": body},
    )
    delivered = 0
    if settings.slack_webhook_url:
        try:
            _post(
                settings.slack_webhook_url,
                json.dumps({"text": text}).encode(),
                "application/json",
            )
            delivered += 1
        except Exception as exc:
            log.warning("slack alert failed", extra={"error": repr(exc)})
    if settings.telegram_bot_token and settings.telegram_chat_id:
        try:
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            payload = urllib.parse.urlencode(
                {"chat_id": settings.telegram_chat_id, "text": text}
            ).encode()
            _post(url, payload, "application/x-www-form-urlencoded")
            delivered += 1
        except Exception as exc:
            # Don't log the URL: it contains the bot token.
            log.warning("telegram alert failed", extra={"error": type(exc).__name__})
    return delivered
