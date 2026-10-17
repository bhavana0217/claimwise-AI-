"""Email notifications for hospital administrators.

Design choices:
  * Messages contain only counts and dollar totals. No patient names, patient IDs, or claim
    IDs are put in an email. A real deployment would still need privacy/compliance review.
  * Nothing is sent unless an SMTP server is configured through environment variables.
    Otherwise the message is saved as a .eml preview file and the caller is told so.
  * Passwords are never stored in the project; they come from the environment only.

Environment variables for real sending:
    CLAIMWISE_SMTP_HOST, CLAIMWISE_SMTP_PORT (default 587), CLAIMWISE_SMTP_USER,
    CLAIMWISE_SMTP_PASSWORD, CLAIMWISE_FROM
"""
from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

EMAIL_PATTERN = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")

DISCLAIMER = (
    "ClaimWise AI is decision support only. Results require human review and are not a "
    "substitute for payer policy verification."
)


@dataclass
class NotificationSettings:
    """Which events notify, and their thresholds."""

    high_priority_enabled: bool = True
    high_priority_min_count: int = 1
    open_value_enabled: bool = True
    open_value_limit: float = 1_000_000.0
    payer_update_enabled: bool = True
    checker_enabled: bool = True
    checker_min_flagged: int = 1


@dataclass
class Alert:
    event_id: str
    title: str
    severity: str  # High / Medium / Low
    detail: str


@dataclass
class SendResult:
    sent: bool
    message: str
    saved_to: str | None = None
    recipients: list[str] = field(default_factory=list)


def parse_recipients(text: str) -> tuple[list[str], list[str]]:
    """Split a comma/semicolon/newline separated list into (valid, invalid) addresses."""
    parts = [p.strip() for p in re.split(r"[,;\n]", text or "") if p.strip()]
    valid = [p for p in parts if EMAIL_PATTERN.match(p)]
    invalid = [p for p in parts if not EMAIL_PATTERN.match(p)]
    return valid, invalid


def evaluate_events(
    queue: pd.DataFrame,
    checker_summary: dict | None,
    impact: pd.DataFrame,
    settings: NotificationSettings,
) -> list[Alert]:
    """Return the alerts that are currently triggered under the given settings."""
    alerts: list[Alert] = []
    open_q = queue[queue["is_open"]] if len(queue) else queue

    if settings.high_priority_enabled and len(open_q):
        high = open_q[open_q["priority_level"] == "High"]
        if len(high) >= settings.high_priority_min_count:
            alerts.append(
                Alert(
                    "HIGH_PRIORITY_DENIALS",
                    "High-priority open denials need follow-up",
                    "High",
                    f"{len(high):,} open denial(s) are rated High priority, totaling "
                    f"${high['denied_amount'].fillna(0).sum():,.0f} in denied amount.",
                )
            )

    if settings.open_value_enabled and len(open_q):
        total = float(open_q["denied_amount"].fillna(0).sum())
        if total >= settings.open_value_limit:
            alerts.append(
                Alert(
                    "OPEN_VALUE_LIMIT",
                    "Open denied value is above your limit",
                    "Medium",
                    f"Open denials total ${total:,.0f}, above the ${settings.open_value_limit:,.0f} limit "
                    f"({len(open_q):,} denials).",
                )
            )

    if settings.payer_update_enabled and len(impact):
        hit = impact[(impact["open_denials_affected"] > 0) & (impact["status"] != "Date unknown")]
        for row in hit.itertuples(index=False):
            alerts.append(
                Alert(
                    f"PAYER_UPDATE:{row.update_id}",
                    f"{row.payer} rule update affects open denials",
                    row.severity,
                    f"'{row.topic}' ({row.status.lower()}, effective {row.effective_date:%Y-%m-%d}) relates to "
                    f"{row.open_denials_affected:,} open denial(s) worth ${row.open_value_affected:,.0f}. "
                    f"Source: {row.source}.",
                )
            )

    if settings.checker_enabled and checker_summary:
        n = checker_summary.get("flagged_claims", 0)
        if n >= settings.checker_min_flagged:
            alerts.append(
                Alert(
                    "CHECKER_FLAGS",
                    "Claims were flagged for review before submission",
                    "Medium",
                    f"{n:,} claim(s) were flagged by the claim checker, with an estimated "
                    f"${checker_summary.get('billed_value_requiring_review', 0):,.0f} of billed value "
                    "requiring review (an estimate, not a guaranteed loss).",
                )
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(alerts, key=lambda a: order.get(a.severity, 3))


def build_email(
    alerts: list[Alert],
    recipients: list[str],
    sender: str = "claimwise@localhost",
    dashboard_url: str = "http://localhost:8501",
) -> EmailMessage:
    """Build the notification email (plain text; aggregates only)."""
    msg = EmailMessage()
    n = len(alerts)
    msg["Subject"] = f"ClaimWise AI: {n} alert{'s' if n != 1 else ''} need{'s' if n == 1 else ''} attention"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    lines = ["Hello,", "", f"ClaimWise AI found {n} item(s) for your attention:", ""]
    for i, a in enumerate(alerts, start=1):
        lines += [f"{i}. [{a.severity}] {a.title}", f"   {a.detail}", ""]
    lines += [
        f"Open the dashboard for details: {dashboard_url}",
        "",
        "This email contains totals only - no patient information.",
        DISCLAIMER,
    ]
    msg.set_content("\n".join(lines))
    return msg


def smtp_settings_from_env(env=None) -> dict | None:
    """Return SMTP settings if configured, else None."""
    env = os.environ if env is None else env
    host = env.get("CLAIMWISE_SMTP_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(env.get("CLAIMWISE_SMTP_PORT", "587")),
        "user": env.get("CLAIMWISE_SMTP_USER"),
        "password": env.get("CLAIMWISE_SMTP_PASSWORD"),
        "from": env.get("CLAIMWISE_FROM") or env.get("CLAIMWISE_SMTP_USER") or "claimwise@localhost",
    }


def send_email(msg: EmailMessage, outbox_dir: str | Path = "output/outbox", smtp: dict | None = None) -> SendResult:
    """Send through SMTP if configured; otherwise save a preview .eml and say so."""
    recipients = [r.strip() for r in str(msg["To"]).split(",") if r.strip()]
    if not recipients:
        return SendResult(False, "No valid recipients were provided.")

    if smtp is None:
        outbox = Path(outbox_dir)
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / f"claimwise_alert_{datetime.now():%Y%m%d_%H%M%S_%f}.eml"
        path.write_bytes(bytes(msg))
        return SendResult(
            False,
            "No email server is configured, so nothing was sent. The message was saved as a preview file.",
            saved_to=str(path),
            recipients=recipients,
        )

    msg.replace_header("From", smtp["from"])
    try:
        with smtplib.SMTP(smtp["host"], smtp["port"], timeout=15) as server:
            server.starttls()
            if smtp.get("user") and smtp.get("password"):
                server.login(smtp["user"], smtp["password"])
            server.send_message(msg, to_addrs=recipients)
    except (smtplib.SMTPException, OSError) as exc:
        return SendResult(False, f"The email could not be sent: {exc}", recipients=recipients)
    return SendResult(True, f"Email sent to {len(recipients)} recipient(s).", recipients=recipients)
