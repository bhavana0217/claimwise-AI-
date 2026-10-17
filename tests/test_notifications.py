import email
import smtplib
from unittest import mock

import pandas as pd

from src.notifications import (
    Alert, NotificationSettings, build_email, evaluate_events, parse_recipients, send_email,
    smtp_settings_from_env,
)


def queue_stub(n_high=3, amount=1000.0, extra_low=2):
    rows = [{"is_open": True, "priority_level": "High", "denied_amount": amount, "claim_id": f"CLM{i}"} for i in range(n_high)]
    rows += [{"is_open": True, "priority_level": "Low", "denied_amount": amount, "claim_id": f"CLMX{i}"} for i in range(extra_low)]
    rows += [{"is_open": False, "priority_level": "Closed", "denied_amount": 99999.0, "claim_id": "CLMC"}]
    return pd.DataFrame(rows)


def impact_stub():
    return pd.DataFrame([
        {"update_id": "U1", "severity": "High", "payer": "Cigna", "topic": "Prior auth",
         "effective_date": pd.Timestamp("2025-07-01"), "status": "Upcoming",
         "open_denials_affected": 9, "open_value_affected": 11571.0, "source": "SYNTHETIC DEMO"},
        {"update_id": "U2", "severity": "Low", "payer": "BCBS", "topic": "Fields",
         "effective_date": pd.Timestamp("2025-07-01"), "status": "Upcoming",
         "open_denials_affected": 0, "open_value_affected": 0.0, "source": "SYNTHETIC DEMO"},
    ])


def ids(alerts):
    return [a.event_id for a in alerts]


def test_parse_recipients_splits_and_validates():
    valid, invalid = parse_recipients("a@x.org, b@y.com; bad, c@d\nnew@z.io")
    assert valid == ["a@x.org", "b@y.com", "new@z.io"]
    assert invalid == ["bad", "c@d"]
    assert parse_recipients("") == ([], [])


def test_events_respect_thresholds_and_toggles():
    q = queue_stub(n_high=3, amount=1000)   # open total = 5 * 1000 = 5000 (closed row ignored)
    s = NotificationSettings(open_value_limit=5000)
    assert ids(evaluate_events(q, None, impact_stub(), s)) == ["HIGH_PRIORITY_DENIALS", "PAYER_UPDATE:U1", "OPEN_VALUE_LIMIT"]
    s = NotificationSettings(open_value_limit=5001, high_priority_min_count=4)
    assert ids(evaluate_events(q, None, impact_stub(), s)) == ["PAYER_UPDATE:U1"]
    s = NotificationSettings(high_priority_enabled=False, open_value_enabled=False, payer_update_enabled=False)
    assert evaluate_events(q, None, impact_stub(), s) == []


def test_payer_update_with_no_open_denials_does_not_alert():
    s = NotificationSettings(high_priority_enabled=False, open_value_enabled=False)
    out = evaluate_events(queue_stub(), None, impact_stub(), s)
    assert ids(out) == ["PAYER_UPDATE:U1"]


def test_checker_alert_uses_summary_and_threshold():
    summary = {"flagged_claims": 18, "billed_value_requiring_review": 43870.0}
    s = NotificationSettings(high_priority_enabled=False, open_value_enabled=False, payer_update_enabled=False)
    out = evaluate_events(queue_stub(), summary, impact_stub(), s)
    assert ids(out) == ["CHECKER_FLAGS"] and "18 claim(s)" in out[0].detail and "estimate" in out[0].detail
    s.checker_min_flagged = 19
    assert evaluate_events(queue_stub(), summary, impact_stub(), s) == []


def test_alerts_sorted_high_first():
    out = evaluate_events(queue_stub(), {"flagged_claims": 1, "billed_value_requiring_review": 5.0}, impact_stub(),
                          NotificationSettings(open_value_limit=1))
    order = [a.severity for a in out]
    assert order == sorted(order, key=lambda s: {"High": 0, "Medium": 1, "Low": 2}[s])


def test_email_contains_alerts_disclaimer_and_no_claim_ids():
    alerts = evaluate_events(queue_stub(), None, impact_stub(), NotificationSettings())
    msg = build_email(alerts, ["admin@hospital.org", "lead@hospital.org"])
    body = msg.get_content()
    assert msg["To"] == "admin@hospital.org, lead@hospital.org"
    assert f"{len(alerts)} alerts need attention" in msg["Subject"]
    assert "decision support only" in body.lower()
    assert "no patient information" in body
    assert "CLM" not in body and "P0" not in body  # no claim/patient identifiers
    assert all(a.title in body for a in alerts)


def test_single_alert_subject_grammar():
    msg = build_email([Alert("X", "Only one", "High", "detail")], ["a@b.co"])
    assert msg["Subject"] == "ClaimWise AI: 1 alert needs attention"


def test_send_without_smtp_saves_parseable_eml_and_reports_not_sent(tmp_path):
    msg = build_email([Alert("X", "Title", "High", "detail")], ["a@b.co"])
    result = send_email(msg, outbox_dir=tmp_path, smtp=None)
    assert result.sent is False and "nothing was sent" in result.message
    saved = email.message_from_bytes(open(result.saved_to, "rb").read())
    assert saved["Subject"] == msg["Subject"] and saved["To"] == "a@b.co"


def test_send_with_smtp_uses_tls_login_and_recipients():
    msg = build_email([Alert("X", "Title", "High", "detail")], ["a@b.co", "c@d.org"])
    smtp = {"host": "mail.example.org", "port": 587, "user": "u", "password": "p", "from": "claimwise@example.org"}
    with mock.patch("smtplib.SMTP") as smtp_cls:
        server = smtp_cls.return_value.__enter__.return_value
        result = send_email(msg, smtp=smtp)
    smtp_cls.assert_called_once_with("mail.example.org", 587, timeout=15)
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("u", "p")
    assert server.send_message.call_args.kwargs["to_addrs"] == ["a@b.co", "c@d.org"]
    assert result.sent is True and msg["From"] == "claimwise@example.org"


def test_smtp_failure_is_reported_not_raised():
    msg = build_email([Alert("X", "Title", "High", "detail")], ["a@b.co"])
    smtp = {"host": "mail.example.org", "port": 587, "user": None, "password": None, "from": "f@x.org"}
    with mock.patch("smtplib.SMTP", side_effect=smtplib.SMTPException("boom")):
        result = send_email(msg, smtp=smtp)
    assert result.sent is False and "boom" in result.message


def test_send_with_no_recipients_is_refused(tmp_path):
    msg = build_email([Alert("X", "Title", "High", "detail")], [])
    result = send_email(msg, outbox_dir=tmp_path)
    assert result.sent is False and "No valid recipients" in result.message
    assert list(tmp_path.iterdir()) == []


def test_smtp_settings_from_env():
    assert smtp_settings_from_env({}) is None
    cfg = smtp_settings_from_env({"CLAIMWISE_SMTP_HOST": "h", "CLAIMWISE_SMTP_USER": "me@x.org"})
    assert cfg["port"] == 587 and cfg["from"] == "me@x.org" and cfg["password"] is None
