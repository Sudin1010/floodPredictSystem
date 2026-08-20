import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailDeliveryResult:
    success: bool
    disabled: bool = False
    error_message: str | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_alert_email(
    *,
    sender: str,
    recipient_email: str,
    alert_title: str,
    alert_message: str,
    district: str,
    probability: float,
    risk_level: str,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient_email
    message["Subject"] = alert_title
    message.set_content(
        "\n".join(
            [
                "Flood Prediction System",
                "",
                alert_title,
                "",
                f"District: {district}",
                f"Risk Level: {risk_level}",
                f"Estimated Flood Probability: {probability}%",
                "",
                alert_message,
                "",
                "This alert was reviewed through the Flood Prediction System.",
                "",
                "Prototype Disclaimer:",
                "This system is an academic/prototype decision-support tool and should support, not replace, official emergency information.",
            ]
        )
    )
    return message


def send_alert_email(
    *,
    recipient_email: str,
    alert_title: str,
    alert_message: str,
    district: str,
    probability: float,
    risk_level: str,
) -> EmailDeliveryResult:
    if not _env_flag("EMAIL_DELIVERY_ENABLED", default=False):
        logger.info("Email delivery disabled; alert email was not sent.")
        return EmailDeliveryResult(
            success=False,
            disabled=True,
            error_message="Email delivery is disabled.",
        )

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port_value = os.getenv("SMTP_PORT", "587").strip()
    smtp_username = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
    smtp_use_tls = _env_flag("SMTP_USE_TLS", default=True)

    if not smtp_host or not smtp_from_email:
        logger.error("SMTP configuration is incomplete; alert email was not sent.")
        return EmailDeliveryResult(
            success=False,
            error_message="Email service is not configured.",
        )

    try:
        smtp_port = int(smtp_port_value)
    except ValueError:
        logger.error("SMTP_PORT must be an integer; alert email was not sent.")
        return EmailDeliveryResult(
            success=False,
            error_message="Email service is not configured.",
        )

    email_message = _build_alert_email(
        sender=smtp_from_email,
        recipient_email=recipient_email,
        alert_title=alert_title,
        alert_message=alert_message,
        district=district,
        probability=probability,
        risk_level=risk_level,
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if smtp_use_tls:
                smtp.starttls()
            if smtp_username or smtp_password:
                smtp.login(smtp_username, smtp_password)
            smtp.send_message(email_message)
    except Exception as exc:
        logger.warning("Alert email delivery failed for one recipient: %s", exc.__class__.__name__)
        return EmailDeliveryResult(
            success=False,
            error_message="Email delivery failed for one recipient.",
        )

    return EmailDeliveryResult(success=True)
