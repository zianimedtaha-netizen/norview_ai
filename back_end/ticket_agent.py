import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ADMIN_EMAIL     = os.getenv("ADMIN_EMAIL", "support@emsi.ma")
SENDER_EMAIL    = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")

def create_ticket(question: str, answer: str, session_id: str = "default") -> str:

    # Extract fields
    name        = question.split('Name: ')[1].split('\n')[0].strip()        if 'Name: '        in question else ""
    email       = question.split('Email: ')[1].split('\n')[0].strip()       if 'Email: '       in question else ""
    subject     = question.split('Subject: ')[1].split('\n')[0].strip()     if 'Subject: '     in question else ""
    description = question.split('Description: ')[1].strip()                if 'Description: ' in question else question.strip()

    # Block empty tickets
    if not name or name == "N/A":
        print(f"[Ticket] Blocked — missing name")
        return "BLOCKED"
    if not email or email == "N/A" or "@" not in email:
        print(f"[Ticket] Blocked — missing or invalid email")
        return "BLOCKED"
    if not subject or subject == "N/A":
        print(f"[Ticket] Blocked — missing subject")
        return "BLOCKED"
    if not description or len(description.strip()) < 10:
        print(f"[Ticket] Blocked — description too short")
        return "BLOCKED"

    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    ticket_body = f"""
NEW SUPPORT TICKET
==================
Ticket ID:  {ticket_id}
Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {session_id}

STUDENT INFORMATION:
- Name:  {name}
- Email: {email}

ISSUE:
- Subject:     {subject}
- Description: {description}

ACTION REQUIRED: Please contact the student as soon as possible.
"""

    if SENDER_EMAIL and SENDER_PASSWORD:
        try:
            msg = MIMEMultipart()
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = ADMIN_EMAIL
            msg["Subject"] = f"[NorView] Support Ticket — {subject} ({ticket_id})"
            msg.attach(MIMEText(ticket_body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
            print(f"[Ticket] {ticket_id} sent to {ADMIN_EMAIL}")
        except Exception as e:
            print(f"[Ticket] Email failed: {e}")
    else:
        print(f"[Ticket] {ticket_id} created (no email configured)")

    return ticket_id