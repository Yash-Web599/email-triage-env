import random
from faker import Faker
from .models import Email

fake = Faker()

SPAM_SUBJECTS = [
    "You won a prize!",
    "Click here to claim your reward",
    "URGENT: Your account will be closed",
    "Make money fast!!!",
    "Free iPhone giveaway",
]

REAL_SUBJECTS = [
    "Team meeting tomorrow at 10am",
    "Project deadline reminder",
    "Invoice attached for your review",
    "Follow up on our last discussion",
    "Urgent: Server is down",
    "Q3 report ready for review",
    "Client complaint needs attention",
]

SPAM_BODIES = [
    "Congratulations! You have been selected. Click here now!",
    "Dear user, your account needs verification. Send your password.",
    "Win $10,000 cash! Limited time offer. Act now!",
]

REAL_BODIES = [
    "Hi team, please join the standup call tomorrow at 10am sharp.",
    "This is a reminder that the project deadline is this Friday.",
    "Please find attached the invoice for last month services.",
    "Following up on our conversation, can we schedule a call?",
    "URGENT: The production server is down. Please investigate immediately.",
]


def generate_email(email_id: str, force_spam: bool = False,
                   force_real: bool = False) -> Email:
    if force_spam or (not force_real and random.random() < 0.4):
        subject = random.choice(SPAM_SUBJECTS)
        body = random.choice(SPAM_BODIES)
        is_urgent = False
    else:
        subject = random.choice(REAL_SUBJECTS)
        body = random.choice(REAL_BODIES)
        is_urgent = "URGENT" in subject or "URGENT" in body

    return Email(
        id=email_id,
        subject=subject,
        sender=fake.email(),
        body=body,
        timestamp=fake.date_time_this_month().isoformat(),
        is_urgent=is_urgent,
    )


def generate_inbox(size: int = 5) -> list:
    emails = []
    # Always include at least one spam and one real
    emails.append(generate_email("email_0", force_spam=True))
    emails.append(generate_email("email_1", force_real=True))
    for i in range(2, size):
        emails.append(generate_email(f"email_{i}"))
    random.shuffle(emails)
    return emails