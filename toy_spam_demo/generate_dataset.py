from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
ROWS_PER_CLASS = 1200
OUT_PATH = Path(__file__).resolve().parent / "emails.csv"

ham_templates = [
    "Meeting at {time} {day}.",
    "Please review the attached {document}.",
    "Your order {status}.",
    "Lunch at {time}?",
    "Project deadline extended to {day}.",
    "Reminder: Appointment {day}.",
    "Your bill has been generated.",
    "Let us catch up {when}.",
    "The {event} starts at {time}.",
    "Please verify your attendance for {event}.",
    "Your OTP is {otp}.",
    "The package will arrive {day}.",
    "Could you share the updated {document}?",
    "Meeting notes from {day} are attached.",
    "We have moved the call to {time}.",
]

spam_templates = [
    "Congratulations! You won a {prize}!",
    "Earn Rs. {amount} per week from home!",
    "Click here to claim your {prize}.",
    "Urgent! Verify your account now.",
    "Lose weight in {days} days!",
    "Exclusive deal just for you!",
    "Free {item} waiting!",
    "Make money fast!",
    "Limited time offer!",
    "You are a lucky winner!",
    "Claim your bonus now and get {prize} instantly.",
    "Act now to unlock your reward.",
    "Free access to premium content.",
    "Last chance to save big today!",
    "Get rich quick with this simple trick.",
]

ham_vocab = {
    "time": ["9 AM", "10 AM", "1 PM", "2:30 PM", "5 PM", "tomorrow morning", "this afternoon"],
    "day": ["tomorrow", "on Monday", "on Tuesday", "this Friday", "next week"],
    "document": ["document", "report", "presentation", "invoice", "agenda", "slide deck"],
    "status": ["has been shipped", "is ready for pickup", "has been approved", "was updated", "is due tomorrow"],
    "when": ["this evening", "on Saturday", "next week", "after lunch", "later today"],
    "event": ["team meeting", "client call", "seminar", "interview", "workshop", "lecture"],
    "otp": ["123456", "678901", "246810", "975310"],
}

spam_vocab = {
    "prize": ["lottery", "gift card", "cash reward", "bonus", "voucher", "free trip"],
    "amount": ["1,00,000", "50,000", "75,000", "2,00,000"],
    "days": ["3", "5", "7", "10"],
    "item": ["gift card", "coupon", "voucher", "coupon code", "subscription"],
}

ham_noise = [
    "Thanks for the update.",
    "See you soon.",
    "Attached is the latest file.",
    "Please let me know if that works.",
    "Kind reminder for the schedule.",
]

spam_noise = [
    "Reply stop to unsubscribe.",
    "Offer valid for today only.",
    "Terms and conditions apply.",
    "This is a limited time promotion.",
    "No credit card required.",
]


def fill_template(template: str, vocab: dict[str, list[str]], extra_suffix: list[str]) -> str:
    message = template.format(**{key: random.choice(values) for key, values in vocab.items()})
    if random.random() < 0.35:
        message = f"{message} {random.choice(extra_suffix)}"
    return message


def build_rows() -> list[list[str]]:
    random.seed(RANDOM_SEED)
    rows: list[list[str]] = []
    for _ in range(ROWS_PER_CLASS):
        rows.append(["ham", fill_template(random.choice(ham_templates), ham_vocab, ham_noise)])
        rows.append(["spam", fill_template(random.choice(spam_templates), spam_vocab, spam_noise)])
    random.shuffle(rows)
    return rows


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Dataset generated: {OUT_PATH}")
    print(df.head().to_string(index=False))
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
