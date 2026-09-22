"""Best-effort, regex-based PII redaction for training text.

This is **not** a substitute for reviewing your training data, and it is not a proper PII-detection model (no
named-entity recognition, no locale-aware address/phone formats, no context awareness). It matches a handful of
common, high-confidence shapes — email addresses, phone numbers in common separated formats, US SSN-shaped
numbers, credit-card-shaped numbers, and IPv4 addresses — with plain regexes, and nothing else. Expect real false
negatives (anything not matching these exact shapes slips through untouched) and occasional false positives (a
non-PII number that happens to match a credit-card shape). Treat it as one layer, not the whole plan for keeping
PII out of a trained model — see SECURITY.md's "Training data" section.
"""

import re
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d{4}[-\s]){3}\d{4}(?!\d)"),
    "IPV4": re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"),
}


@dataclass
class PIIRedactor:
    """Call ``redact(text)`` on every row of text before it reaches a tokenizer; ``report()`` at the end for a
    per-category match count (how much was actually redacted, not what leaked through)."""

    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in _PATTERNS})

    def redact(self, text: str) -> str:
        for name, pattern in _PATTERNS.items():
            text, matched = pattern.subn(f"<{name}>", text)
            if matched:
                self.counts[name] += matched
        return text

    def report(self) -> dict[str, int]:
        return dict(self.counts)

    def total(self) -> int:
        return sum(self.counts.values())
