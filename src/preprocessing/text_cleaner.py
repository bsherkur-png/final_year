"""Deterministic boilerplate removal for extracted article text.

Patterns were identified by manual inspection of all 15 articles
in the Shamima Begum corpus. Each outlet's non-editorial content
(comment prompts, bylines, privacy notices, image captions,
copyright footers) is matched by exact-phrase regex and removed.
"""

import re


# Boilerplate patterns identified by manual corpus inspection.
# Each tuple is (pattern_type, compiled_regex).
# "head" patterns are removed from the start of text.
# "tail" patterns truncate text at the match position.
# "inline" patterns are removed wherever they appear.
_BOILERPLATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Daily Mail: "By NAME Published: ... View comments"
    ("head", re.compile(
        r"^By\s+[A-Z\s,]+(?:FOR\s+[A-Z\s]+)?\s*"
        r"Published:\s*[\d:,\s]+\w+\s+\d{4}\s*"
        r"(?:\|\s*Updated:\s*[\d:,\s]+\w+\s+\d{4}\s*)?"
        r"[\d.]+[k]?\s*View\s+comments\s*",
        re.IGNORECASE,
    )),
    # The Independent: subheadline + "Removed from bookmarks ... Privacy notice"
    ("head", re.compile(
        r"^.{0,300}?Removed\s+from\s+bookmarks\s+"
        r"I\s+would\s+like\s+to\s+be\s+emailed\s+about\s+offers.*?"
        r"Read\s+our\s+Privacy\s+notice\s*",
        re.IGNORECASE | re.DOTALL,
    )),
    # Daily Mail: comment section block
    ("tail", re.compile(
        r"\s*Share\s+what\s+you\s+think\s+The\s+comments\s+below\b",
        re.IGNORECASE,
    )),
    # The Independent: engagement CTA
    ("tail", re.compile(
        r"\s*Join\s+thought-provoking\s+conversations,\s+follow\s+other\s+"
        r"Independent\s+readers\b",
        re.IGNORECASE,
    )),
    # BBC: copyright footer (English and non-English variants)
    ("tail", re.compile(
        r"\s*(?:Copyright|©)\s+20\d{2}\s+BBC\b",
        re.IGNORECASE,
    )),
    # BBC: social/email CTA
    ("tail", re.compile(
        r"\s*Follow\s+BBC\s+\w+\s+on\s+Facebook\b",
        re.IGNORECASE,
    )),
    # The Guardian: letters page CTA
    ("tail", re.compile(
        r"\s*Join\s+the\s+debate\s+.{0,10}email\s+guardian\.letters\b",
        re.IGNORECASE,
    )),
    # The Mirror: TV scheduling
    ("tail", re.compile(
        r"\s*\*This\s+Morning\s+airs\b",
        re.IGNORECASE,
    )),
    # The Mirror: contact CTA
    ("tail", re.compile(
        r"\s*Do\s+you\s+have\s+a\s+story\s+to\s+sell\?\s+Get\s+in\s+touch\b",
        re.IGNORECASE,
    )),
    # The Mirror: inline image captions — "(Image: SOURCE)"
    ("inline", re.compile(
        r"\(Image:\s*[^)]+\)",
        re.IGNORECASE,
    )),
]


def strip_boilerplate(text: str) -> str:
    """Remove non-editorial boilerplate identified by corpus inspection.

    Applies head patterns (strip from start), inline patterns (remove
    all occurrences), then tail patterns (truncate at first match).

    Args:
        text: Raw extracted article text.

    Returns:
        Article text with boilerplate removed.
    """
    for pattern_type, pattern in _BOILERPLATE_PATTERNS:
        if pattern_type == "head":
            text = pattern.sub("", text, count=1)

    for pattern_type, pattern in _BOILERPLATE_PATTERNS:
        if pattern_type == "inline":
            text = pattern.sub("", text)

    for pattern_type, pattern in _BOILERPLATE_PATTERNS:
        if pattern_type == "tail":
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
                break

    return text
