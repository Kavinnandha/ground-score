"""Text normalisation and thread-level filters.

Everything here is pure and deterministic so it can be unit-tested without the
dataset, and so re-running the pipeline never silently changes what got
embedded.

The normalisation is deliberately conservative. Aggressive cleaning (stripping
emoji, casefolding, removing punctuation) would destroy exactly the signals the
routing step needs: ALL CAPS, repeated punctuation and rage emoji are among the
strongest predictors that a message should go to a human. So normalisation only
removes things that are noise for *retrieval* -- handles, URLs, whitespace --
and leaves affect intact.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# @handle at the start of a tweet is routing metadata, not content: every
# inbound support tweet opens with "@BrandHelp". Mid-tweet handles are usually
# also noise (other users pulled into the thread).
_HANDLE_RE = re.compile(r"@\w{1,15}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# The dataset ships pre-anonymised @ mentions as tokens like "@115712".
_ANON_HANDLE_RE = re.compile(r"@\d+")

# Deflection = "we can't help you here, move to DMs". Measured per brand in
# profile_brands.py because a brand that deflects everything makes the whole
# "draft a grounded reply" task vacuous -- there is nothing to ground in.
_DEFLECTION_PATTERNS = [
    r"\bdm\b", r"\bdms\b", r"direct message", r"send us a (?:message|dm|note)",
    r"(?:shoot|slide|drop) (?:us|me) a", r"follow (?:us|and)", r"private message",
    r"\bpm us\b", r"message us",
]
_DEFLECTION_RE = re.compile("|".join(_DEFLECTION_PATTERNS), re.IGNORECASE)

# Cheap script check. Full language ID (fasttext/langdetect) is another
# dependency for a marginal gain on 100-character strings; the ratio of Latin
# characters separates the cases that actually matter here.
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalise(text: str) -> str:
    """Canonical form used for embedding and for display in the labelling CLI."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub("<URL>", text)
    text = _ANON_HANDLE_RE.sub("", text)
    text = _HANDLE_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS_RE.sub(" ", text).strip()


def is_deflection(reply: str) -> bool:
    """True if a brand reply just pushes the customer to a private channel."""
    return bool(_DEFLECTION_RE.search(reply or ""))


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _LATIN_RE.match(c)) / len(letters)


def is_usable_customer_message(text: str, *, min_chars: int = 15, min_latin: float = 0.6) -> bool:
    """Filter for messages worth putting in the corpus or the golden pool.

    Note the asymmetry with the adversarial slice: ultra-short and emoji-heavy
    messages are excluded from the *retrieval corpus* (they teach nothing) but
    are deliberately hand-picked back into the golden set, because they are
    exactly where an auto-reply agent fails. See data/golden/LABELING_NOTES.md.
    """
    if len(text) < min_chars:
        return False
    if latin_ratio(text) < min_latin:
        return False
    return True


def dedupe_key(text: str) -> str:
    """Key for near-duplicate collapse.

    Casefold + strip non-alphanumerics so "Spotify is DOWN!!!" and "spotify is
    down" collapse. Used only for corpus dedup, never for evaluation, because
    the transform is lossy in ways that matter for scoring.
    """
    reduced = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    reduced = _WS_RE.sub(" ", reduced).strip()
    return hashlib.sha1(reduced.encode("utf-8")).hexdigest()


def stable_bucket(key: str, buckets: int = 100) -> int:
    """Deterministic hash bucket in [0, buckets).

    Python's builtin hash() is salted per process, so it cannot be used for a
    split that must stay identical across runs and machines.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets
