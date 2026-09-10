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

# Deflection = "we can't help you here, move somewhere private". Measured per
# brand in profile_brands.py because a brand that deflects everything makes the
# "draft a grounded reply" task vacuous -- there is nothing to ground in.
#
# There are two distinct forms, and conflating them was a real error in the
# first version of this file. DM-style deflection is what most brands in this
# dataset do. AmazonHelp almost never says "DM us" (0.7%) but hands off to a
# contact page 9.5% of the time -- measuring only the first form undercounted
# its true handoff rate by 13x. Both are reported separately.
_DM_PATTERNS = [
    r"\bdm\b", r"\bdms\b", r"direct message", r"send us a (?:message|dm|note)",
    r"(?:shoot|slide|drop) (?:us|me) a", r"follow (?:us|and)", r"private message",
    r"\bpm us\b", r"message us",
]
_DEFLECTION_RE = re.compile("|".join(_DM_PATTERNS), re.IGNORECASE)

# "Take this elsewhere" via a link or phone number rather than a DM. Requires
# BOTH a contact verb and a URL: ~46% of AmazonHelp replies contain a URL, and
# most of those are genuinely helpful (a tracking page, a help article), so a
# bare URL is not evidence of a handoff.
_CONTACT_VERB_RE = re.compile(
    r"(get in touch|reach (?:us|out to us)|contact us|give us a (?:ring|call)|"
    r"call us|speak (?:to|with) (?:us|our)|use this link)",
    re.IGNORECASE,
)

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
    """True if a brand reply pushes the customer into DMs."""
    return bool(_DEFLECTION_RE.search(reply or ""))


def is_link_handoff(reply: str) -> bool:
    """True if the reply routes the customer to a contact page or phone line."""
    reply = reply or ""
    return bool(_CONTACT_VERB_RE.search(reply)) and "<URL>" in reply


def is_handoff(reply: str) -> bool:
    """Either form of 'not resolved here'. This is the metric that matters."""
    return is_deflection(reply) or is_link_handoff(reply)


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _LATIN_RE.match(c)) / len(letters)


# English function words. Deliberately words with no cognate in the Romance or
# Germanic languages that dominate this queue's non-English traffic, so the
# ratio separates them rather than scoring them as borderline.
_EN_STOPWORDS = frozenset("""
the and is was are were be been have has had do does did will would can could
should my your our their this that these those with from about for not you
they it its there here what when where why how please thank thanks still yet
been get got give need want know said says because but they've i'm don't
""".split())

# Characters essentially absent from English but common in the languages seen
# in this corpus (es/fr/de/pt).
_NON_EN_CHARS = re.compile(r"[ñçãõäöüßéèêàâîôûíóúáêÊ¿¡]", re.IGNORECASE)

# Function words from the four languages that actually appear in this queue.
# Absence of English is NOT evidence of another language -- plenty of real
# English tweets ("ur app shows expected delivery on 25th") carry almost no
# function words and were being wrongly rejected by a ratio test alone. So the
# filter requires positive evidence of a different language.
_FOREIGN_STOPWORDS = frozenset("""
que de la el los las una uno por para con sin pero como cuando donde muy este
esta esto mi tu su nos ya no si sobre entre hasta desde
le les des du au aux et est sont ete pour avec sans mais comme quand ou je tu
il elle nous vous ils elles ce cette mon ton son pas plus tres
der die das den dem ein eine einer und oder aber ist sind war waren nicht ich
du er sie es wir ihr mein dein sein mit von zu auf fur uber sehr noch schon
nao sim uma um dos das para com sem mas como quando onde muito este esta isso
meu teu seu nos ja sobre entre ate desde voce esta estao
""".split())


def foreign_score(text: str) -> float:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _FOREIGN_STOPWORDS) / len(tokens)


def english_score(text: str) -> float:
    """Share of tokens that are common English function words.

    A dependency-free language check. `latin_ratio` is not sufficient: Spanish,
    French, German and Portuguese are all Latin-script, and clustering revealed
    that 16.6% of the corpus was non-English traffic that had passed straight
    through the script filter -- producing four clusters that were languages
    rather than intents. Function-word ratio separates them cleanly because
    these words have no cognates in those languages.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _EN_STOPWORDS) / len(tokens)


def is_probably_english(text: str) -> bool:
    """Cheap English filter requiring positive evidence of another language.

    Rejects only when a message looks *more* like one of the four languages
    actually present in this queue than like English. A pure "not enough English
    function words" test over-rejects: terse but genuinely English tweets
    ("ur app shows expected delivery on 25th") contain almost none, and an
    earlier version of this filter discarded them at a rate ~4 points above the
    true non-English share.
    """
    english = english_score(text)
    foreign = foreign_score(text)

    if foreign > english and foreign >= 0.12:
        return False
    # Diacritics are strong evidence on their own, but only when the message
    # also fails to look English -- English tweets do quote "café" and "naïve".
    if _NON_EN_CHARS.search(text) and english < 0.08 and foreign > 0:
        return False
    return True


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
