"""Frozen shared text normalizer for gold and ASR transcripts (plan section 10.2).

The SAME normalizer is applied to gold transcripts and to every ASR system's
output. This is what makes the C2 -> C3/C4 substitution controlled: the only
difference between conditions is transcription content, never orthographic or
punctuation style.

Design decisions, all deliberate and reported in the paper:

1. Unicode NFC. Note that Bengali nukta forms (U+09DC, U+09DD, U+09DF) are
   Unicode composition exclusions, so NFC *decomposes* them to base + nukta.
   That is desirable here: it canonicalises the two ways of encoding য়/ড়/ঢ়
   that gold text and ASR output may disagree on.
2. Model-specific special tokens are stripped (Whisper <|...|> tags, bracketed
   event tags such as [MUSIC], NeMo <unk>-style tokens).
3. Punctuation is REMOVED, not merely standardised. Whisper emits punctuation;
   CTC-based Indic models generally do not. Keeping punctuation would penalise
   the Indic system for a stylistic difference rather than a recognition error,
   and would confound the downstream substitution comparison. Removal is applied
   to gold text too, so all conditions see the same punctuation-free surface.
4. Bengali digits are mapped to ASCII digits so the two scripts' numerals are
   not counted as substitutions. A digit-vs-spelled-out-word difference remains
   a genuine ASR error and is left alone.
5. ZWJ/ZWNJ are removed. They alter conjunct rendering but not word identity,
   and ASR systems emit them inconsistently.
6. No spelling correction, no translation, no repair of sentiment-bearing words,
   no ASR-only repetition removal.

Freeze this file before computing final metrics. If it changes, bump
NORMALIZATION_VERSION and re-run every downstream stage.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "v1"

# Whisper-style tags such as <|bn|>, <|transcribe|>, <|0.00|>, and NeMo <unk>.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>|<[a-z_]{1,16}>", re.IGNORECASE)

# Bracketed non-speech event annotations, e.g. [MUSIC], (laughs), 【音】.
_EVENT_TAG_RE = re.compile(r"[\[\(\{][^\]\)\}]{0,40}[\]\)\}]")

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))

_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Punctuation removed from both gold and ASR text. Covers the Bengali danda and
# double danda, the ASCII set, and the common Unicode quotation/dash forms.
_PUNCTUATION_CHARS = (
    "।॥"  # danda, double danda
    "৽"  # Bengali abbreviation sign
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "‐‑‒–—―"  # hyphens and dashes
    "‘’‚‛“”„‟"  # curly quotes
    "…′″«»‹›·•"
)
_PUNCTUATION = dict.fromkeys(map(ord, _PUNCTUATION_CHARS), " ")

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_bangla(text: str) -> str:
    """Return the frozen normalized form of a Bengali transcript.

    Non-string or missing input normalizes to the empty string so that a failed
    ASR call is recorded as an empty hypothesis rather than crashing the run.
    """
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = _SPECIAL_TOKEN_RE.sub(" ", normalized)
    normalized = _EVENT_TAG_RE.sub(" ", normalized)
    normalized = normalized.translate(_ZERO_WIDTH)
    normalized = normalized.translate(_BENGALI_DIGITS)
    normalized = normalized.translate(_PUNCTUATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def word_tokens(normalized_text: str) -> list[str]:
    """Whitespace tokenization used for WER (plan section 10.3)."""
    return normalized_text.split()


def grapheme_clusters(normalized_text: str) -> list[str]:
    """Grapheme-cluster tokenization used for CER (plan section 10.3).

    Bengali is an abugida: a single visual unit routinely spans several code
    points (consonant + virama + consonant + vowel sign). Counting code points
    would overstate character accuracy, so CER is computed over extended
    grapheme clusters. Spaces are excluded consistently on both sides.

    Falls back to code points if the `regex` module is unavailable; callers
    should record which path was taken via `cer_is_grapheme_aware()`.
    """
    text = normalized_text.replace(" ", "")
    if _REGEX_GRAPHEME is None:
        return list(text)
    return _REGEX_GRAPHEME.findall(text)


try:  # pragma: no cover - import-time capability probe
    import regex as _regex

    _REGEX_GRAPHEME = _regex.compile(r"\X")
except ImportError:  # pragma: no cover
    _REGEX_GRAPHEME = None


def cer_is_grapheme_aware() -> bool:
    """True when CER uses extended grapheme clusters rather than code points."""
    return _REGEX_GRAPHEME is not None
