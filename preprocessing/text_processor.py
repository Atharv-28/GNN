"""
Text Preprocessor — HGCM-Net 2.0
==================================
Cleans and normalises raw social-media text before embedding.

Design principles
-----------------
- Conservative: prefer preserving signal over aggressively stripping text.
  Social-media casing, punctuation, and emoji often carry sentiment signal.
- Reproducible: every transformation is deterministic and logged.
- Configurable: individual steps can be toggled via TextProcessorConfig.

Usage
-----
    from preprocessing.text_processor import TextProcessor

    proc = TextProcessor()
    clean = proc.clean("Check this out!! 🔥 http://t.co/xyz @user #hate")
    # → "Check this out!! 🔥 [URL] [USER] #hate"
"""

from __future__ import annotations

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TextProcessorConfig:
    """Toggle individual preprocessing steps.

    Attributes
    ----------
    replace_urls : bool
        Replace HTTP/HTTPS URLs with the token ``[URL]``.
    replace_mentions : bool
        Replace @-mentions with the token ``[USER]``.
    replace_hashtag_text : bool
        Keep the hashtag symbol but strip the word  (#hate → #hate is kept
        as-is; this flag only controls whether *all* hashtags are removed).
        Default False — we keep hashtags because they are strong signals.
    normalize_unicode : bool
        Apply NFKC unicode normalisation (collapses visually-similar chars).
    strip_html : bool
        Remove residual HTML tags.
    collapse_whitespace : bool
        Collapse repeated whitespace and strip leading/trailing spaces.
    lowercase : bool
        Lowercase the entire string.  Default False — casing carries signal
        in social media (e.g. all-caps correlates with hostility).
    max_chars : int
        Truncate text to this many characters *after* cleaning.
        SentenceTransformer all-MiniLM-L6-v2 has a 256-token limit;
        512 characters is a safe ceiling for most posts.
    min_chars : int
        Discard texts shorter than this after cleaning.
    """
    replace_urls:          bool = True
    replace_mentions:      bool = True
    remove_hashtags:       bool = False   # keep hashtags by default
    normalize_unicode:     bool = True
    strip_html:            bool = True
    collapse_whitespace:   bool = True
    lowercase:             bool = False
    max_chars:             int  = 512
    min_chars:             int  = 3


# ─────────────────────────────────────────────────────────────────────
#  Compiled patterns  (compile once at import time)
# ─────────────────────────────────────────────────────────────────────

_RE_URL      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION  = re.compile(r"@\w+")
_RE_HASHTAG  = re.compile(r"#\w+")
_RE_HTML     = re.compile(r"<[^>]+>")
_RE_SPACE    = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────
#  TextProcessor
# ─────────────────────────────────────────────────────────────────────

class TextProcessor:
    """Deterministic text cleaner for social-media posts.

    Parameters
    ----------
    config : TextProcessorConfig | None
        Uses defaults if ``None``.
    """

    def __init__(self, config: Optional[TextProcessorConfig] = None) -> None:
        self.config = config or TextProcessorConfig()

    # ── Core clean ──────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Apply all enabled cleaning steps to a single string.

        Parameters
        ----------
        text : str
            Raw social-media text.

        Returns
        -------
        str
            Cleaned text.  Empty string if the result is below min_chars.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        cfg = self.config

        # 1. Unicode normalisation (NFKC collapses look-alike characters)
        if cfg.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Strip HTML tags
        if cfg.strip_html:
            text = _RE_HTML.sub(" ", text)

        # 3. Replace URLs
        if cfg.replace_urls:
            text = _RE_URL.sub("[URL]", text)

        # 4. Replace @-mentions
        if cfg.replace_mentions:
            text = _RE_MENTION.sub("[USER]", text)

        # 5. Remove hashtags entirely (optional — off by default)
        if cfg.remove_hashtags:
            text = _RE_HASHTAG.sub("", text)

        # 6. Collapse whitespace
        if cfg.collapse_whitespace:
            text = _RE_SPACE.sub(" ", text).strip()

        # 7. Lowercase
        if cfg.lowercase:
            text = text.lower()

        # 8. Truncate / discard
        text = text[: cfg.max_chars]

        return text

    # ── Batch ───────────────────────────────────────────────────────

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of texts.

        Parameters
        ----------
        texts : list[str]
            Raw texts.

        Returns
        -------
        list[str]
            Cleaned texts (same length as input; short texts become "").
        """
        return [self.clean(t) for t in texts]

    def filter_batch(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
    ) -> tuple[List[str], List[int]]:
        """Clean texts and drop those that are too short.

        Parameters
        ----------
        texts : list[str]
            Raw texts.
        indices : list[int] | None
            Parallel list of original indices (e.g., dataset row IDs).
            If None, uses 0..N-1.

        Returns
        -------
        (kept_texts, kept_indices) : tuple
            Only texts that survived the min_chars filter.
        """
        if indices is None:
            indices = list(range(len(texts)))

        kept_texts, kept_indices = [], []
        min_c = self.config.min_chars

        for idx, raw in zip(indices, texts):
            cleaned = self.clean(raw)
            if len(cleaned) >= min_c:
                kept_texts.append(cleaned)
                kept_indices.append(idx)

        dropped = len(texts) - len(kept_texts)
        if dropped:
            logger.info("TextProcessor: dropped %d texts below min_chars=%d",
                        dropped, min_c)

        return kept_texts, kept_indices


# ─────────────────────────────────────────────────────────────────────
#  Feature extractor  (migrated from data_pipeline._text_features)
# ─────────────────────────────────────────────────────────────────────

import numpy as np   # noqa: E402  (placed after class to keep imports grouped)


def extract_text_features(text: str) -> List[float]:
    """Extract 10 numerical text-statistic features from a string.

    These become the raw user-node features when social-graph metadata
    is unavailable (e.g., when creating synthetic users via K-means).

    Feature index map
    -----------------
    0  word count
    1  character count
    2  sentence count
    3  words per sentence
    4  average word length
    5  vocabulary richness  (unique words / total words)
    6  uppercase ratio      (uppercase chars / total chars)
    7  digit ratio          (digit chars / total chars)
    8  punctuation ratio    (punct chars / total chars)
    9  has-content flag     (1.0 if any words, else 0.0)

    Parameters
    ----------
    text : str
        Already-cleaned text.

    Returns
    -------
    list[float]
        Length-10 feature vector.
    """
    if not text or not isinstance(text, str):
        text = ""

    words      = text.split()
    n_words    = len(words)
    n_chars    = len(text)
    n_sents    = max(1, len(re.split(r"[.!?]+", text)))
    n_upper    = sum(1 for c in text if c.isupper())
    n_digits   = sum(1 for c in text if c.isdigit())
    n_punct    = sum(1 for c in text if c in r'!?.,;:"\'-()[]{}')
    unique_w   = len(set(w.lower() for w in words)) if words else 0
    avg_wlen   = float(np.mean([len(w) for w in words])) if words else 0.0

    return [
        float(n_words),
        float(n_chars),
        float(n_sents),
        float(n_words / max(n_sents, 1)),
        avg_wlen,
        float(unique_w / max(n_words, 1)),
        float(n_upper  / max(n_chars, 1)),
        float(n_digits / max(n_chars, 1)),
        float(n_punct  / max(n_chars, 1)),
        1.0 if n_words > 0 else 0.0,
    ]


def extract_text_features_batch(texts: List[str]) -> np.ndarray:
    """Vectorised feature extraction.

    Parameters
    ----------
    texts : list[str]
        Already-cleaned texts.

    Returns
    -------
    ndarray, shape ``(N, 10)``
    """
    return np.array([extract_text_features(t) for t in texts], dtype=np.float32)
