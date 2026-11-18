"""
HateXplain Dataset Loader — HGCM-Net 2.0
==========================================
Loads the HateXplain dataset (Mathew et al., AAAI 2021) from HuggingFace,
applies majority-vote label aggregation, maps to our canonical 3-class
label contract, and returns a clean, analysis-ready DataFrame.

HateXplain specifics
--------------------
- Source: ``Hate-speech-CNERG/hatexplain`` on HuggingFace Hub
- Each sample has 3 annotators; we use majority vote.
- Native labels: ``normal``, ``offensive``, ``hatespeech``
- Splits: pre-defined train / validation / test  → we use them as-is.
- Extras retained: target communities, rationale tokens, platform (twitter/gab)

Usage
-----
    from tasks.hate_speech.dataset import HateXplainLoader

    loader = HateXplainLoader()
    train_df = loader.load("train")
    # Columns: post_id, text, label, label_name, targets, source,
    #          uncertain, rationale_tokens, annotator_labels

    # Quick stats
    loader.print_stats(train_df)
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Project imports ──────────────────────────────────────────────────
# Support running from both root and tasks/hate_speech/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.labels import (
    HATEXPLAIN_TO_CANONICAL,
    HATE_SPEECH_LABELS,
    UNCERTAIN_LABEL,
    HateSpeechLabel,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────

HATEXPLAIN_HF_ID   = "Hate-speech-CNERG/hatexplain"   # kept for reference
VALID_SPLITS       = ("train", "validation", "test")

# GitHub raw URLs for the HateXplain data files
# (HuggingFace repo uses a legacy script unsupported by datasets v4+)
_GITHUB_BASE       = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data"
_DATASET_JSON_URL  = f"{_GITHUB_BASE}/dataset.json"
_SPLITS_JSON_URL   = f"{_GITHUB_BASE}/post_id_divisions.json"

# GitHub split key ↔ our split name
# GitHub uses "val"; we expose "validation" for consistency with HF conventions
_SPLIT_KEY_MAP     = {"train": "train", "validation": "val", "test": "test"}

# Column names in the output DataFrame
COL_ID             = "post_id"
COL_TEXT           = "text"
COL_LABEL          = "label"          # canonical integer 0 / 1 / 2 / -1
COL_LABEL_NAME     = "label_name"     # human-readable string
COL_TARGETS        = "targets"        # list[str] — targeted communities
COL_SOURCE         = "source"         # "twitter" or "gab"
COL_UNCERTAIN      = "uncertain"      # True if no majority vote
COL_RATIONALE      = "rationale_tokens"  # list[str] highlighted tokens
COL_ANNOTATORS     = "annotator_labels"  # list of raw annotator labels


# ─────────────────────────────────────────────────────────────────────
#  Label aggregation helpers
# ─────────────────────────────────────────────────────────────────────

def _majority_vote(labels: List[str]) -> Tuple[str, bool]:
    """Return the majority label and whether the vote was ambiguous.

    Parameters
    ----------
    labels : list[str]
        Raw annotator labels (e.g. ["hatespeech", "offensive", "hatespeech"]).

    Returns
    -------
    (majority_label, is_uncertain) : tuple
        ``is_uncertain`` is True if no single label received > 50 % of votes.
        In a 3-annotator setup, ties happen when all three labels differ.
    """
    if not labels:
        return "normal", True

    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    is_uncertain = top_count <= len(labels) / 2   # tie or minority win

    return top_label, is_uncertain


def _extract_rationale_tokens(
    post_tokens: List[str],
    rationales: List[List[int]],
) -> List[str]:
    """Return tokens that at least one annotator highlighted as a rationale.

    Parameters
    ----------
    post_tokens : list[str]
        Tokenised post words.
    rationales : list[list[int]]
        One binary mask per annotator, 1 = highlighted.

    Returns
    -------
    list[str]
        Unique highlighted tokens (preserving order of first appearance).
    """
    if not post_tokens or not rationales:
        return []

    seen, result = set(), []
    for mask in rationales:
        for idx, flag in enumerate(mask):
            if flag and idx < len(post_tokens):
                token = post_tokens[idx]
                if token not in seen:
                    seen.add(token)
                    result.append(token)
    return result


# ─────────────────────────────────────────────────────────────────────
#  HateXplainLoader
# ─────────────────────────────────────────────────────────────────────

class HateXplainLoader:
    """Load, aggregate, and label-map HateXplain splits.

    Parameters
    ----------
    cache_dir : str | None
        HuggingFace cache directory.  Uses the HF default if ``None``.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir  = cache_dir
        # Local JSON cache so we don't re-download on every run
        self._data_cache_dir = Path("cached_data") / "hate_speech" / "raw"
        self._data_cache_dir.mkdir(parents=True, exist_ok=True)
        self._full_data: Optional[dict] = None   # lazy-loaded full dataset

    # ── Public API ──────────────────────────────────────────────────

    def load(self, split: str = "train") -> pd.DataFrame:
        """Load a HateXplain split and return a clean DataFrame.

        Parameters
        ----------
        split : str
            One of ``"train"``, ``"validation"``, ``"test"``.

        Returns
        -------
        pd.DataFrame
            Columns: post_id, text, label, label_name, targets, source,
            uncertain, rationale_tokens, annotator_labels.

        Raises
        ------
        ValueError
            If ``split`` is not one of the valid split names.
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'.  Choose from {VALID_SPLITS}."
            )

        logger.info("Loading HateXplain '%s' split from HuggingFace ...", split)
        raw = self._download(split)
        df  = self._process(raw)

        total      = len(df)
        uncertain  = df[COL_UNCERTAIN].sum()
        usable     = total - uncertain

        logger.info(
            "HateXplain '%s': %d total  |  %d uncertain (excluded from training)"
            "  |  %d usable",
            split, total, uncertain, usable,
        )
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all three splits.

        Returns
        -------
        dict[str, DataFrame]
            Keys: ``"train"``, ``"validation"``, ``"test"``
        """
        return {s: self.load(s) for s in VALID_SPLITS}

    def print_stats(self, df: pd.DataFrame, split_name: str = "") -> None:
        """Print label distribution statistics for a split.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :py:meth:`load`.
        split_name : str
            Optional label printed in the header.
        """
        header = f"HateXplain stats — {split_name}" if split_name else "HateXplain stats"
        print(f"\n{'='*55}")
        print(f"  {header}")
        print(f"{'='*55}")
        print(f"  Total samples  : {len(df):>6,}")

        uncertain = df[COL_UNCERTAIN].sum()
        print(f"  Uncertain      : {uncertain:>6,}  "
              f"({100*uncertain/len(df):.1f}%  — excluded from training)")

        usable = df[~df[COL_UNCERTAIN]]
        print(f"  Usable         : {len(usable):>6,}")
        print()

        # Per-class breakdown (usable only)
        counts = usable[COL_LABEL].value_counts().sort_index()
        for idx, name in HATE_SPEECH_LABELS.items():
            n = counts.get(idx, 0)
            pct = 100 * n / len(usable) if len(usable) > 0 else 0.0
            bar = "█" * int(pct / 2)
            print(f"  [{idx}] {name:<12} : {n:>5,}  ({pct:5.1f}%)  {bar}")

        # Sources
        if COL_SOURCE in df.columns:
            print()
            print("  Sources:")
            for src, cnt in df[COL_SOURCE].value_counts().items():
                print(f"    {src:<10} : {cnt:>5,}")

        print(f"{'='*55}\n")

    # ── Internal ────────────────────────────────────────────────────

    def _download(self, split: str):
        """Download HateXplain from GitHub and return a list of raw dicts.

        Uses the official GitHub release JSON files:
          dataset.json         — all 20,148 annotated posts
          post_id_divisions.json — train / val / test post-ID lists

        Files are cached locally under cached_data/hate_speech/raw/ so
        they are only fetched once per machine.
        """
        import urllib.request
        import json

        raw_cache  = self._data_cache_dir / "dataset.json"
        split_cache = self._data_cache_dir / "post_id_divisions.json"

        def _fetch_or_load(url: str, cache_path: Path) -> dict:
            if cache_path.exists():
                logger.info("Loading cached JSON: %s", cache_path)
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            logger.info("Downloading %s ...", url)
            req = urllib.request.Request(
                url, headers={"User-Agent": "HGCM-Net/2.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info("Cached → %s", cache_path)
            return data

        full_data  = _fetch_or_load(_DATASET_JSON_URL, raw_cache)
        split_ids  = _fetch_or_load(_SPLITS_JSON_URL, split_cache)

        github_key    = _SPLIT_KEY_MAP[split]       # "validation" → "val"
        ids_for_split = set(split_ids[github_key])

        rows = [
            entry for pid, entry in full_data.items()
            if pid in ids_for_split
        ]
        logger.info(
            "GitHub data loaded — split='%s' (%s): %d rows",
            split, github_key, len(rows),
        )
        return rows

    def _process(self, raw_dataset) -> pd.DataFrame:
        """Process raw dataset rows into a clean DataFrame.

        Handles both a list-of-dicts (GitHub JSON download) and an
        iterable HuggingFace Dataset object — same field names in both.
        """
        records = []

        for row in raw_dataset:
            post_id     = row.get("post_id", "")
            post_tokens = row.get("post_tokens", [])

            # Reconstruct text from tokens (HateXplain stores tokenised text)
            text = " ".join(post_tokens) if post_tokens else ""

            # Annotator labels and targets
            annotators  = row.get("annotators", [])
            # GitHub format: list of dicts [{label, annotator_id, target}, ...]
            if isinstance(annotators, list):
                raw_labels  = [a.get("label", "") for a in annotators]
                raw_targets = [a.get("target", []) for a in annotators]
            else:
                # HuggingFace format: dict of lists
                raw_labels  = annotators.get("label", [])
                raw_targets = annotators.get("target", [])

            # Majority vote
            maj_label, is_uncertain = _majority_vote(raw_labels)

            # Canonical label integer
            canonical = HATEXPLAIN_TO_CANONICAL.get(maj_label, UNCERTAIN_LABEL)
            if is_uncertain:
                canonical = UNCERTAIN_LABEL

            label_name = HATE_SPEECH_LABELS.get(canonical, "uncertain")

            # Flatten and deduplicate targets
            flat_targets: List[str] = []
            seen_targets: set       = set()
            for group in raw_targets:
                if isinstance(group, list):
                    for t in group:
                        if t and t not in seen_targets and t != "None":
                            seen_targets.add(t)
                            flat_targets.append(t)
                elif isinstance(group, str) and group not in seen_targets and group != "None":
                    seen_targets.add(group)
                    flat_targets.append(group)

            # Source platform — encoded in post_id suffix (_twitter / _gab)
            # or in a "domain" field if present
            source = row.get("domain", "")
            if not source:
                if post_id.endswith("_twitter"):
                    source = "twitter"
                elif post_id.endswith("_gab"):
                    source = "gab"
                else:
                    source = "unknown"

            # Rationale tokens
            rationale_data   = row.get("rationales", [])
            rationale_tokens = _extract_rationale_tokens(post_tokens, rationale_data)

            records.append({
                COL_ID:         post_id,
                COL_TEXT:       text,
                COL_LABEL:      canonical,
                COL_LABEL_NAME: label_name,
                COL_TARGETS:    flat_targets,
                COL_SOURCE:     source,
                COL_UNCERTAIN:  is_uncertain,
                COL_RATIONALE:  rationale_tokens,
                COL_ANNOTATORS: raw_labels,
            })

        df = pd.DataFrame(records)
        logger.info(
            "Processed %d records  |  uncertain=%d",
            len(df), df[COL_UNCERTAIN].sum(),
        )
        return df


# ─────────────────────────────────────────────────────────────────────
#  Convenience: load filtered training data
# ─────────────────────────────────────────────────────────────────────

def load_hatexplain_for_training(
    split: str = "train",
    cache_dir: Optional[str] = None,
    include_uncertain: bool = False,
) -> pd.DataFrame:
    """Load HateXplain split and filter out uncertain samples.

    Parameters
    ----------
    split : str
        Dataset split name.
    cache_dir : str | None
        HuggingFace cache directory.
    include_uncertain : bool
        If True, retain samples with no majority vote (label = -1).
        Default False — uncertain samples are excluded from training.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with columns from :py:class:`HateXplainLoader`.
    """
    loader = HateXplainLoader(cache_dir=cache_dir)
    df = loader.load(split)

    if not include_uncertain:
        before = len(df)
        df = df[~df[COL_UNCERTAIN]].reset_index(drop=True)
        logger.info(
            "Filtered uncertain: %d → %d samples", before, len(df)
        )

    return df
