"""Leakage-safe binary hate-speech dataset preparation.

The binary target is deliberately separate from the existing 3-class task:
``0`` means non-hate and ``1`` means hate.  HateXplain remains the source of
the held-out validation and test sets; TweetEval-Hate is used for training
augmentation only.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.text_processor import TextProcessor
from tasks.hate_speech.dataset import (
    COL_LABEL,
    COL_TEXT,
    load_hatexplain_for_training,
)

logger = logging.getLogger(__name__)


def _key(text: str) -> str:
    """Create a conservative key for cross-dataset duplicate removal."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _hatexplain_binary(split: str, processor: TextProcessor) -> pd.DataFrame:
    df = load_hatexplain_for_training(split=split)
    return pd.DataFrame(
        {
            "text": [processor.clean(text) for text in df[COL_TEXT]],
            "label": [int(label == 2) for label in df[COL_LABEL]],
            "source": "hatexplain",
        }
    )


def _tweeteval_binary(processor: TextProcessor) -> pd.DataFrame:
    """Load TweetEval-Hate, raising an actionable dependency error."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "TweetEval augmentation requires the 'datasets' package. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    dataset = load_dataset("cardiffnlp/tweet_eval", "hate")
    records: List[dict] = []
    for row in dataset["train"]:
        text = processor.clean(row["text"])
        if text:
            records.append(
                {"text": text, "label": int(row["label"]), "source": "tweeteval"}
            )
    return pd.DataFrame(records)


def load_binary_splits(
    include_tweeteval: bool = True,
    processor: TextProcessor | None = None,
) -> Dict[str, pd.DataFrame]:
    """Return train, validation, and test data with binary labels.

    TweetEval is never added to validation or test.  Any TweetEval text that
    duplicates an original HateXplain evaluation item is removed from train.
    """
    processor = processor or TextProcessor()
    splits = {
        "validation": _hatexplain_binary("validation", processor),
        "test": _hatexplain_binary("test", processor),
    }
    train = _hatexplain_binary("train", processor)

    if include_tweeteval:
        extra = _tweeteval_binary(processor)
        held_out_keys = {
            _key(text)
            for frame in splits.values()
            for text in frame["text"].tolist()
        }
        extra = extra[~extra["text"].map(_key).isin(held_out_keys)]
        train = pd.concat([train, extra], ignore_index=True)
        logger.info("Added %d TweetEval-Hate training examples.", len(extra))

    train = train.drop_duplicates(subset=["text"]).reset_index(drop=True)
    for name, frame in splits.items():
        # Preserve every original evaluation row so the official split sizes
        # and test protocol remain unchanged.
        splits[name] = frame.reset_index(drop=True)
        logger.info(
            "%s: %d rows (hate=%d, non_hate=%d)",
            name,
            len(splits[name]),
            int(splits[name]["label"].sum()),
            int((splits[name]["label"] == 0).sum()),
        )
    logger.info(
        "train: %d rows (hate=%d, non_hate=%d)",
        len(train),
        int(train["label"].sum()),
        int((train["label"] == 0).sum()),
    )
    splits["train"] = train
    return splits
