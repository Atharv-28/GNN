"""
Label Contract — HGCM-Net 2.0
==============================
Canonical label definitions for every task this model supports.
This file is frozen once a task enters production.  Changing an
index after training invalidates all saved checkpoints.

Usage
-----
    from config.labels import HATE_SPEECH_LABELS, HateSpeechLabel

    # Integer → string
    HATE_SPEECH_LABELS[2]            # → "hate_speech"

    # String → integer (use the reverse map)
    HATE_SPEECH_LABEL_TO_IDX["hate_speech"]   # → 2

    # Enum for type-safe code
    HateSpeechLabel.HATE_SPEECH.value          # → 2
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

# ─────────────────────────────────────────────────────────────────────
#  Task: Hate Speech Detection  (3-class)
# ─────────────────────────────────────────────────────────────────────

class HateSpeechLabel(IntEnum):
    """Integer enum for the hate-speech 3-class task.

    Index order is intentional and must not change:
      0 → normal      (benign, no offensive content)
      1 → offensive   (personal insults, profanity, non-group-directed)
      2 → hate_speech (group-directed hostility / dehumanisation)

    Class 2 is the high-stakes class: recall for this class is the
    primary safety metric throughout development.
    """
    NORMAL      = 0
    OFFENSIVE   = 1
    HATE_SPEECH = 2


# Human-readable names for each index (use in reports, UI, API responses)
HATE_SPEECH_LABELS: Dict[int, str] = {
    HateSpeechLabel.NORMAL:      "normal",
    HateSpeechLabel.OFFENSIVE:   "offensive",
    HateSpeechLabel.HATE_SPEECH: "hate_speech",
}

# Reverse map: string → index (for dataset loaders)
HATE_SPEECH_LABEL_TO_IDX: Dict[str, int] = {
    v: k for k, v in HATE_SPEECH_LABELS.items()
}

NUM_HATE_SPEECH_CLASSES: int = len(HATE_SPEECH_LABELS)   # 3


# ─────────────────────────────────────────────────────────────────────
#  Dataset-specific label maps
#  These translate raw dataset strings into our canonical indices.
#  NEVER use dataset-native strings outside the loaders.
# ─────────────────────────────────────────────────────────────────────

# HateXplain (Mathew et al., AAAI 2021)
# Source labels: "normal", "offensive", "hatespeech" (note: no underscore)
HATEXPLAIN_TO_CANONICAL: Dict[str, int] = {
    "normal":     HateSpeechLabel.NORMAL,
    "offensive":  HateSpeechLabel.OFFENSIVE,
    "hatespeech": HateSpeechLabel.HATE_SPEECH,   # maps to our "hate_speech"
}

# OLID — Offensive Language Identification Dataset (SemEval 2019 Task 6)
# OLID uses a hierarchical subtask scheme, not a flat 3-class label.
# This mapping is approximate and must be evaluated separately before
# mixing OLID samples into the main training corpus.
#   Subtask A: NOT / OFF
#   Subtask B (if OFF): UNT / TIN
#   Subtask C (if TIN): IND / GRP / OTH
#
# Mapping rationale:
#   NOT          → normal     (not offensive at all)
#   OFF + UNT    → offensive  (untargeted offensive language)
#   OFF + TIN + IND → offensive  (targeted at individual, not a group)
#   OFF + TIN + GRP → hate_speech (targeted at a group / identity)
#   OFF + TIN + OTH → offensive  (conservative: other targets, not group)
OLID_TO_CANONICAL: Dict[str, int] = {
    "NOT":     HateSpeechLabel.NORMAL,
    "OFF_UNT": HateSpeechLabel.OFFENSIVE,
    "OFF_IND": HateSpeechLabel.OFFENSIVE,
    "OFF_GRP": HateSpeechLabel.HATE_SPEECH,
    "OFF_OTH": HateSpeechLabel.OFFENSIVE,
}


# ─────────────────────────────────────────────────────────────────────
#  Task: Fake News Detection  (2-class, existing task — do not change)
# ─────────────────────────────────────────────────────────────────────

class FakeNewsLabel(IntEnum):
    REAL = 0
    FAKE = 1


FAKE_NEWS_LABELS: Dict[int, str] = {
    FakeNewsLabel.REAL: "real",
    FakeNewsLabel.FAKE: "fake",
}

FAKE_NEWS_LABEL_TO_IDX: Dict[str, int] = {
    v: k for k, v in FAKE_NEWS_LABELS.items()
}

NUM_FAKE_NEWS_CLASSES: int = len(FAKE_NEWS_LABELS)   # 2


# ─────────────────────────────────────────────────────────────────────
#  Sentinel for uncertain / unannotatable samples
# ─────────────────────────────────────────────────────────────────────

UNCERTAIN_LABEL: int = -1   # Exclude from training; retain for analysis


# ─────────────────────────────────────────────────────────────────────
#  Convenience aliases used by inference / predictor code
# ─────────────────────────────────────────────────────────────────────

# idx → label string  (alias for HATE_SPEECH_LABELS — avoids long names in hot paths)
IDX_TO_LABEL: Dict[int, str] = HATE_SPEECH_LABELS

# Individual string constants (avoid bare string literals in inference code)
LABEL_NORMAL:      str = HATE_SPEECH_LABELS[HateSpeechLabel.NORMAL]       # "normal"
LABEL_OFFENSIVE:   str = HATE_SPEECH_LABELS[HateSpeechLabel.OFFENSIVE]    # "offensive"
LABEL_HATE_SPEECH: str = HATE_SPEECH_LABELS[HateSpeechLabel.HATE_SPEECH]  # "hate_speech"
