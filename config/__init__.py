"""
config package — HGCM-Net 2.0

Backward-compatibility note
----------------------------
The original project had a root-level ``config.py`` that exported
``HGCMConfig``.  Now that ``config/`` is a package, Python resolves
``from config import HGCMConfig`` here.  We import ``HGCMConfig``
from the sibling module so existing files (graph_builder.py, train.py,
etc.) continue to work without modification.
"""

# ── Backward-compatible re-export from the original root config.py ───
# graph_builder.py, train.py and others do: from config import HGCMConfig
# We satisfy that by importing it here from the original module location.
import importlib, sys

# Load the root-level config.py directly by file path so the package
# name collision does not cause a circular import.
import importlib.util as _ilu
import pathlib as _pl

_root_config_path = _pl.Path(__file__).resolve().parent.parent / "config.py"
_spec = _ilu.spec_from_file_location("_root_config", _root_config_path)
_root_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)

HGCMConfig = _root_config.HGCMConfig   # re-export for backward compat

# ── New label contract ───────────────────────────────────────────────
from config.labels import (
    HateSpeechLabel,
    HATE_SPEECH_LABELS,
    HATE_SPEECH_LABEL_TO_IDX,
    NUM_HATE_SPEECH_CLASSES,
    HATEXPLAIN_TO_CANONICAL,
    OLID_TO_CANONICAL,
    FakeNewsLabel,
    FAKE_NEWS_LABELS,
    FAKE_NEWS_LABEL_TO_IDX,
    NUM_FAKE_NEWS_CLASSES,
    UNCERTAIN_LABEL,
)

__all__ = [
    # backward compat
    "HGCMConfig",
    # new labels
    "HateSpeechLabel",
    "HATE_SPEECH_LABELS",
    "HATE_SPEECH_LABEL_TO_IDX",
    "NUM_HATE_SPEECH_CLASSES",
    "HATEXPLAIN_TO_CANONICAL",
    "OLID_TO_CANONICAL",
    "FakeNewsLabel",
    "FAKE_NEWS_LABELS",
    "FAKE_NEWS_LABEL_TO_IDX",
    "NUM_FAKE_NEWS_CLASSES",
    "UNCERTAIN_LABEL",
]
