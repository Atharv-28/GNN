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
import importlib.util as _ilu
import sys as _sys
import pathlib as _pl

_root_config_path = _pl.Path(__file__).resolve().parent.parent / "config.py"

# Register under a stable module name so pickle can round-trip HGCMConfig
if "_hgcm_root_config" not in _sys.modules:
    _spec = _ilu.spec_from_file_location("_hgcm_root_config", _root_config_path)
    _root_config = _ilu.module_from_spec(_spec)
    _sys.modules["_hgcm_root_config"] = _root_config
    _spec.loader.exec_module(_root_config)
else:
    _root_config = _sys.modules["_hgcm_root_config"]

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
