"""
Smoke test — Phase 1 data pipeline verification
================================================
Verifies the full Phase 1 stack:
  labels.py → text_processor.py → dataset.py → pipeline.py

Run from the project root:
    python tests/test_phase1_smoke.py

What it checks
--------------
1. Label contract is consistent (maps round-trip correctly).
2. TextProcessor produces expected outputs for known inputs.
3. HateXplain loads correctly from HuggingFace.
4. HeteroData graph has the right shapes and label range.
5. Class weights are computed without errors.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(condition: bool, message: str) -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {message}")
    return condition


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────
#  Test 1: Label contract
# ─────────────────────────────────────────────────────────────────────

def test_labels() -> bool:
    section("1. Label contract")
    from config.labels import (
        HateSpeechLabel,
        HATE_SPEECH_LABELS,
        HATE_SPEECH_LABEL_TO_IDX,
        HATEXPLAIN_TO_CANONICAL,
        UNCERTAIN_LABEL,
        NUM_HATE_SPEECH_CLASSES,
    )
    ok = True

    ok &= check(NUM_HATE_SPEECH_CLASSES == 3, "3 hate-speech classes defined")
    ok &= check(HateSpeechLabel.NORMAL == 0,      "NORMAL  = 0")
    ok &= check(HateSpeechLabel.OFFENSIVE == 1,   "OFFENSIVE = 1")
    ok &= check(HateSpeechLabel.HATE_SPEECH == 2, "HATE_SPEECH = 2")
    ok &= check(UNCERTAIN_LABEL == -1,            "UNCERTAIN_LABEL = -1")

    # Round-trip: index → name → index
    for idx, name in HATE_SPEECH_LABELS.items():
        ok &= check(
            HATE_SPEECH_LABEL_TO_IDX[name] == idx,
            f"Round-trip: {idx} → '{name}' → {idx}",
        )

    # HateXplain map covers all native labels
    for native in ("normal", "offensive", "hatespeech"):
        ok &= check(
            native in HATEXPLAIN_TO_CANONICAL,
            f"HateXplain native label '{native}' in canonical map",
        )

    ok &= check(HATEXPLAIN_TO_CANONICAL["hatespeech"] == 2,
                "HateXplain 'hatespeech' → canonical index 2")
    return ok


# ─────────────────────────────────────────────────────────────────────
#  Test 2: TextProcessor
# ─────────────────────────────────────────────────────────────────────

def test_text_processor() -> bool:
    section("2. TextProcessor")
    from preprocessing.text_processor import TextProcessor, extract_text_features

    proc = TextProcessor()
    ok = True

    cases = [
        ("https://t.co/abc hello",        "[URL] hello",    "URL replaced"),
        ("@user great post",              "[USER] great post", "Mention replaced"),
        ("normal text",                   "normal text",    "Plain text unchanged"),
        ("<b>bold</b> text",              "bold text",      "HTML stripped"),
        ("lots   of   spaces",            "lots of spaces", "Whitespace collapsed"),
        ("",                              "",               "Empty string handled"),
        (None,                            "",               "None handled"),
    ]

    for inp, expected, desc in cases:
        result = proc.clean(inp)
        ok &= check(result == expected, f"{desc}: got '{result}'")

    # Features should return a 10-element list
    feats = extract_text_features("Hello world, how are you?")
    ok &= check(len(feats) == 10, f"extract_text_features returns 10 features: {feats}")
    ok &= check(feats[0] == 5.0,  f"Word count correct (5): got {feats[0]}")

    # filter_batch drops short texts
    texts = ["hi", "this is long enough", "ok"]
    kept, kept_idx = proc.filter_batch(texts)
    ok &= check("hi" not in kept, f"Short text 'hi' filtered out")
    ok &= check("this is long enough" in kept, "Long text retained")

    return ok


# ─────────────────────────────────────────────────────────────────────
#  Test 3: HateXplain dataset loader
# ─────────────────────────────────────────────────────────────────────

def test_hatexplain_loader() -> bool:
    section("3. HateXplain dataset loader")
    from tasks.hate_speech.dataset import (
        HateXplainLoader,
        load_hatexplain_for_training,
        COL_TEXT, COL_LABEL, COL_LABEL_NAME, COL_UNCERTAIN,
        COL_TARGETS, COL_SOURCE,
    )
    from config.labels import UNCERTAIN_LABEL, NUM_HATE_SPEECH_CLASSES
    ok = True

    try:
        loader = HateXplainLoader()
        # Load the small test split first (fastest)
        df = loader.load("test")
        logger.info("Loaded HateXplain test split: %d rows", len(df))

        # Required columns present
        for col in (COL_TEXT, COL_LABEL, COL_LABEL_NAME,
                    COL_UNCERTAIN, COL_TARGETS, COL_SOURCE):
            ok &= check(col in df.columns, f"Column '{col}' present")

        # Labels are in valid range
        valid_labels = set(range(NUM_HATE_SPEECH_CLASSES)) | {UNCERTAIN_LABEL}
        bad = df[~df[COL_LABEL].isin(valid_labels)]
        ok &= check(len(bad) == 0, f"All labels in valid range (bad={len(bad)})")

        # Text is non-empty for all rows
        empty_text = df[df[COL_TEXT].str.strip() == ""]
        ok &= check(len(empty_text) == 0,
                    f"No empty text rows (empty={len(empty_text)})")

        # Uncertain samples exist and are labelled correctly
        uncertain = df[df[COL_UNCERTAIN]]
        ok &= check(
            (uncertain[COL_LABEL] == UNCERTAIN_LABEL).all(),
            f"Uncertain rows have label=-1 ({len(uncertain)} rows)",
        )

        # load_hatexplain_for_training excludes uncertain
        filtered = load_hatexplain_for_training("test")
        ok &= check(
            filtered[COL_UNCERTAIN].sum() == 0,
            "load_hatexplain_for_training excludes uncertain rows",
        )

        loader.print_stats(df, "test")

    except Exception as e:
        ok = False
        print(f"  {FAIL}  HateXplain loader raised exception: {e}")
        import traceback; traceback.print_exc()

    return ok


# ─────────────────────────────────────────────────────────────────────
#  Test 4: Graph pipeline
# ─────────────────────────────────────────────────────────────────────

def test_pipeline() -> bool:
    section("4. Graph pipeline (validation split only for speed)")
    import torch
    from tasks.hate_speech.pipeline import HateSpeechPipeline, compute_class_weights
    from config.labels import NUM_HATE_SPEECH_CLASSES
    ok = True

    try:
        pipeline = HateSpeechPipeline(
            n_users=50,          # small for speed
            force_rebuild=False,
        )
        # Only run validation split to keep test fast (train takes ~2 min)
        result = pipeline.run(splits=("validation",))
        data   = result["validation"]

        # Check graph structure
        ok &= check("content" in data.node_types, "Content node type exists")
        ok &= check("user"    in data.node_types, "User node type exists")

        et = ("user", "interacts_with", "content")
        ok &= check(et in data.edge_types, "Forward edge type exists")

        # Content node feature shape
        x_content = data["content"].x
        ok &= check(x_content.shape[1] == 384, f"Content embed dim=384: got {x_content.shape[1]}")

        # User node feature shape
        x_user = data["user"].x
        ok &= check(x_user.shape[1] == 10, f"User feature dim=10: got {x_user.shape[1]}")
        ok &= check(x_user.shape[0] == 50, f"50 user nodes: got {x_user.shape[0]}")

        # Labels
        y = data["content"].y
        ok &= check(y.min().item() >= 0,
                    f"Min label >= 0: got {y.min().item()}")
        ok &= check(y.max().item() < NUM_HATE_SPEECH_CLASSES,
                    f"Max label < {NUM_HATE_SPEECH_CLASSES}: got {y.max().item()}")

        # Edge index shape
        edge_idx = data[et].edge_index
        ok &= check(edge_idx.shape[0] == 2,
                    f"edge_index has shape [2, E]: got {list(edge_idx.shape)}")

        # Class weights
        weights = compute_class_weights(y.tolist())
        ok &= check(weights.shape[0] == NUM_HATE_SPEECH_CLASSES,
                    f"Class weights shape ({NUM_HATE_SPEECH_CLASSES},): got {weights.shape}")
        ok &= check((weights > 0).all(),
                    f"All class weights positive: {weights.tolist()}")

    except Exception as e:
        ok = False
        print(f"  {FAIL}  Pipeline raised exception: {e}")
        import traceback; traceback.print_exc()

    return ok


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  HGCM-Net 2.0 — Phase 1 Smoke Test")
    print("=" * 55)

    results = {
        "Labels":        test_labels(),
        "TextProcessor": test_text_processor(),
        "HateXplainLoader": test_hatexplain_loader(),
        "Pipeline":      test_pipeline(),
    }

    print("\n" + "=" * 55)
    print("  Results")
    print("=" * 55)
    all_pass = True
    for name, passed in results.items():
        icon = PASS if passed else FAIL
        print(f"  {icon}  {name}")
        all_pass = all_pass and bool(passed)

    if all_pass:
        print("\n  All tests passed.  Phase 1 is ready.\n")
        sys.exit(0)
    else:
        print("\n  Some tests FAILED.  See output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
