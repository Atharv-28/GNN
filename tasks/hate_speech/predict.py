"""
HGCM-Net 2.0 — Text-Only Inference Script
==========================================
Loads the best Phase 2 checkpoint and classifies one or more posts
from the command line.

Usage
-----
    # Classify a single post
    python tasks/hate_speech/predict.py --text "Your example post here"

    # Classify multiple posts from a file (one per line)
    python tasks/hate_speech/predict.py --file posts.txt

    # JSON output (for piping)
    python tasks/hate_speech/predict.py --text "Your post" --json

    # Use a specific checkpoint
    python tasks/hate_speech/predict.py --text "Your post" \\
        --checkpoint checkpoints/hate_speech/best_model_text_only.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── make project root importable ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.predictor import HateSpeechPredictor
from inference.schemas  import PredictionRequest


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HGCM-Net 2.0 — hate-speech inference (text-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text", type=str,
        help="A single post to classify.",
    )
    group.add_argument(
        "--file", type=Path,
        help="Path to a text file; each line is classified as a separate post.",
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/hate_speech/best_model_text_only.pt"),
        help="Path to the model checkpoint.",
    )
    p.add_argument(
        "--platform", type=str, default=None,
        choices=["x", "instagram", "reddit", "facebook", "youtube", "other"],
        help="Source platform (metadata only — not fed into the model).",
    )
    p.add_argument(
        "--review-threshold", type=float, default=0.70,
        help="Confidence below this triggers requires_review=True.",
    )
    p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print output as JSON (one object per line).",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU inference even when a GPU is available.",
    )
    return p


def _fmt_human(resp, text: str) -> str:
    """Pretty-print a PredictionResponse for terminal output."""
    LABEL_EMOJI = {
        "normal":      "✅ NORMAL",
        "offensive":   "⚠️  OFFENSIVE",
        "hate_speech": "🚨 HATE SPEECH",
    }
    flag = " [REVIEW]" if resp.requires_review else ""
    lines = [
        f"\n{'─' * 60}",
        f"  Text     : {text[:120]}{'…' if len(text) > 120 else ''}",
        f"  Result   : {LABEL_EMOJI[resp.prediction]}{flag}",
        f"  Confidence : {resp.confidence:.1%}",
        f"  Scores   :",
        f"    normal      {resp.scores['normal']:.3f}",
        f"    offensive   {resp.scores['offensive']:.3f}",
        f"    hate_speech {resp.scores['hate_speech']:.3f}",
        f"  Mode     : {resp.inference_mode}",
        f"  Model    : {resp.model_version}",
        f"{'─' * 60}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    import torch
    device_str = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")

    predictor = HateSpeechPredictor(
        checkpoint_path  = args.checkpoint,
        device           = device_str,
        review_threshold = args.review_threshold,
    )

    # ── Build list of texts ──────────────────────────────────────────
    if args.text:
        texts = [args.text]
    else:
        file_path: Path = args.file
        if not file_path.exists():
            print(f"ERROR: file not found — {file_path}", file=sys.stderr)
            sys.exit(1)
        texts = [ln.strip() for ln in file_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        if not texts:
            print("ERROR: file is empty or contains only blank lines.", file=sys.stderr)
            sys.exit(1)

    # ── Predict ──────────────────────────────────────────────────────
    if len(texts) == 1:
        req  = PredictionRequest(text=texts[0], platform=args.platform)
        resp = predictor.predict(req)
        if args.as_json:
            print(json.dumps(resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__,
                             ensure_ascii=False))
        else:
            print(_fmt_human(resp, texts[0]))
    else:
        from inference.schemas import BatchPredictionRequest
        reqs = [PredictionRequest(text=t, platform=args.platform) for t in texts]
        batch_resp = predictor.predict_batch(BatchPredictionRequest(posts=reqs))
        for t, resp in zip(texts, batch_resp.predictions):
            if args.as_json:
                d = resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__
                d["input_text"] = t
                print(json.dumps(d, ensure_ascii=False))
            else:
                print(_fmt_human(resp, t))
        if not args.as_json:
            flagged = batch_resp.total_flagged
            total   = len(texts)
            print(f"\n  Summary: {flagged}/{total} posts flagged for review.")


if __name__ == "__main__":
    main()
