"""Command-line inference for the fine-tuned binary hate classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary hate-speech prediction")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/hate_speech_binary"),
    )
    args = parser.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires transformers. Install requirements first."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device).eval()

    with torch.no_grad():
        batch = tokenizer(
            args.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities = torch.softmax(model(**batch).logits, dim=-1)[0]

    result = {
        "prediction": "hate" if int(probabilities.argmax()) == 1 else "non_hate",
        "confidence": float(probabilities.max()),
        "scores": {
            "non_hate": float(probabilities[0]),
            "hate": float(probabilities[1]),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
