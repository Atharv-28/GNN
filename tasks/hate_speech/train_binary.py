"""Fine-tune a compact transformer for binary hate-speech detection.

The original 3-class HGCM-Net experiment is unchanged.  This script trains a
binary hate/non-hate classifier using HateXplain plus TweetEval-Hate and
evaluates only on the untouched HateXplain validation/test splits.

Example:
    python tasks/hate_speech/train_binary.py --model distilroberta-base
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.hate_speech.binary_dataset import load_binary_splits

CHECKPOINT_DIR = Path("checkpoints") / "hate_speech_binary"
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _make_loader(
    frame,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    dataset = TextDataset(frame["text"].tolist(), frame["label"].tolist())

    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        all_logits.append(model(**batch).logits.cpu())
        all_labels.append(labels.cpu())
    return _metrics(torch.cat(all_logits), torch.cat(all_labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary hate-speech transformer training")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tweeteval", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training rows for a quick smoke test.",
    )
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> dict[str, float]:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Binary fine-tuning requires transformers. Install requirements first."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    splits = load_binary_splits(include_tweeteval=not args.no_tweeteval)
    if args.max_train_samples is not None:
        if args.max_train_samples < 2:
            raise ValueError("--max-train-samples must be at least 2")
        splits["train"] = splits["train"].head(args.max_train_samples).copy()
        logger.info("Smoke-test limit: %d training rows", len(splits["train"]))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    train_loader = _make_loader(
        splits["train"], tokenizer, args.batch_size, True, args.max_length
    )
    val_loader = _make_loader(
        splits["validation"], tokenizer, args.batch_size, False, args.max_length
    )
    test_loader = _make_loader(
        splits["test"], tokenizer, args.batch_size, False, args.max_length
    )

    labels = torch.tensor(splits["train"]["label"].tolist())
    counts = torch.bincount(labels, minlength=2).float()
    weights = (counts.sum() / (2 * counts)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_f1 = -1.0
    stale_epochs = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_batch = batch.pop("labels")
            loss = criterion(model(**batch).logits, labels_batch)
            (loss / args.gradient_accumulation).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate(model, val_loader, device)
        logger.info("epoch=%d validation=%s", epoch, val_metrics)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            stale_epochs = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "task": "binary_hate_speech",
                        "base_model": args.model,
                        "max_length": args.max_length,
                        "label_names": ["non_hate", "hate"],
                        "validation_metrics": val_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    test_metrics = evaluate(model, test_loader, device)
    logger.info("HateXplain test=%s", test_metrics)
    print(json.dumps(test_metrics, indent=2))
    return test_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
