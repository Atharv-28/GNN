"""
Hate-Speech Training — HGCM-Net 2.0  (Phase 2: text-only baseline)
=====================================================================
Trains HGCM-Net 2.0 in text-only mode on HateXplain.

What's different from the original train.py
--------------------------------------------
- 3-class CrossEntropyLoss with class weights
- Macro F1 is the primary checkpoint metric (not binary F1)
- Hate-class recall is logged at every epoch as the safety metric
- Early stopping on macro F1 (patience configurable)
- Contrastive loss is OFF in text-only mode (no meaningful user/content
  pair to contrast; reintroduced in Phase 3 with the full graph)
- Saves per-epoch CSV log in addition to the standard training.log

Run
---
    cd d:\\Coding\\Machine Learning\\GNN Mini-pro
    venv\\Scripts\\python.exe tasks/hate_speech/train.py

    # Force pipeline rebuild:
    venv\\Scripts\\python.exe tasks/hate_speech/train.py --force-rebuild

    # Quick sanity check (1 epoch, tiny model):
    venv\\Scripts\\python.exe tasks/hate_speech/train.py --epochs 1 --debug
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import HGCMConfig
from config.labels import NUM_HATE_SPEECH_CLASSES, HateSpeechLabel
from models.hgcm_net2 import HGCMNet2
from models.classification_head import build_loss
from tasks.hate_speech.pipeline import HateSpeechPipeline, compute_class_weights
from tasks.hate_speech.evaluate import HateSpeechEvaluator, compute_metrics

# ─────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────

CKPT_DIR  = Path("checkpoints") / "hate_speech"
LOG_DIR   = Path("logs") / "hate_speech"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Split masks
# ─────────────────────────────────────────────────────────────────────

def _graph_to_masks(data) -> dict[str, torch.Tensor]:
    """Identify which graph this is and return a mask dictionary.

    Since the pipeline now creates per-split graphs (not a single
    merged graph), every node in ``data["content"]`` is in that split.
    We return an all-True mask for the content nodes.
    """
    n = data["content"].num_nodes
    return torch.ones(n, dtype=torch.bool)


# ─────────────────────────────────────────────────────────────────────
#  Training step
# ─────────────────────────────────────────────────────────────────────

def train_epoch(
    model,
    data,
    criterion,
    optimizer,
    device:    torch.device,
    text_only: bool = True,
) -> dict:
    """Run one full-batch training epoch.

    Returns
    -------
    dict
        loss, accuracy, macro_f1, hate_recall.
    """
    model.train()
    optimizer.zero_grad()

    x_dict    = {k: data[k].x.to(device)           for k in data.node_types}
    edge_dict = {k: data[k].edge_index.to(device)  for k in data.edge_types}
    labels    = data["content"].y.to(device)

    logits, _, _ = model(x_dict, edge_dict, text_only=text_only)

    loss = criterion(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    m = compute_metrics(logits.detach(), labels.detach())
    m["loss"] = loss.item()
    # Remove internal tensors not needed in the log
    m.pop("_probs", None)
    m.pop("_labels", None)
    m.pop("confusion_matrix", None)
    return m


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HGCM-Net 2.0 hate-speech training")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=25,
                   help="Early-stop patience (epochs without macro-F1 improvement)")
    p.add_argument("--warmup",        type=int,   default=15,
                   help="Cosine LR warmup epochs")
    p.add_argument("--n-users",       type=int,   default=400)
    p.add_argument("--force-rebuild", action="store_true",
                   help="Ignore cached pipeline data and rebuild everything")
    p.add_argument("--debug",         action="store_true",
                   help="Run 1 epoch with tiny model for smoke-testing")
    return p.parse_args()


def main():
    args   = parse_args()
    config = HGCMConfig()

    if args.debug:
        args.epochs  = 1
        args.patience = 999

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── 1. Data ──────────────────────────────────────────────────────
    logger.info("Building / loading HateXplain pipeline ...")
    pipeline = HateSpeechPipeline(
        n_users       = args.n_users,
        force_rebuild = args.force_rebuild,
    )
    splits = pipeline.run(splits=("train", "validation", "test"))
    train_data = splits["train"]
    val_data   = splits["validation"]
    test_data  = splits["test"]

    train_labels = train_data["content"].y.tolist()
    class_weights = compute_class_weights(
        train_labels, device=str(device)
    )
    logger.info("Class weights: %s",
                {f"c{i}": f"{w:.4f}" for i, w in enumerate(class_weights.tolist())})

    # ── 2. Model ─────────────────────────────────────────────────────
    model = HGCMNet2(config, num_classes=NUM_HATE_SPEECH_CLASSES).to(device)

    # Lazy initialisation — must run BOTH modes so that HeteroGAT's
    # lazy parameters are initialised even if training uses text_only=True.
    with torch.no_grad():
        x_d = {k: train_data[k].x.to(device) for k in train_data.node_types}
        e_d = {k: train_data[k].edge_index.to(device) for k in train_data.edge_types}
        model(x_d, e_d, text_only=False)   # initialises GAT lazy params
        model(x_d, e_d, text_only=True)    # warms up text-only path

    n_params = sum(p.numel() for p in model.parameters() if p.numel() > 0)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── 3. Loss / optimiser / scheduler ─────────────────────────────
    criterion = build_loss(
        num_classes   = NUM_HATE_SPEECH_CLASSES,
        class_weights = class_weights,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.warmup, T_mult=2
    )

    evaluator = HateSpeechEvaluator(device=str(device))

    # ── 4. Training loop ─────────────────────────────────────────────
    best_macro_f1  = 0.0
    patience_ctr   = 0
    csv_path       = LOG_DIR / "epoch_log.csv"
    csv_fields     = [
        "epoch", "lr",
        "train_loss", "train_macro_f1", "train_hate_recall",
        "val_loss",   "val_macro_f1",   "val_hate_recall",
        "elapsed_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()

        logger.info("=" * 65)
        logger.info("Training started — text_only=True  epochs=%d  patience=%d",
                    args.epochs, args.patience)
        logger.info("=" * 65)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tm = train_epoch(
                model, train_data, criterion, optimizer, device,
                text_only=True,
            )

            # Validation
            val_metrics = evaluator.evaluate(
                model, val_data,
                mask      = _graph_to_masks(val_data).to(device),
                loss_fn   = criterion,
                text_only = True,
            )

            scheduler.step()
            elapsed = time.time() - t0
            cur_lr  = optimizer.param_groups[0]["lr"]

            # ── Log ──────────────────────────────────────────────────
            logger.info(
                "Ep %3d | lr=%.2e | "
                "tr loss=%.4f macro_f1=%.4f hate_rec=%.4f | "
                "val loss=%.4f macro_f1=%.4f hate_rec=%.4f | %.1fs",
                epoch, cur_lr,
                tm["loss"],         tm["macro_f1"],    tm["hate_recall"],
                val_metrics["loss"], val_metrics["macro_f1"], val_metrics["hate_recall"],
                elapsed,
            )

            writer.writerow({
                "epoch":           epoch,
                "lr":              f"{cur_lr:.2e}",
                "train_loss":      f"{tm['loss']:.4f}",
                "train_macro_f1":  f"{tm['macro_f1']:.4f}",
                "train_hate_recall": f"{tm['hate_recall']:.4f}",
                "val_loss":        f"{val_metrics['loss']:.4f}",
                "val_macro_f1":    f"{val_metrics['macro_f1']:.4f}",
                "val_hate_recall": f"{val_metrics['hate_recall']:.4f}",
                "elapsed_s":       f"{elapsed:.1f}",
            })
            fh.flush()

            # ── Checkpoint ───────────────────────────────────────────
            val_f1 = val_metrics["macro_f1"]
            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                patience_ctr  = 0
                ckpt_path = CKPT_DIR / "best_model_text_only.pt"
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "val_macro_f1":    best_macro_f1,
                    "val_hate_recall": val_metrics["hate_recall"],
                    "config_dict":     vars(config),   # plain dict, always picklable
                    "num_classes":     NUM_HATE_SPEECH_CLASSES,
                    "text_only":       True,
                    "model_version":   "hgcm-net-2.0-hatexplain-text-only",
                }, ckpt_path)
                logger.info("  >> Checkpoint saved  val_macro_f1=%.4f  hate_rec=%.4f",
                            best_macro_f1, val_metrics["hate_recall"])
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    logger.info("Early stop at epoch %d  (patience=%d)",
                                epoch, args.patience)
                    break

    # ── 5. Test evaluation ───────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Loading best checkpoint for test evaluation ...")
    ckpt_path = CKPT_DIR / "best_model_text_only.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded checkpoint from epoch %d  val_macro_f1=%.4f",
                    ckpt["epoch"], ckpt["val_macro_f1"])

    test_metrics = evaluator.evaluate(
        model, test_data,
        mask      = _graph_to_masks(test_data).to(device),
        loss_fn   = criterion,
        text_only = True,
    )

    evaluator.print_report(test_metrics, split_name="Test (text-only baseline)")

    # Save confusion matrix
    evaluator.save_confusion_matrix(
        test_metrics,
        save_path=str(LOG_DIR / "confusion_matrix_test.png"),
        title="Test Confusion Matrix — Text-Only Baseline",
    )

    logger.info("=" * 65)
    logger.info("TEST RESULTS")
    logger.info("  Accuracy    : %.4f", test_metrics["accuracy"])
    logger.info("  Macro F1    : %.4f", test_metrics["macro_f1"])
    logger.info("  Hate Recall : %.4f", test_metrics["hate_recall"])
    logger.info("  PR-AUC hate : %.4f", test_metrics.get("pr_auc_hate", float("nan")))
    logger.info("=" * 65)

    return test_metrics


if __name__ == "__main__":
    main()
