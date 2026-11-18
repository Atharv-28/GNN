"""
Hate-Speech Evaluation — HGCM-Net 2.0
========================================
Full metric suite for the 3-class hate-speech classifier.

Metrics tracked (in priority order)
------------------------------------
1. Macro F1              — primary optimisation target
2. Hate-class recall     — primary safety metric (class 2)
3. Per-class P / R / F1
4. Confusion matrix
5. PR-AUC (hate class)
6. Accuracy              — reported but not optimised

Usage
-----
    from tasks.hate_speech.evaluate import HateSpeechEvaluator

    evaluator = HateSpeechEvaluator(device="cuda")
    metrics   = evaluator.evaluate(model, data, mask)

    evaluator.print_report(metrics)
    evaluator.save_confusion_matrix(metrics, "results/cm.png")
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.labels import HATE_SPEECH_LABELS, NUM_HATE_SPEECH_CLASSES, HateSpeechLabel

logger = logging.getLogger(__name__)

CLASS_NAMES = [HATE_SPEECH_LABELS[i] for i in range(NUM_HATE_SPEECH_CLASSES)]


# ─────────────────────────────────────────────────────────────────────
#  Core metric computation (no sklearn dependency required)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_HATE_SPEECH_CLASSES,
) -> Dict[str, float]:
    """Compute classification metrics from raw logits and integer labels.

    Parameters
    ----------
    logits : Tensor, shape ``(N, num_classes)``
        Raw logits from the classifier.
    labels : Tensor, shape ``(N,)``
        Ground-truth integer class indices.
    num_classes : int
        Number of classes.

    Returns
    -------
    dict
        Keys: accuracy, macro_f1, hate_recall, per_class (dict),
        confusion_matrix (list[list[int]]).
    """
    probs  = F.softmax(logits.float(), dim=-1)
    preds  = probs.argmax(dim=-1)

    # ── Accuracy ─────────────────────────────────────────────────────
    accuracy = (preds == labels).float().mean().item()

    # ── Per-class P / R / F1 ─────────────────────────────────────────
    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores = []

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().float().item()
        fp = ((preds == c) & (labels != c)).sum().float().item()
        fn = ((preds != c) & (labels == c)).sum().float().item()

        prec   = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * recall / (prec + recall + 1e-8)

        name = HATE_SPEECH_LABELS.get(c, str(c))
        per_class[name] = {
            "precision": round(prec,   4),
            "recall":    round(recall, 4),
            "f1":        round(f1,     4),
            "support":   int(tp + fn),
        }
        f1_scores.append(f1)

    macro_f1    = float(np.mean(f1_scores))
    hate_recall = per_class[HATE_SPEECH_LABELS[HateSpeechLabel.HATE_SPEECH]]["recall"]

    # ── Confusion matrix ─────────────────────────────────────────────
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.cpu(), preds.cpu()):
        cm[t.item(), p.item()] += 1
    confusion_matrix = cm.tolist()

    return {
        "accuracy":        round(accuracy,    4),
        "macro_f1":        round(macro_f1,    4),
        "hate_recall":     round(hate_recall, 4),
        "per_class":       per_class,
        "confusion_matrix": confusion_matrix,
        # Raw tensors for PR-AUC computation
        "_probs":  probs.cpu(),
        "_labels": labels.cpu(),
    }


def compute_pr_auc(
    probs:  torch.Tensor,
    labels: torch.Tensor,
    target_class: int = HateSpeechLabel.HATE_SPEECH,
) -> float:
    """Compute PR-AUC for a single class (default: hate_speech).

    Parameters
    ----------
    probs : Tensor, shape ``(N, C)``
        Softmax probabilities.
    labels : Tensor, shape ``(N,)``
        Ground-truth labels.
    target_class : int
        Class index for one-vs-rest PR-AUC.

    Returns
    -------
    float
        Area under the Precision-Recall curve.
    """
    try:
        from sklearn.metrics import average_precision_score
        scores  = probs[:, target_class].numpy()
        binary  = (labels == target_class).numpy().astype(int)
        return float(average_precision_score(binary, scores))
    except ImportError:
        logger.warning("sklearn not available; PR-AUC skipped.")
        return float("nan")
    except Exception as e:
        logger.warning("PR-AUC computation failed: %s", e)
        return float("nan")


# ─────────────────────────────────────────────────────────────────────
#  Evaluator class
# ─────────────────────────────────────────────────────────────────────

class HateSpeechEvaluator:
    """Evaluate a hate-speech model on a HeteroData graph split.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data,
        mask:        torch.Tensor,
        loss_fn:     Optional[torch.nn.Module] = None,
        text_only:   bool = True,
    ) -> Dict:
        """Run evaluation on a masked subset of a HeteroData graph.

        Parameters
        ----------
        model : HGCMNet2
            The model to evaluate (must be in eval mode).
        data : HeteroData
            Full graph (or split graph).
        mask : Tensor, shape ``(N_content,)``
            Boolean mask selecting evaluation nodes.
        loss_fn : nn.Module | None
            If provided, computes masked loss for logging.
        text_only : bool
            Use text-only inference (bypass GAT).

        Returns
        -------
        dict
            All metrics from :func:`compute_metrics` plus pr_auc and loss.
        """
        model.eval()

        x_dict    = {k: data[k].x.to(self.device)           for k in data.node_types}
        edge_dict = {k: data[k].edge_index.to(self.device)  for k in data.edge_types}
        labels    = data["content"].y.to(self.device)

        logits, _, _ = model(x_dict, edge_dict, text_only=text_only)

        logits_m = logits[mask]
        labels_m = labels[mask]

        metrics = compute_metrics(logits_m, labels_m)

        # PR-AUC for hate class
        metrics["pr_auc_hate"] = round(
            compute_pr_auc(metrics["_probs"], metrics["_labels"]), 4
        )

        # Optional loss
        if loss_fn is not None:
            metrics["loss"] = loss_fn(logits_m, labels_m).item()
        else:
            metrics["loss"] = float("nan")

        # Clean internal tensors from the returned dict
        metrics.pop("_probs",  None)
        metrics.pop("_labels", None)

        return metrics

    # ── Reporting ───────────────────────────────────────────────────

    def print_report(self, metrics: Dict, split_name: str = "") -> None:
        """Print a human-readable evaluation report.

        Parameters
        ----------
        metrics : dict
            Output of :meth:`evaluate` or :func:`compute_metrics`.
        split_name : str
            Label printed in the header.
        """
        header = f"Evaluation — {split_name}" if split_name else "Evaluation"
        print(f"\n{'='*58}")
        print(f"  {header}")
        print(f"{'='*58}")
        print(f"  Accuracy    : {metrics.get('accuracy',    'N/A'):.4f}")
        print(f"  Macro F1    : {metrics.get('macro_f1',    'N/A'):.4f}  ← primary metric")
        print(f"  Hate Recall : {metrics.get('hate_recall', 'N/A'):.4f}  ← safety metric")
        print(f"  PR-AUC hate : {metrics.get('pr_auc_hate', 'N/A')}")
        print(f"  Loss        : {metrics.get('loss',        'N/A')}")

        pc = metrics.get("per_class", {})
        if pc:
            print(f"\n  Per-class breakdown:")
            print(f"  {'Class':<14} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}")
            print(f"  {'-'*48}")
            for name, m in pc.items():
                marker = " ◄ SAFETY" if name == "hate_speech" else ""
                print(
                    f"  {name:<14} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                    f"{m['f1']:>7.4f} {m['support']:>9,}{marker}"
                )

        cm = metrics.get("confusion_matrix")
        if cm:
            print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
            print(f"  {'':14}", end="")
            for name in CLASS_NAMES:
                print(f"  {name[:8]:>8}", end="")
            print()
            for i, row in enumerate(cm):
                print(f"  {CLASS_NAMES[i]:<14}", end="")
                for val in row:
                    print(f"  {val:>8,}", end="")
                print()

        print(f"{'='*58}\n")

    def save_confusion_matrix(
        self,
        metrics:    Dict,
        save_path:  str,
        title:      str = "Confusion Matrix — Hate Speech",
    ) -> None:
        """Save a confusion-matrix plot to disk.

        Parameters
        ----------
        metrics : dict
            Output of :meth:`evaluate`.
        save_path : str
            Path to save the PNG.
        title : str
            Plot title.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; confusion matrix plot skipped.")
            return

        cm    = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=range(NUM_HATE_SPEECH_CLASSES),
            yticks=range(NUM_HATE_SPEECH_CLASSES),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            xlabel="Predicted",
            ylabel="Actual",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )

        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", save_path)
