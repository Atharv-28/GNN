"""
HateXplain Data Pipeline — HGCM-Net 2.0
=========================================
Full pipeline: HateXplain text → ST embeddings → synthetic user nodes
(K-means) → HeteroData graph, mirroring the design of data_pipeline.py
for WELFake.

Outputs are cached to disk under ``cached_data/hate_speech/``.

Usage
-----
    from tasks.hate_speech.pipeline import HateSpeechPipeline

    pipeline = HateSpeechPipeline()
    splits = pipeline.run()
    # splits["train"], splits["validation"], splits["test"] — each HeteroData

    # Or a single split (useful during debugging)
    train_data = pipeline.run_split("train")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import HeteroData

# ── Project root on path ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.labels import UNCERTAIN_LABEL, NUM_HATE_SPEECH_CLASSES
from preprocessing.text_processor import TextProcessor, extract_text_features_batch
from tasks.hate_speech.dataset import (
    HateXplainLoader,
    load_hatexplain_for_training,
    COL_TEXT,
    COL_LABEL,
    COL_SOURCE,
    COL_UNCERTAIN,
)
from graph_builder import compute_edge_weights_batch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ─────────────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────────────

CACHE_BASE       = Path("cached_data") / "hate_speech"
SENTENCE_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CONTENT_EMBED_DIM = 384
USER_RAW_DIM     = 10          # matches config.py user_raw_dim
N_USERS_DEFAULT  = 400         # K-means clusters (smaller than WELFake: HateXplain ~20K)


# ─────────────────────────────────────────────────────────────────────
#  Pipeline class
# ─────────────────────────────────────────────────────────────────────

class HateSpeechPipeline:
    """End-to-end data pipeline for the hate-speech task.

    Parameters
    ----------
    n_users : int
        Number of synthetic user nodes (K-means clusters).
    sentence_model_name : str
        HuggingFace sentence-transformer model ID.
    cache_dir : Path | str | None
        Directory for caching embeddings and graphs.
        Defaults to ``cached_data/hate_speech/``.
    force_rebuild : bool
        If True, ignore existing cache and recompute everything.
    """

    def __init__(
        self,
        n_users: int               = N_USERS_DEFAULT,
        sentence_model_name: str   = SENTENCE_MODEL,
        cache_dir: Optional[Path]  = None,
        force_rebuild: bool        = False,
    ) -> None:
        self.n_users             = n_users
        self.sentence_model_name = sentence_model_name
        self.cache_dir           = Path(cache_dir) if cache_dir else CACHE_BASE
        self.force_rebuild       = force_rebuild
        self.processor           = TextProcessor()
        self._st_model           = None   # lazy load

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────

    def run(self, splits=("train", "validation", "test")) -> Dict[str, HeteroData]:
        """Run the full pipeline for all requested splits.

        K-means user nodes are built on the *training split only* and then
        applied to val/test splits.  This prevents data leakage through
        the user clustering step.

        Parameters
        ----------
        splits : tuple[str]
            Splits to process.

        Returns
        -------
        dict[str, HeteroData]
        """
        logger.info("=" * 60)
        logger.info("HateSpeechPipeline — full run, splits=%s", splits)
        logger.info("=" * 60)

        # ── 1. Determine which splits to load ───────────────────────
        # K-means user nodes are built on TRAIN only.
        # If train is not in the requested splits, we still need to load
        # it for clustering, then discard its graph (unless requested).
        splits_set        = set(splits)
        need_train_for_km = "train" not in splits_set
        splits_to_load    = splits_set | {"train"}  # always load train for K-means
        loader = HateXplainLoader()
        all_dfs = {}
        for split in splits_to_load:
            all_dfs[split] = load_hatexplain_for_training(split=split)

        # ── 2. Clean texts ──────────────────────────────────────────
        all_texts  = {}
        all_labels = {}
        for split, df in all_dfs.items():
            texts  = self.processor.clean_batch(df[COL_TEXT].tolist())
            labels = df[COL_LABEL].tolist()
            all_texts[split]  = texts
            all_labels[split] = labels
            logger.info("Split '%s': %d samples", split, len(texts))

        # ── 3. Encode all texts (embedding cached per split) ────────
        all_embeddings = {}
        for split in splits_to_load:
            emb = self._encode_texts(
                all_texts[split],
                cache_path=self.cache_dir / f"embeddings_{split}.pt",
            )
            all_embeddings[split] = emb

        # ── 4. Build K-means user nodes from TRAINING split only ────
        logger.info("Building K-means user nodes from training split ...")
        train_emb = all_embeddings["train"]
        kmeans, user_features, train_assignments = self._build_user_nodes(
            train_emb,
            all_texts["train"],
            cache_path=self.cache_dir / "kmeans_user_features.pt",
        )

        # ── 5. Assign val/test texts to nearest cluster ─────────────
        all_assignments = {"train": train_assignments}
        for split in splits_to_load:
            if split == "train":
                continue
            emb_np = all_embeddings[split].cpu().numpy()
            assignments = kmeans.predict(emb_np)
            all_assignments[split] = assignments
            logger.info(
                "Split '%s': assigned %d samples to %d clusters",
                split, len(assignments), self.n_users,
            )

        # ── 6. Build HeteroData graph — only for originally requested splits ──
        result = {}
        for split in splits:
            graph_path = self.cache_dir / f"hetero_graph_{split}.pt"
            if graph_path.exists() and not self.force_rebuild:
                logger.info("Loading cached graph for split='%s'", split)
                result[split] = torch.load(graph_path, map_location="cpu",
                                           weights_only=False)
                continue

            data = self._build_graph(
                embeddings   = all_embeddings[split],
                labels       = all_labels[split],
                user_features= user_features,
                assignments  = all_assignments[split],
                split_name   = split,
            )
            torch.save(data, graph_path)
            sz = graph_path.stat().st_size / 1e6
            logger.info("Saved '%s' graph → %s (%.1f MB)", split, graph_path, sz)
            result[split] = data

        self._print_summary(result)
        return result

    def run_split(self, split: str) -> HeteroData:
        """Run the pipeline for a single split (convenience wrapper)."""
        results = self.run(splits=(split,))
        return results[split]

    # ── Embedding ───────────────────────────────────────────────────

    def _encode_texts(
        self,
        texts: list[str],
        cache_path: Optional[Path] = None,
    ) -> torch.Tensor:
        """Encode texts with SentenceTransformer; cache to disk."""
        if cache_path and cache_path.exists() and not self.force_rebuild:
            logger.info("Loading cached embeddings: %s", cache_path)
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        st = self._get_st_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Encoding %d texts with %s on %s ...",
            len(texts), self.sentence_model_name, device,
        )

        embeddings = st.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
        ).float()

        if cache_path:
            torch.save(embeddings, cache_path)
            logger.info(
                "Saved embeddings → %s  (%.1f MB)",
                cache_path, embeddings.nelement() * 4 / 1e6,
            )
        return embeddings

    def _get_st_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.sentence_model_name)
        return self._st_model

    # ── User nodes ──────────────────────────────────────────────────

    def _build_user_nodes(
        self,
        embeddings: torch.Tensor,
        texts: list[str],
        cache_path: Optional[Path] = None,
    ) -> Tuple[MiniBatchKMeans, torch.Tensor, np.ndarray]:
        """Cluster content embeddings to form synthetic user nodes.

        Returns
        -------
        (kmeans, user_features, assignments)
            kmeans        : fitted MiniBatchKMeans (use .predict() for new splits)
            user_features : (n_users, USER_RAW_DIM) normalised feature tensor
            assignments   : (N_content,) int array, cluster index per sample
        """
        if cache_path and cache_path.exists() and not self.force_rebuild:
            logger.info("Loading cached user features: %s", cache_path)
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            return saved["kmeans"], saved["user_features"], saved["assignments"]

        logger.info(
            "K-means clustering: %d samples → %d user nodes ...",
            embeddings.shape[0], self.n_users,
        )

        emb_np = embeddings.cpu().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_users,
            batch_size=2048,
            random_state=42,
            n_init=5,
        )
        assignments = kmeans.fit_predict(emb_np)

        # Aggregate text-statistic features per cluster
        logger.info("Computing per-cluster text features ...")
        text_feats = extract_text_features_batch(texts)   # (N, 10)

        user_feats = np.zeros((self.n_users, USER_RAW_DIM), dtype=np.float32)
        for uid in range(self.n_users):
            mask = assignments == uid
            if mask.sum() == 0:
                continue
            user_feats[uid] = text_feats[mask].mean(axis=0)

        # Normalise (zero-mean, unit-var per feature)
        mu  = user_feats.mean(axis=0, keepdims=True)
        std = user_feats.std(axis=0,  keepdims=True) + 1e-8
        user_feats = (user_feats - mu) / std

        user_features_tensor = torch.tensor(user_feats, dtype=torch.float32)

        if cache_path:
            torch.save(
                {"kmeans": kmeans, "user_features": user_features_tensor,
                 "assignments": assignments},
                cache_path,
            )
            logger.info("Saved user nodes → %s", cache_path)

        return kmeans, user_features_tensor, assignments

    # ── Graph construction ──────────────────────────────────────────

    def _build_graph(
        self,
        embeddings:    torch.Tensor,
        labels:        list[int],
        user_features: torch.Tensor,
        assignments:   np.ndarray,
        split_name:    str = "",
        rng_seed:      int = 42,
    ) -> HeteroData:
        """Construct HeteroData with content nodes, user nodes, and edges.

        Edge strategy (mirrors data_pipeline.py):
        - Primary edge: each content → its assigned K-means cluster (user).
        - Secondary edges: ~20 % random cross-cluster links with weaker ρ.
        - Edge weights: temporal-decay  w = ρ / (1 + log(1 + τ))
        """
        rng = np.random.RandomState(rng_seed)
        n_content = embeddings.shape[0]

        # Primary edges
        interactions = []
        for cid in range(n_content):
            uid = int(assignments[cid])
            rho = rng.uniform(0.7, 1.0)
            tau = rng.uniform(0.0, 48.0)   # hours since post creation
            interactions.append((uid, cid, rho, tau))

        # Secondary cross-cluster edges (~20 %)
        n_extra = int(0.2 * n_content)
        max_uid = int(assignments.max()) + 1
        for _ in range(n_extra):
            cid = rng.randint(0, n_content)
            uid = rng.randint(0, max_uid)
            rho = rng.uniform(0.3, 0.7)
            tau = rng.uniform(24.0, 720.0)
            interactions.append((uid, cid, rho, tau))

        src_l, dst_l, rho_l, tau_l = zip(*interactions)
        src   = torch.tensor(src_l, dtype=torch.long)
        dst   = torch.tensor(dst_l, dtype=torch.long)
        rhos  = torch.tensor(rho_l, dtype=torch.float)
        taus  = torch.tensor(tau_l, dtype=torch.float)
        weights = compute_edge_weights_batch(rhos, taus)

        data = HeteroData()

        # Content nodes
        data["content"].x         = embeddings.float()
        data["content"].num_nodes = n_content
        data["content"].y         = torch.tensor(labels, dtype=torch.long)

        # User nodes
        data["user"].x         = user_features.float()
        data["user"].num_nodes = user_features.shape[0]

        # Edges
        et  = ("user",    "interacts_with",     "content")
        ret = ("content", "rev_interacts_with",  "user")
        data[et].edge_index  = torch.stack([src, dst], dim=0)
        data[et].edge_weight = weights
        data[ret].edge_index = torch.stack([dst, src], dim=0)
        data[ret].edge_weight = weights

        logger.info(
            "Graph '%s': content=%d  users=%d  edges=%d  "
            "edge_w∈[%.4f, %.4f]",
            split_name, n_content, user_features.shape[0],
            len(interactions),
            weights.min().item(), weights.max().item(),
        )
        return data

    # ── Reporting ───────────────────────────────────────────────────

    def _print_summary(self, splits: Dict[str, HeteroData]) -> None:
        from config.labels import HATE_SPEECH_LABELS
        print("\n" + "=" * 60)
        print("  HateXplain Graph Summary")
        print("=" * 60)
        for split_name, data in splits.items():
            y = data["content"].y
            n = y.shape[0]
            et = ("user", "interacts_with", "content")
            n_edges = data[et].edge_index.shape[1]
            print(f"\n  Split: {split_name}  ({n} content nodes, "
                  f"{data['user'].num_nodes} user nodes, {n_edges} edges)")
            counts = y.bincount(minlength=NUM_HATE_SPEECH_CLASSES)
            for idx, name in HATE_SPEECH_LABELS.items():
                cnt = counts[idx].item()
                pct = 100 * cnt / n if n > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    [{idx}] {name:<12}: {cnt:>5,}  ({pct:5.1f}%)  {bar}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────
#  Class weight computation
# ─────────────────────────────────────────────────────────────────────

def compute_class_weights(
    labels: list[int] | torch.Tensor,
    num_classes: int = NUM_HATE_SPEECH_CLASSES,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Parameters
    ----------
    labels : list[int] | Tensor
        Training set labels (integers 0, 1, 2).
    num_classes : int
        Total number of classes.
    device : str
        Target device for the returned tensor.

    Returns
    -------
    Tensor, shape ``(num_classes,)``
        Weights passed as ``weight=`` argument to ``nn.CrossEntropyLoss``.
    """
    if isinstance(labels, torch.Tensor):
        label_list = labels.tolist()
    else:
        label_list = [l for l in labels if l != UNCERTAIN_LABEL]

    try:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        classes = list(range(num_classes))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(classes),
            y=np.array(label_list),
        )
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    except Exception as e:
        logger.warning("Class weight computation failed (%s); using uniform weights.", e)
        w_tensor = torch.ones(num_classes, dtype=torch.float32, device=device)

    logger.info(
        "Class weights: %s",
        {f"class_{i}": f"{w:.4f}" for i, w in enumerate(w_tensor.tolist())},
    )
    return w_tensor


# ─────────────────────────────────────────────────────────────────────
#  Entry point (smoke test)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    pipeline = HateSpeechPipeline(n_users=400, force_rebuild=False)
    splits   = pipeline.run()

    # Class weights for training
    train_labels = splits["train"]["content"].y.tolist()
    weights = compute_class_weights(train_labels)
    print(f"\nClass weights for CrossEntropyLoss: {weights.tolist()}")
