"""
Data Pipeline for HGCM-Net
===========================
Downloads WELFake dataset (72K+ articles), pre-computes sentence embeddings,
creates synthetic user nodes via K-means clustering, and builds the
heterogeneous graph.  All heavy computation is cached to disk.
"""

import os
import re
import logging
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.cluster import MiniBatchKMeans

from config import HGCMConfig
from graph_builder import compute_edge_weights_batch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

CACHE_DIR = Path("cached_data")


# ====================================================================
#  Text feature extraction (for user-node metadata)
# ====================================================================

def _text_features(text: str) -> list:
    """Extract 10 simple text-statistics features from a string."""
    if not text or not isinstance(text, str):
        text = ""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_sentences = max(1, len(re.split(r'[.!?]+', text)))
    n_upper = sum(1 for c in text if c.isupper())
    n_digits = sum(1 for c in text if c.isdigit())
    n_punct = sum(1 for c in text if c in '!?.,;:"\'-()[]{}')
    unique_words = len(set(w.lower() for w in words)) if words else 0
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    return [
        n_words,                                    # 0  word count
        n_chars,                                    # 1  char count
        n_sentences,                                # 2  sentence count
        n_words / max(n_sentences, 1),              # 3  words per sentence
        avg_word_len,                               # 4  avg word length
        unique_words / max(n_words, 1),             # 5  vocabulary richness
        n_upper / max(n_chars, 1),                  # 6  uppercase ratio
        n_digits / max(n_chars, 1),                 # 7  digit ratio
        n_punct / max(n_chars, 1),                  # 8  punctuation ratio
        1.0 if n_words > 0 else 0.0,                # 9  has-content flag
    ]


# ====================================================================
#  Download & prepare dataset
# ====================================================================

def download_welfake():
    """Download the WELFake dataset from HuggingFace."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset
    logger.info("Downloading WELFake dataset from HuggingFace...")
    ds = load_dataset("davanstrien/WELFake", split="train")
    logger.info("Downloaded %d samples", len(ds))
    return ds


def prepare_texts(ds) -> tuple:
    """Clean and extract texts + labels from the dataset.

    Returns (texts, labels) where texts combine title+text.
    """
    texts, labels = [], []
    skipped = 0
    for row in ds:
        title = row.get("title", "") or ""
        body = row.get("text", "") or ""
        label = row.get("label", None)
        combined = f"{title.strip()}. {body.strip()}"
        if len(combined.strip()) < 20 or label is None:
            skipped += 1
            continue
        # Truncate very long texts for embedding efficiency
        texts.append(combined[:2048])
        labels.append(int(label))
    logger.info("Prepared %d texts (%d skipped)", len(texts), skipped)
    return texts, labels


# ====================================================================
#  Pre-compute sentence embeddings (CPU / batched GPU)
# ====================================================================

def encode_texts(texts, config, cache_path=None):
    """Encode texts with SentenceTransformer, with disk caching."""
    if cache_path and Path(cache_path).exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Encoding %d texts with %s on %s...",
                len(texts), config.sentence_model_name, device)

    model = SentenceTransformer(config.sentence_model_name)
    # Encode in batches to control memory (128 fits well in 4GB VRAM)
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )
    embeddings = embeddings.float()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        logger.info("Saved embeddings to %s (%.1f MB)",
                    cache_path, embeddings.nelement() * 4 / 1e6)
    return embeddings


# ====================================================================
#  Create user nodes via K-means clustering
# ====================================================================

def create_user_nodes(embeddings, texts, n_users=800, config=None):
    """Cluster content embeddings to create synthetic user nodes.

    Each cluster = one "user" (latent source archetype).
    User features are text-statistics aggregated per cluster.

    Returns (user_features, assignments) where assignments[i] = user_id
    for content node i.
    """
    config = config or HGCMConfig()
    logger.info("Clustering %d embeddings into %d user nodes...",
                embeddings.shape[0], n_users)

    emb_np = embeddings.cpu().numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=n_users, batch_size=2048,
        random_state=42, n_init=3,
    )
    assignments = kmeans.fit_predict(emb_np)

    # Compute per-user (per-cluster) text features
    logger.info("Computing user features...")
    all_text_feats = np.array([_text_features(t) for t in texts])  # (N, 10)

    user_features = np.zeros((n_users, config.user_raw_dim), dtype=np.float32)
    for uid in range(n_users):
        mask = assignments == uid
        if mask.sum() == 0:
            continue
        cluster_feats = all_text_feats[mask]  # (n_members, 10)
        user_features[uid] = cluster_feats.mean(axis=0)

    # Normalise user features (zero-mean, unit-var)
    mu = user_features.mean(axis=0, keepdims=True)
    std = user_features.std(axis=0, keepdims=True) + 1e-8
    user_features = (user_features - mu) / std

    logger.info("User features shape: %s", user_features.shape)
    return torch.tensor(user_features, dtype=torch.float32), assignments


# ====================================================================
#  Build interactions (edges) with temporal decay
# ====================================================================

def build_interactions(assignments, n_contents, rng_seed=42):
    """Create user-content interactions with synthetic temporal data.

    Primary edge:  each content → its assigned user cluster.
    Secondary edges: random cross-cluster links (simulates sharing).

    Returns list of (user_idx, content_idx, rho, tau_hours).
    """
    rng = np.random.RandomState(rng_seed)
    interactions = []

    # Primary edges: each content linked to its user
    for cid in range(n_contents):
        uid = int(assignments[cid])
        rho = rng.uniform(0.7, 1.0)
        tau = rng.uniform(0, 48)  # 0-48 hours
        interactions.append((uid, cid, rho, tau))

    # Secondary edges: ~20% extra cross-links
    n_extra = int(0.2 * n_contents)
    for _ in range(n_extra):
        cid = rng.randint(0, n_contents)
        uid = rng.randint(0, int(assignments.max()) + 1)
        rho = rng.uniform(0.3, 0.7)
        tau = rng.uniform(24, 720)  # 1-30 days
        interactions.append((uid, cid, rho, tau))

    logger.info("Created %d interactions (%d primary + %d cross-links)",
                len(interactions), n_contents, n_extra)
    return interactions


# ====================================================================
#  Assemble the full HeteroData graph
# ====================================================================

def build_hetero_graph(content_emb, user_features, labels,
                       interactions, config=None):
    """Build HeteroData from pre-computed components."""
    config = config or HGCMConfig()
    data = HeteroData()

    # Content nodes
    data["content"].x = content_emb
    data["content"].y = torch.tensor(labels, dtype=torch.long)
    data["content"].num_nodes = content_emb.size(0)

    # User nodes
    data["user"].x = user_features
    data["user"].num_nodes = user_features.size(0)

    # Edges with temporal decay weights
    src_list, dst_list, rho_list, tau_list = zip(*interactions)
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    rhos = torch.tensor(rho_list, dtype=torch.float)
    taus = torch.tensor(tau_list, dtype=torch.float)
    weights = compute_edge_weights_batch(rhos, taus)

    et = ("user", "interacts_with", "content")
    data[et].edge_index = torch.stack([src, dst], dim=0)
    data[et].edge_weight = weights

    ret = ("content", "rev_interacts_with", "user")
    data[ret].edge_index = torch.stack([dst, src], dim=0)
    data[ret].edge_weight = weights

    return data


# ====================================================================
#  Main pipeline
# ====================================================================

def run_pipeline(config=None, n_users=800, force=False):
    """Execute the full data pipeline. Returns HeteroData.

    Results are cached in ./cached_data/.  Pass force=True to rebuild.
    """
    config = config or HGCMConfig()
    CACHE_DIR.mkdir(exist_ok=True)

    graph_path = CACHE_DIR / "hetero_graph.pt"
    if graph_path.exists() and not force:
        logger.info("Loading cached graph from %s", graph_path)
        return torch.load(graph_path, map_location="cpu", weights_only=False)

    # Step 1: Download
    ds = download_welfake()
    texts, labels = prepare_texts(ds)

    # Step 2: Encode
    emb_path = CACHE_DIR / "content_embeddings.pt"
    content_emb = encode_texts(texts, config, cache_path=emb_path)

    # Step 3: Create users
    user_features, assignments = create_user_nodes(
        content_emb, texts, n_users=n_users, config=config,
    )

    # Step 4: Build interactions
    interactions = build_interactions(assignments, len(texts))

    # Step 5: Build graph
    data = build_hetero_graph(
        content_emb, user_features,
        labels, interactions, config,
    )

    # Save
    torch.save(data, graph_path)
    sz = graph_path.stat().st_size / 1e6
    logger.info("Graph saved to %s (%.1f MB)", graph_path, sz)
    logger.info("Graph: %s", data)
    return data


if __name__ == "__main__":
    data = run_pipeline()
    print("\n=== Graph Summary ===")
    print(f"Content nodes: {data['content'].num_nodes}")
    print(f"User nodes:    {data['user'].num_nodes}")
    print(f"Labels:        {data['content'].y.unique(return_counts=True)}")
    et = ("user", "interacts_with", "content")
    print(f"Edges:         {data[et].edge_index.shape[1]}")
    print(f"Edge weights:  [{data[et].edge_weight.min():.4f}, "
          f"{data[et].edge_weight.max():.4f}]")
