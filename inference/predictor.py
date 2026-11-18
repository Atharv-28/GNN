"""
HGCM-Net 2.0 — Unified Inference Predictor
============================================
HateSpeechPredictor is the single, stable entry point for all inference.

It loads a trained checkpoint, wraps it with the correct pre-processing,
and exposes ``predict()`` / ``predict_batch()`` that accept the stable
``PredictionRequest`` / ``BatchPredictionRequest`` schemas and return the
matching response objects.

This is the layer that future integrations — a FastAPI app, a browser
extension backend, a batch pipeline — will call.  The ML architecture
below it may evolve; this interface must remain stable.

Usage
-----
    from inference.predictor import HateSpeechPredictor
    from inference.schemas  import PredictionRequest

    predictor = HateSpeechPredictor(
        checkpoint_path="checkpoints/hate_speech/best_model_text_only.pt"
    )

    resp = predictor.predict(PredictionRequest(text="You should be deported."))
    print(resp.prediction, resp.confidence)   # → hate_speech  0.87
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ── make project root importable ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from config.labels import (
    IDX_TO_LABEL,
    NUM_HATE_SPEECH_CLASSES,
    LABEL_NORMAL,
    LABEL_OFFENSIVE,
    LABEL_HATE_SPEECH,
)
from preprocessing.text_processor import TextProcessor
from models.hgcm_net2 import HGCMNet2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_CHECKPOINT = Path("checkpoints/hate_speech/best_model_text_only.pt")
_DEFAULT_REVIEW_THRESHOLD = 0.70   # confidence below this → requires_review
_DEFAULT_HATE_HIGH_CONF   = 0.85   # hate below this but ≥ review_threshold → still review
_ENCODER_MODEL            = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────
#  HateSpeechPredictor
# ─────────────────────────────────────────────────────────────────────

class HateSpeechPredictor:
    """Load a trained HGCM-Net 2.0 checkpoint and serve predictions.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to the ``*.pt`` checkpoint saved by ``tasks/hate_speech/train.py``.
    device : str
        ``"cuda"`` or ``"cpu"``.  Defaults to CUDA if available.
    review_threshold : float
        Predictions with confidence below this value are flagged for
        human review regardless of the predicted class.
    hate_high_conf : float
        Hate-speech predictions below this value (but above
        review_threshold) are still flagged for review, because false
        negatives on hate content are more costly than false positives.
    """

    def __init__(
        self,
        checkpoint_path:  Path | str           = _DEFAULT_CHECKPOINT,
        device:           Optional[str]        = None,
        review_threshold: float                = _DEFAULT_REVIEW_THRESHOLD,
        hate_high_conf:   float                = _DEFAULT_HATE_HIGH_CONF,
    ) -> None:
        self.checkpoint_path  = Path(checkpoint_path)
        self.review_threshold = review_threshold
        self.hate_high_conf   = hate_high_conf
        self.device           = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._text_processor  = TextProcessor()
        self._encoder: Optional[SentenceTransformer] = None   # lazy-loaded
        self._model:   Optional[HGCMNet2]            = None   # lazy-loaded
        self._model_version:  str                    = "unloaded"
        self._config                                 = None

        logger.info("HateSpeechPredictor initialised (checkpoint not yet loaded).")

    # ── Lazy loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load model + encoder on first call; subsequent calls are no-ops."""
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Run  python tasks/hate_speech/train.py  to train the model first."
            )

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # ── Re-build config from saved dict ──────────────────────────
        # The checkpoint stores config as a plain dict (pickle-safe).
        # Reconstruct the HGCMConfig object from it.
        import importlib.util, types
        config_path = _ROOT / "config.py"
        spec = importlib.util.spec_from_file_location("_hgcm_root_config", config_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["_hgcm_root_config"] = mod
        spec.loader.exec_module(mod)
        HGCMConfig = mod.HGCMConfig

        config_dict = ckpt.get("config_dict", {})
        self._config = HGCMConfig()
        for k, v in config_dict.items():
            if hasattr(self._config, k):
                setattr(self._config, k, v)

        # ── Build model ───────────────────────────────────────────────
        # HeteroGATBackbone has NO lazy layers — load_state_dict alone is
        # sufficient; no dummy forward pass is needed.
        model = HGCMNet2(self._config, num_classes=NUM_HATE_SPEECH_CLASSES)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self._model = model

        self._model_version = ckpt.get(
            "model_version",
            f"hgcm-net-2.0-hatexplain-epoch{ckpt.get('epoch', '?')}"
        )
        logger.info("Model loaded — version: %s  device: %s", self._model_version, self.device)

        # ── Load SentenceTransformer encoder ─────────────────────────
        logger.info("Loading SentenceTransformer encoder (%s) …", _ENCODER_MODEL)
        self._encoder = SentenceTransformer(_ENCODER_MODEL, device=str(self.device))
        logger.info("Encoder ready.")

    # ── Core encode + forward ────────────────────────────────────────

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Tokenise + embed texts.  Returns (N, 384) on self.device."""
        # TextProcessor cleans the text the same way as during training
        clean = [self._text_processor.clean(t) for t in texts]
        embeddings = self._encoder.encode(
            clean,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device)                      # (N, 384)

    def _build_dummy_graph(self, n: int) -> tuple[dict, dict]:
        """Build the minimal x_dict / edge_index_dict for text-only forward."""
        # In text-only mode the GAT is bypassed, so we only need the
        # content features.  The user tensor must exist for lazy init.
        x_dict = {
            "content": torch.zeros(n, self._config.content_embed_dim, device=self.device),
            "user":    torch.zeros(1, self._config.user_raw_dim,      device=self.device),
        }
        edge_dict = {
            ("user", "authored", "content"):    torch.zeros(2, 0, dtype=torch.long, device=self.device),
            ("content", "rev_authored", "user"): torch.zeros(2, 0, dtype=torch.long, device=self.device),
        }
        return x_dict, edge_dict

    @torch.no_grad()
    def _forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run the model forward in text-only mode.  Returns softmax probs (N, 3)."""
        n = embeddings.size(0)
        x_dict, edge_dict = self._build_dummy_graph(n)
        # Inject the real ST embeddings into x_dict["content"]
        x_dict["content"] = embeddings
        logits, _, _ = self._model(x_dict, edge_dict, text_only=True)
        return F.softmax(logits, dim=-1)                       # (N, 3)

    # ── Review flag logic ────────────────────────────────────────────

    def _requires_review(self, label: str, confidence: float) -> bool:
        """Determine if a prediction should be flagged for human review."""
        if confidence < self.review_threshold:
            return True
        # Hate predictions below the high-confidence bar still need review
        if label == LABEL_HATE_SPEECH and confidence < self.hate_high_conf:
            return True
        return False

    # ── Public API ───────────────────────────────────────────────────

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Classify a single post.

        Parameters
        ----------
        request : PredictionRequest
            Input post.  ``platform`` is logged but not fed to the model.
            ``context`` is accepted but ignored in text-only mode (Phase 2).

        Returns
        -------
        PredictionResponse
        """
        self._ensure_loaded()
        emb   = self._encode([request.text])          # (1, 384)
        probs = self._forward(emb)[0]                 # (3,)

        confidence = float(probs.max())
        pred_idx   = int(probs.argmax())
        label      = IDX_TO_LABEL[pred_idx]

        scores = {
            LABEL_NORMAL:      float(probs[0]),
            LABEL_OFFENSIVE:   float(probs[1]),
            LABEL_HATE_SPEECH: float(probs[2]),
        }

        return PredictionResponse(
            prediction     = label,
            confidence     = confidence,
            scores         = scores,
            requires_review= self._requires_review(label, confidence),
            model_version  = self._model_version,
            inference_mode = "text_only",
        )

    def predict_batch(
        self,
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        """Classify a batch of posts efficiently.

        Parameters
        ----------
        request : BatchPredictionRequest
            Up to 512 posts.

        Returns
        -------
        BatchPredictionResponse
        """
        self._ensure_loaded()

        texts = [r.text for r in request.posts]
        embs  = self._encode(texts)                   # (N, 384)
        probs = self._forward(embs)                   # (N, 3)

        responses: list[PredictionResponse] = []
        for i, req in enumerate(request.posts):
            p          = probs[i]
            confidence = float(p.max())
            pred_idx   = int(p.argmax())
            label      = IDX_TO_LABEL[pred_idx]
            scores     = {
                LABEL_NORMAL:      float(p[0]),
                LABEL_OFFENSIVE:   float(p[1]),
                LABEL_HATE_SPEECH: float(p[2]),
            }
            responses.append(PredictionResponse(
                prediction     = label,
                confidence     = confidence,
                scores         = scores,
                requires_review= self._requires_review(label, confidence),
                model_version  = self._model_version,
                inference_mode = "text_only",
            ))

        total_flagged = sum(1 for r in responses if r.requires_review)
        return BatchPredictionResponse(
            predictions   = responses,
            model_version = self._model_version,
            total_flagged = total_flagged,
        )
