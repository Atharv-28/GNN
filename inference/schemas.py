"""
Inference Schemas — HGCM-Net 2.0
==================================
Pydantic data-models defining the stable input / output contract for
the hate-speech (and future) inference interface.

These schemas are the integration boundary between the ML system and
any external caller — a browser extension, a REST API, a batch script,
or a future platform integration.

Rules
-----
- The schema must remain backward-compatible.  Adding optional fields
  is safe; removing or renaming existing fields is a breaking change.
- ``platform`` is metadata only.  It is logged but never fed into the
  model's forward pass.
- ``context`` carries optional social-graph information.  Its absence
  must never prevent a prediction.

Usage
-----
    from inference.schemas import PredictionRequest, PredictionResponse

    req  = PredictionRequest(text="You are trash.", platform="x")
    resp = PredictionResponse(
        prediction="offensive",
        confidence=0.78,
        scores={"normal": 0.10, "offensive": 0.78, "hate_speech": 0.12},
        requires_review=True,
        model_version="hgcm-net-2.0-hatexplain-v1",
        inference_mode="text_only",
    )
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    # Graceful degradation: fall back to plain dataclasses if pydantic is absent
    _PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass, field as dc_field


if _PYDANTIC_AVAILABLE:

    class GraphContext(BaseModel):
        """Optional social-graph context for context-aware inference.

        All fields are optional; missing fields are silently ignored.
        The model degrades to text-only mode if context is insufficient.
        """
        author_id:        Optional[str]        = None
        platform_user_id: Optional[str]        = None
        following_count:  Optional[int]        = None
        follower_count:   Optional[int]        = None
        post_count:       Optional[int]        = None
        account_age_days: Optional[int]        = None
        verified:         Optional[bool]       = None
        # Interaction signals (for future graph-enhanced inference)
        reply_to_post_id: Optional[str]        = None
        conversation_ids: Optional[List[str]]  = None


    class PredictionRequest(BaseModel):
        """Input contract for a single hate-speech prediction.

        Parameters
        ----------
        text : str
            The raw social-media post text.  Required; must be non-empty.
        platform : str | None
            Source platform identifier.  Metadata only — not used by the
            classifier.  Logged for dataset stratification and analysis.
        context : GraphContext | None
            Optional social-graph context.  When None (the default), the
            model uses text-only inference mode.
        model_version : str | None
            Request a specific model checkpoint version.  Future use.
        """
        text:          str
        platform:      Optional[Literal["x", "instagram", "reddit",
                                        "facebook", "youtube", "other"]] = None
        context:       Optional[GraphContext] = None
        model_version: Optional[str]         = None

        @field_validator("text")
        @classmethod
        def text_must_not_be_empty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("text must be a non-empty string")
            return v


    class BatchPredictionRequest(BaseModel):
        """Input contract for batch predictions.

        Parameters
        ----------
        posts : list[PredictionRequest]
            List of posts to classify.  Maximum 512 posts per batch.
        """
        posts: List[PredictionRequest] = Field(..., min_length=1, max_length=512)


    class PredictionResponse(BaseModel):
        """Output contract for a single hate-speech prediction.

        Parameters
        ----------
        prediction : str
            The predicted class: ``"normal"``, ``"offensive"``, or
            ``"hate_speech"``.
        confidence : float
            Probability of the predicted class (max softmax score).
        scores : dict[str, float]
            Full softmax probability distribution over all three classes.
            Values sum to 1.0.
        requires_review : bool
            True when the prediction should be flagged for human review.
            Triggered when confidence is below the review threshold, or
            when the predicted class is ``hate_speech`` below the high-
            confidence threshold.
        model_version : str
            Version string of the model that produced this prediction.
        inference_mode : str
            ``"text_only"`` or ``"context_aware"``.
        input_text : str | None
            Echo of the input text.  Included when the caller requests it
            (future flag); None by default to avoid storing sensitive text.
        """
        prediction:     Literal["normal", "offensive", "hate_speech"]
        confidence:     float
        scores:         Dict[str, float]
        requires_review: bool
        model_version:  str
        inference_mode: Literal["text_only", "context_aware"]
        input_text:     Optional[str] = None


    class BatchPredictionResponse(BaseModel):
        """Output contract for batch predictions."""
        predictions: List[PredictionResponse]
        model_version: str
        total_flagged: int    # count of requires_review=True in this batch

else:
    # ── Fallback dataclasses (no Pydantic) ───────────────────────────
    # Field validation is skipped; callers must validate inputs themselves.

    from dataclasses import dataclass
    from typing import Optional, Dict, List, Literal

    @dataclass
    class GraphContext:
        author_id:        Optional[str]  = None
        platform_user_id: Optional[str]  = None
        following_count:  Optional[int]  = None
        follower_count:   Optional[int]  = None
        post_count:       Optional[int]  = None
        account_age_days: Optional[int]  = None
        verified:         Optional[bool] = None
        reply_to_post_id: Optional[str]  = None
        conversation_ids: Optional[List[str]] = None

    @dataclass
    class PredictionRequest:
        text:          str
        platform:      Optional[str] = None
        context:       Optional[GraphContext] = None
        model_version: Optional[str] = None

    @dataclass
    class BatchPredictionRequest:
        posts: List[PredictionRequest] = dc_field(default_factory=list)

    @dataclass
    class PredictionResponse:
        prediction:      str
        confidence:      float
        scores:          Dict[str, float]
        requires_review: bool
        model_version:   str
        inference_mode:  str
        input_text:      Optional[str] = None

    @dataclass
    class BatchPredictionResponse:
        predictions:   List[PredictionResponse]
        model_version: str
        total_flagged: int
