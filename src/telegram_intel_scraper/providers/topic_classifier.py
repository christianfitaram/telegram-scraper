"""
Topic Classifier (zero-shot, BART MNLI)

Standalone helper focused only on topic classification.
- Uses facebook/bart-large-mnli zero-shot pipeline.
- Handles device selection (CUDA -> MPS -> CPU) and cache location.
- Provides both class-based and convenience function APIs plus a tiny CLI.

Usage (Python):
    from ingest.topic_classifier import TopicClassifier, get_topic, get_topics, DEFAULT_TOPICS
    clf = TopicClassifier()
    res = clf.classify("The Fed raised interest rates.")
    res = clf.classify("Messi scored twice", candidate_labels=["sports", "politics"], multi_label=False)
    print(res.top_label, res.top_score)

CLI examples:
    python -m ingest.topic_classifier --text "Stocks rallied after the Fed decision"
    python -m ingest.topic_classifier --text "Elections are near" --labels "politics,health,tech"
    python -m ingest.topic_classifier --file articles.txt --top-k 3
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "facebook/bart-large-mnli"
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE") 

DEFAULT_TOPICS: List[str] = [
    "politics and government",
    "sports and athletics",
    "science and research",
    "technology and innovation",
    "health and medicine",
    "business and finance",
    "entertainment and celebrity",
    "crime and justice",
    "climate and environment",
    "education and schools",
    "war and conflict",
    "travel and tourism",
]

# Device detection: prefer CUDA, then MPS, else CPU
TORCH_DEVICE = torch.device("cpu")
PIPELINE_DEVICE = -1
if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda:0")
    PIPELINE_DEVICE = 0
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    PIPELINE_DEVICE = -1  # HF pipeline expects int indices; keep CPU index

@dataclass(frozen=True)
class TopicResult:
    top_label: str
    top_score: float
    labels: List[str]
    scores: List[float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "top_label": self.top_label,
                "top_score": self.top_score,
                "labels": self.labels,
                "scores": self.scores,
            },
            ensure_ascii=False,
        )

class TopicClassifier:
    """Zero-shot topic classifier wrapper."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        candidate_labels: Optional[Sequence[str]] = None,
        cache_dir: Optional[str] = CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.candidate_labels = list(candidate_labels) if candidate_labels else list(DEFAULT_TOPICS)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        try:
            self.model.to(TORCH_DEVICE)
        except Exception:
            pass
        self.pipeline = self._create_pipeline_with_fallback()

    def _create_pipeline_with_fallback(self):
        try:
            return pipeline(
                task="zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=PIPELINE_DEVICE,
                max_length=512,
                truncation=True,
            )
        except Exception:
            return pipeline(
                task="zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
                max_length=512,
                truncation=True,
            )

    def classify(
        self,
        text: str,
        candidate_labels: Optional[Sequence[str]] = None,
        multi_label: bool = False,
        top_k: Optional[int] = None,
    ) -> Optional[TopicResult]:
        """Classify a single text. Returns None for empty input."""
        if not text or not text.strip():
            return None
        labels = list(candidate_labels) if candidate_labels else self.candidate_labels
        if not labels:
            raise ValueError("candidate_labels must be non-empty")

        result = self.pipeline(
            text.strip(),
            candidate_labels=labels,
            multi_label=multi_label,
        )

        labels_out = result["labels"]
        scores_out = [float(s) for s in result["scores"]]
        if top_k is not None:
            labels_out = labels_out[:top_k]
            scores_out = scores_out[:top_k]

        return TopicResult(
            top_label=labels_out[0],
            top_score=scores_out[0],
            labels=labels_out,
            scores=scores_out,
        )

    def classify_batch(
        self,
        texts: Iterable[str],
        candidate_labels: Optional[Sequence[str]] = None,
        multi_label: bool = False,
        top_k: Optional[int] = None,
    ) -> List[Optional[TopicResult]]:
        batch = [t.strip() if t is not None else "" for t in texts]
        if not batch:
            return []
        labels = list(candidate_labels) if candidate_labels else self.candidate_labels
        if not labels:
            raise ValueError("candidate_labels must be non-empty")

        indices = [i for i, t in enumerate(batch) if t]
        outputs: List[Optional[TopicResult]] = [None] * len(batch)
        if indices:
            inputs = [batch[i] for i in indices]
            preds = self.pipeline(inputs, candidate_labels=labels, multi_label=multi_label)
            # HF pipeline returns dict when single input; list of dicts for batch
            if isinstance(preds, dict):
                preds_list = [preds]
            else:
                preds_list = preds
            for i, pred in zip(indices, preds_list):
                labels_out = pred["labels"]
                scores_out = [float(s) for s in pred["scores"]]
                if top_k is not None:
                    labels_out = labels_out[:top_k]
                    scores_out = scores_out[:top_k]
                outputs[i] = TopicResult(
                    top_label=labels_out[0],
                    top_score=scores_out[0],
                    labels=labels_out,
                    scores=scores_out,
                )
        return outputs

# Convenience singleton API
_topic_singleton: Optional[TopicClassifier] = None

def _get_singleton() -> TopicClassifier:
    global _topic_singleton
    if _topic_singleton is None:
        _topic_singleton = TopicClassifier()
    return _topic_singleton

def get_topic(
    text: str,
    candidate_labels: Optional[Sequence[str]] = None,
    multi_label: bool = False,
    top_k: Optional[int] = None,
) -> Optional[TopicResult]:
    return _get_singleton().classify(text, candidate_labels=candidate_labels, multi_label=multi_label, top_k=top_k)

def get_topics(
    texts: Iterable[str],
    candidate_labels: Optional[Sequence[str]] = None,
    multi_label: bool = False,
    top_k: Optional[int] = None,
) -> List[Optional[TopicResult]]:
    return _get_singleton().classify_batch(texts, candidate_labels=candidate_labels, multi_label=multi_label, top_k=top_k)

# Minimal CLI

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Topic-only zero-shot classifier (BART MNLI)")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="Path to file with one text per line")
    parser.add_argument("--labels", type=str, help="Comma-separated candidate labels; defaults to built-in list")
    parser.add_argument("--multi-label", action="store_true", help="Enable multi-label mode")
    parser.add_argument("--top-k", type=int, default=None, help="Limit output to top K labels")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    clf = _get_singleton()

    if args.text:
        res = clf.classify(args.text, candidate_labels=labels, multi_label=args.multi_label, top_k=args.top_k)
        if args.json:
            print(res.to_json() if res else json.dumps(None))
        else:
            print(res)
        return 0

    if args.file:
        path = Path(args.file)
        lines = path.read_text(encoding="utf-8").splitlines()
        results = clf.classify_batch(lines, candidate_labels=labels, multi_label=args.multi_label, top_k=args.top_k)
        if args.json:
            print(json.dumps([r.__dict__ if r else None for r in results], ensure_ascii=False))
        else:
            for r in results:
                print(r)
        return 0

    parser.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(_cli())
