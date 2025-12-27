"""
Sentiment Detector

A lightweight, sentiment-only utility built on Hugging Face
`distilbert-base-uncased-finetuned-sst-2-english`.

- Focuses purely on sentiment analysis (no DB or topic logic)
- Handles device selection (CUDA, MPS, CPU) defensively
- Respects TRANSFORMERS_CACHE if provided, else uses local models/transformers

Usage (Python):
    from ingest.sentiment_detector import SentimentDetector, get_sentiment, get_sentiments
    det = SentimentDetector()
    print(det.analyze("I love this product!"))
    print(get_sentiment("Terrible experience."))
    print(get_sentiments(["Great service", "Bad UI"]))

CLI:
    python -m ingest.sentiment_detector --text "Amazing result!"
    python -m ingest.sentiment_detector --file path/to/texts.txt  # one text per line
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# Model and cache configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", None)
# Device detection: prefer CUDA → MPS → CPU
TORCH_DEVICE = torch.device("cpu")
PIPELINE_DEVICE = -1  # -1 means CPU for HF pipeline
if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda:0")
    PIPELINE_DEVICE = 0
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    # HF pipeline typically expects int device ids; keep CPU index and move weights
    PIPELINE_DEVICE = -1

@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float

    def to_json(self) -> str:
        return json.dumps({"label": self.label, "score": self.score}, ensure_ascii=False)

class SentimentDetector:
    """Encapsulates a sentiment pipeline with robust initialization."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Optional[str] = CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        # Place weights on selected torch device when feasible
        try:
            self.model.to(TORCH_DEVICE)
        except Exception:
            pass
        # Try to create pipeline on preferred device; fallback to CPU if needed
        self.pipeline = self._create_pipeline_with_fallback()

    def _create_pipeline_with_fallback(self):
        try:
            return pipeline(
                task="sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=PIPELINE_DEVICE,
                max_length=512,
                truncation=True,
            )
        except Exception:
            return pipeline(
                task="sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
                max_length=512,
                truncation=True,
            )

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Analyze a single text, returning `SentimentResult` or None for empty input."""
        if not text or not text.strip():
            return None
        result = self.pipeline(text.strip())[0]
        return SentimentResult(label=result["label"], score=float(result["score"]))

    def analyze_batch(self, texts: Iterable[str]) -> List[Optional[SentimentResult]]:
        """Analyze a batch of texts; returns list aligned to input order."""
        batch = [t.strip() if t is not None else "" for t in texts]
        if not batch:
            return []
        # Run pipeline only on non-empty entries to preserve indices
        indices = [i for i, t in enumerate(batch) if t]
        outputs = [None] * len(batch)
        if indices:
            padded_inputs = [batch[i] for i in indices]
            preds = self.pipeline(padded_inputs)
            for i, pred in zip(indices, preds):
                outputs[i] = SentimentResult(label=pred["label"], score=float(pred["score"]))
        return outputs

# Convenience free functions
_detector_singleton: Optional[SentimentDetector] = None

def _get_singleton() -> SentimentDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = SentimentDetector()
    return _detector_singleton

def get_sentiment(text: str) -> Optional[SentimentResult]:
    return _get_singleton().analyze(text)

def get_sentiments(texts: Iterable[str]) -> List[Optional[SentimentResult]]:
    return _get_singleton().analyze_batch(texts)

# Minimal CLI for quick checks
def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment-only detector (SST-2)")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="Path to a file with one text per line")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    det = _get_singleton()
    if args.text:
        res = det.analyze(args.text)
        if args.json:
            print(res.to_json() if res else json.dumps(None))
        else:
            print(res)
        return 0

    if args.file:
        path = Path(args.file)
        lines = path.read_text(encoding="utf-8").splitlines()
        results = det.analyze_batch(lines)
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
