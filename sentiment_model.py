from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline


@dataclass
class SentimentResult:
  """Container for a single sentiment prediction."""

  label: str
  score: float
  latency_ms: float


class SentimentAnalyzer:
  """Production-ready wrapper around a Persian BERT sentiment model.

  The underlying model is `HooshvareLab/bert-fa-base-uncased-sentiment-snappfood`,
  which is a binary sentiment classifier trained on SnappFood reviews with
  `HAPPY` (positive) and `SAD` (negative) labels. Since the user-facing
  dashboard expects three classes (positive/neutral/negative), this wrapper
  derives a **pseudo-neutral** class based on score thresholds.
  """

  def __init__(
    self,
    model_name: str = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood",
    device: Optional[int] = None,
    positive_threshold: float = 0.65,
    negative_threshold: float = 0.65,
  ) -> None:
    """Initialize the sentiment analyzer and load the model.

    Args:
      model_name: Hugging Face model identifier.
      device: Torch device index; if None, automatically selects GPU if available.
      positive_threshold: Minimum positive probability to be considered positive.
      negative_threshold: Minimum negative probability to be considered negative.
    """
    self.model_name = model_name
    self.positive_threshold = positive_threshold
    self.negative_threshold = negative_threshold

    if device is None:
      self.device = 0 if torch.cuda.is_available() else -1
    else:
      self.device = device

    self._pipeline: Optional[TextClassificationPipeline] = None

  @property
  def pipeline(self) -> TextClassificationPipeline:
    """Lazy-load and cache the Hugging Face pipeline."""
    if self._pipeline is None:
      tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
      self._pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=self.device,
        return_all_scores=True,
      )
    return self._pipeline

  @staticmethod
  def _map_label_to_persian(label: str) -> str:
    """Map raw model labels (e.g. HAPPY/SAD) to Persian labels."""
    upper = label.upper()
    if "HAPPY" in upper or "POS" in upper:
      return "مثبت"
    if "SAD" in upper or "NEG" in upper:
      return "منفی"
    return "خنثی"

  def _derive_three_class(
    self,
    scores: List[Dict[str, Any]],
  ) -> Tuple[str, float]:
    """Convert binary scores to three sentiment classes.

    If neither class crosses its respective threshold, we treat the
    sentiment as neutral.
    """
    if len(scores) < 2:
      # Fallback: just return the top score as-is
      best = max(scores, key=lambda x: float(x["score"]))
      return self._map_label_to_persian(best["label"]), float(best["score"])

    # Assume two classes (positive/negative)
    s0, s1 = scores[0], scores[1]
    label0 = s0["label"].upper()
    p0 = float(s0["score"])
    p1 = float(s1["score"])

    # Identify positive and negative based on label keywords
    if "HAPPY" in label0 or "POS" in label0:
      pos_prob, neg_prob = p0, p1
    else:
      pos_prob, neg_prob = p1, p0

    if pos_prob >= self.positive_threshold and pos_prob >= neg_prob:
      return "مثبت", pos_prob
    if neg_prob >= self.negative_threshold and neg_prob > pos_prob:
      return "منفی", neg_prob
    # Neither strong enough -> neutral
    return "خنثی", max(pos_prob, neg_prob)

  def predict(self, text: str, max_length: int = 512) -> SentimentResult:
    """Run sentiment analysis on a single text.

    Args:
      text: Input Persian text.
      max_length: Maximum sequence length for the model.

    Returns:
      SentimentResult with predicted label, confidence score, and latency.
    """
    if not text or not text.strip():
      raise ValueError("متن ورودی نمی‌تواند خالی باشد.")

    start = time.perf_counter()
    outputs = self.pipeline(
      text,
      truncation=True,
      max_length=max_length,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    # When return_all_scores=True and a single text is passed, we get a list[dict]
    label, score = self._derive_three_class(outputs)

    return SentimentResult(label=label, score=score, latency_ms=latency_ms)

  def batch_predict(
    self,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 32,
  ) -> List[SentimentResult]:
    """Run batch sentiment analysis on multiple texts.

    Args:
      texts: List of input texts.
      max_length: Maximum sequence length for the model.
      batch_size: Batch size for efficient inference.

    Returns:
      List of SentimentResult objects corresponding to each input text.
    """
    clean_texts = [t for t in texts if t and t.strip()]
    if not clean_texts:
      raise ValueError("هیچ متن معتبری برای تحلیل ارسال نشده است.")

    results: List[SentimentResult] = []
    for i in range(0, len(clean_texts), batch_size):
      batch = clean_texts[i : i + batch_size]
      start = time.perf_counter()
      outputs = self.pipeline(
        batch,
        truncation=True,
        max_length=max_length,
      )
      batch_latency_ms = (time.perf_counter() - start) * 1000.0
      per_item_latency = batch_latency_ms / max(len(outputs), 1)

      for out in outputs:
        label, score = self._derive_three_class(out)
        results.append(
          SentimentResult(
            label=label,
            score=score,
            latency_ms=per_item_latency,
          )
        )

    return results

