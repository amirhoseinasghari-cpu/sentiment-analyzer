from __future__ import annotations

from pathlib import Path

from app.utils import compute_accuracy_placeholder, load_config
from models.sentiment_model import SentimentAnalyzer


def test_config_loading() -> None:
  config_path = Path("config.yaml")
  config = load_config(config_path)
  assert "app" in config
  assert "model" in config


def test_accuracy_placeholder() -> None:
  acc = compute_accuracy_placeholder()
  assert isinstance(acc, float)
  assert 0.0 < acc <= 100.0


def test_sentiment_analyzer_init() -> None:
  """Ensure that the SentimentAnalyzer can be constructed without errors.

  We do not load the heavy model weights here to keep the test fast;
  this test simply verifies that the object can be instantiated.
  """
  analyzer = SentimentAnalyzer(model_name="HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
  assert analyzer.model_name.endswith("snappfood")

