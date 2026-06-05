"""
Unit tests for IntentClassifier (app/classifier.py).

Loads the real ONNX model (no mocks) to verify that predictions are
structurally valid, that edge-case inputs are rejected, and that the
model file meets the size budget required for on-device deployment.
"""

import pytest
from pathlib import Path

# Resolve model paths relative to this file so the tests work both
# locally and in GitHub Actions (where the working directory may differ).
_MODELS_DIR = Path(__file__).parent.parent / "app" / "models"
_MODEL_PATH = _MODELS_DIR / "baseline_model.onnx"
_MAPPING_PATH = _MODELS_DIR / "intent_mapping.json"

# Import after resolving paths so import errors surface clearly.
from app.classifier import IntentClassifier  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture — load the classifier once for the entire test session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def classifier():
    """Single IntentClassifier instance shared across all tests."""
    return IntentClassifier(str(_MODEL_PATH), str(_MAPPING_PATH))


# ---------------------------------------------------------------------------
# Test 1 — model loads without errors
# ---------------------------------------------------------------------------

def test_model_loads():
    # Constructing a fresh instance verifies that both model files are
    # readable and that onnxruntime accepts the graph.
    clf = IntentClassifier(str(_MODEL_PATH), str(_MAPPING_PATH))
    assert clf is not None
    assert clf.num_intents > 0


# ---------------------------------------------------------------------------
# Test 2 — predict() returns a dict with all expected keys
# ---------------------------------------------------------------------------

def test_predict_returns_expected_keys(classifier):
    result = classifier.predict("I want to cancel my order")
    expected_keys = {"intent", "category", "priority", "confidence", "top_3", "inference_time_ms"}
    assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# Test 3 — confidence is a valid probability
# ---------------------------------------------------------------------------

def test_confidence_is_valid_probability(classifier):
    result = classifier.predict("My payment failed, please help")
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — priority is one of the three allowed values
# ---------------------------------------------------------------------------

def test_priority_is_valid_value(classifier):
    result = classifier.predict("I cannot log in to my account")
    assert result["priority"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Test 5 — top_3 contains exactly 3 items, each with intent + confidence
# ---------------------------------------------------------------------------

def test_top3_has_correct_shape(classifier):
    result = classifier.predict("Where is my refund?")
    top3 = result["top_3"]
    assert isinstance(top3, list)
    assert len(top3) == 3
    for item in top3:
        # Each entry must be a dict with exactly these two keys.
        assert "intent" in item
        assert "confidence" in item


# ---------------------------------------------------------------------------
# Test 6 — empty or whitespace-only input raises ValueError
# ---------------------------------------------------------------------------

def test_empty_string_raises_value_error(classifier):
    with pytest.raises(ValueError):
        classifier.predict("")


def test_whitespace_only_raises_value_error(classifier):
    with pytest.raises(ValueError):
        classifier.predict("   ")


# ---------------------------------------------------------------------------
# Test 7 — model file is within the edge-deployment size budget (< 50 MB)
# ---------------------------------------------------------------------------

def test_model_size_within_edge_budget():
    # Edge devices (Raspberry Pi, embedded hardware) have strict storage
    # constraints. 50 MB is the upper bound claimed in the project docs.
    size_mb = _MODEL_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 50, f"Model is {size_mb:.2f} MB — exceeds the 50 MB edge budget"


# ---------------------------------------------------------------------------
# Test 8 — predict_batch() returns one result per input text
# ---------------------------------------------------------------------------

def test_predict_batch_returns_same_length(classifier):
    # Note: predict_batch silently skips empty strings, so we use only
    # non-empty texts here to guarantee the lengths match.
    texts = [
        "I want to cancel my subscription",
        "How do I reset my password?",
        "I never received my order",
    ]
    results = classifier.predict_batch(texts)
    assert len(results) == len(texts)
    # Each item must still be a valid prediction dict.
    for result in results:
        assert "intent" in result
        assert "confidence" in result
