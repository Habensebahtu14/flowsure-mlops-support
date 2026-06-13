import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Logger -> stdout ─────────────────────────────────────────────────────────────
# A single module-level logger that writes INFO+ to stdout. Configured here so
# the rest of the app just imports the functions below.
logger = logging.getLogger("flowsure.monitoring")

if not logger.handlers:  # avoid duplicate handlers on Streamlit reruns
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()  # defaults to stderr/stdout stream
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(_handler)
    logger.propagate = False  # don't double-log through the root logger

# ── Structured log file ──────────────────────────────────────────────────────────
# Anchored to this file's directory so it resolves to /app/logs in the container
# (mountable as a volume) regardless of the current working directory.
LOG_DIR = Path(__file__).resolve().parent / "logs"
PREDICTIONS_FILE = LOG_DIR / "predictions.jsonl"


def log_prediction(result: dict, text_length: int) -> None:
    """Log a single prediction to both stdout and predictions.jsonl.

    `result` is the dict returned by ``IntentClassifier.predict``. Raw text is
    intentionally not logged -- only `text_length` is recorded.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_length": text_length,
        "intent": result.get("intent"),
        "category": result.get("category"),
        "priority": result.get("priority"),
        "confidence": result.get("confidence"),
        "inference_time_ms": result.get("inference_time_ms"),
    }

    # Human-readable system log line.
    logger.info(
        "Prediction: intent=%s category=%s priority=%s confidence=%.2f (%sms)",
        record["intent"],
        record["category"],
        record["priority"],
        record["confidence"] if record["confidence"] is not None else 0.0,
        record["inference_time_ms"],
    )

    # Structured JSONL line. A logging failure must never crash the app.
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Could not write prediction log: %s", exc)


def log_event(message: str, level: str = "info") -> None:
    """Log a lifecycle/error event (app start, model loaded, errors)."""
    log_fn = {
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }.get(level.lower(), logger.info)
    log_fn(message)
