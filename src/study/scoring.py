"""Deterministic mastery scoring calculation from quiz attempts."""

from datetime import datetime, timezone
import math
import sqlite3
from typing import Any, Sequence


# Placeholder tuning constants pending empirical testing
WEIGHT_CORRECTNESS = 0.60
WEIGHT_CONFIDENCE = 0.20
WEIGHT_RECENCY = 0.20

CORRECTNESS_WEIGHTS: dict[str, float] = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
}

# Exponential decay half-life in days (e.g. attempt 14 days ago has half the recency weight of today)
RECENCY_HALF_LIFE_DAYS = 14.0


def parse_timestamp(ts_val: Any) -> datetime:
    """Parse an attempted_at SQLite or ISO timestamp into a timezone-aware UTC datetime."""
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val.astimezone(timezone.utc)

    if not isinstance(ts_val, str) or not ts_val.strip():
        return datetime.now(timezone.utc)

    ts_str = ts_val.strip()
    # Normalize common SQLite format 'YYYY-MM-DD HH:MM:SS' to ISO 'YYYY-MM-DDTHH:MM:SS'
    if " " in ts_str and "T" not in ts_str:
        ts_str = ts_str.replace(" ", "T")

    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def chapter_mastery_score(
    attempts: Sequence[sqlite3.Row | dict[str, Any]],
    *,
    current_time: datetime | None = None,
) -> dict[str, Any] | None:
    """Calculate a deterministic chapter mastery score from quiz attempts.

    The score is a weighted combination of:
      1. correctness_rate (60%): Unweighted average correctness (correct=1.0, partial=0.5, incorrect=0.0).
      2. mean_confidence (20%): Average student confidence mapped from 1-5 to 0.0-1.0.
      3. recency_weighted_correctness (20%): Recency-weighted average of correctness across all attempts,
         where an attempt's weight decays exponentially based on elapsed days (Δt_i) from attempted_at
         to the current time at score computation (current_time), NOT relative to the newest attempt.

    Formula:
      Δt_i = max(0, (now - attempted_at_i) in days)
      weight_i = exp(-ln(2) * Δt_i / RECENCY_HALF_LIFE_DAYS) = 2^(-Δt_i / RECENCY_HALF_LIFE_DAYS)
      recency_weighted_correctness = sum(weight_i * correctness_i) / sum(weight_i)
      mastery_score = round(100 * (W_corr * correctness_rate + W_conf * mean_confidence + W_rec * recency_weighted_correctness), 1)

    Args:
        attempts: Sequence of sqlite3.Row or dict representing quiz attempt records.
        current_time: Optional datetime reference for score computation (defaults to UTC now).

    Returns:
        A dictionary with mastery breakdown, or None if attempts is empty:
        {
            "mastery_score": float,                     # 0.0 to 100.0
            "total_attempts": int,
            "correctness_rate": float,                  # 0.0 to 1.0
            "mean_confidence": float,                   # 0.0 to 1.0 (normalized)
            "recency_weighted_correctness": float,      # 0.0 to 1.0 (decay-weighted average)
        }
    """
    if not attempts:
        return None

    now = current_time.astimezone(timezone.utc) if current_time is not None else datetime.now(timezone.utc)
    decay_constant = math.log(2.0) / RECENCY_HALF_LIFE_DAYS

    correctness_scores: list[float] = []
    confidence_scores: list[float] = []
    recency_weighted_sum = 0.0
    recency_weight_total = 0.0

    for attempt in attempts:
        # 1. Correctness score mapping
        self_correct = str(attempt["self_correct"]).lower() if isinstance(attempt, (sqlite3.Row, dict)) else "incorrect"
        corr_val = CORRECTNESS_WEIGHTS.get(self_correct, 0.0)
        correctness_scores.append(corr_val)

        # 2. Normalized confidence rating (1-5 mapped to 0.0-1.0)
        raw_conf = attempt["confidence"] if isinstance(attempt, (sqlite3.Row, dict)) else 3
        try:
            conf_int = int(raw_conf)
        except (ValueError, TypeError):
            conf_int = 3
        # Clamp to 1..5
        conf_int = max(1, min(5, conf_int))
        norm_conf = (conf_int - 1) / 4.0
        confidence_scores.append(norm_conf)

        # 3. Recency decay weighting relative to current computation time
        ts = parse_timestamp(attempt["attempted_at"] if "attempted_at" in attempt.keys() else None)
        elapsed_seconds = max(0.0, (now - ts).total_seconds())
        elapsed_days = elapsed_seconds / 86400.0

        weight = math.exp(-decay_constant * elapsed_days)
        recency_weighted_sum += weight * corr_val
        recency_weight_total += weight

    correctness_rate = sum(correctness_scores) / len(correctness_scores)
    mean_confidence = sum(confidence_scores) / len(confidence_scores)
    recency_weighted_correctness = (
        recency_weighted_sum / recency_weight_total if recency_weight_total > 0.0 else correctness_rate
    )

    composite = (
        WEIGHT_CORRECTNESS * correctness_rate
        + WEIGHT_CONFIDENCE * mean_confidence
        + WEIGHT_RECENCY * recency_weighted_correctness
    )
    mastery_score = round(composite * 100.0, 1)

    return {
        "mastery_score": mastery_score,
        "total_attempts": len(attempts),
        "correctness_rate": round(correctness_rate, 4),
        "mean_confidence": round(mean_confidence, 4),
        "recency_weighted_correctness": round(recency_weighted_correctness, 4),
    }
