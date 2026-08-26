"""Unit tests for Stage 4 deterministic mastery scoring calculation."""

from datetime import datetime, timedelta, timezone
import unittest

from src.study.scoring import (
    RECENCY_HALF_LIFE_DAYS,
    WEIGHT_CONFIDENCE,
    WEIGHT_CORRECTNESS,
    WEIGHT_RECENCY,
    chapter_mastery_score,
)


class TestScoring(unittest.TestCase):
    def test_empty_attempts_returns_none(self) -> None:
        """Confirm chapter_mastery_score returns None on empty attempts."""
        self.assertIsNone(chapter_mastery_score([]))

    def test_all_correct_max_confidence(self) -> None:
        """All correct with maximum confidence yields 100.0 score."""
        now = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
        attempts = [
            {
                "self_correct": "correct",
                "confidence": 5,
                "attempted_at": now.isoformat(),
            }
            for _ in range(5)
        ]
        result = chapter_mastery_score(attempts, current_time=now)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["mastery_score"], 100.0)
        self.assertEqual(result["total_attempts"], 5)
        self.assertEqual(result["correctness_rate"], 1.0)
        self.assertEqual(result["mean_confidence"], 1.0)
        self.assertEqual(result["recency_weighted_correctness"], 1.0)

    def test_all_incorrect_min_confidence(self) -> None:
        """All incorrect with minimum confidence yields 0.0 score."""
        now = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
        attempts = [
            {
                "self_correct": "incorrect",
                "confidence": 1,
                "attempted_at": now.isoformat(),
            }
            for _ in range(3)
        ]
        result = chapter_mastery_score(attempts, current_time=now)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["mastery_score"], 0.0)
        self.assertEqual(result["total_attempts"], 3)
        self.assertEqual(result["correctness_rate"], 0.0)
        self.assertEqual(result["mean_confidence"], 0.0)
        self.assertEqual(result["recency_weighted_correctness"], 0.0)

    def test_mixed_deterministic_calculation(self) -> None:
        """Verify hand-calculated composite blend for mixed attempts."""
        now = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
        # Attempt 1: correct (1.0), conf 5 (1.0), attempted now (dt=0 -> weight=1.0)
        # Attempt 2: partial (0.5), conf 3 (0.5), attempted now (dt=0 -> weight=1.0)
        # Attempt 3: incorrect (0.0), conf 1 (0.0), attempted now (dt=0 -> weight=1.0)
        attempts = [
            {"self_correct": "correct", "confidence": 5, "attempted_at": now.isoformat()},
            {"self_correct": "partial", "confidence": 3, "attempted_at": now.isoformat()},
            {"self_correct": "incorrect", "confidence": 1, "attempted_at": now.isoformat()},
        ]
        # correctness_rate = (1.0 + 0.5 + 0.0) / 3 = 0.5
        # mean_confidence = (1.0 + 0.5 + 0.0) / 3 = 0.5
        # recency_weighted_correctness = (1.0*1.0 + 1.0*0.5 + 1.0*0.0) / 3.0 = 0.5
        # composite = 0.60 * 0.5 + 0.20 * 0.5 + 0.20 * 0.5 = 0.5 -> 50.0%
        result = chapter_mastery_score(attempts, current_time=now)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["mastery_score"], 50.0)
        self.assertEqual(result["correctness_rate"], 0.5)
        self.assertEqual(result["mean_confidence"], 0.5)
        self.assertEqual(result["recency_weighted_correctness"], 0.5)

    def test_recency_decay_weighting_effect(self) -> None:
        """Recent correct answers pull the score higher than recent incorrect answers."""
        now = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(days=28)  # 2 half-lives ago (weight = 0.25)

        # Set A: old incorrect (wt 0.25), new correct (wt 1.0)
        attempts_improving = [
            {"self_correct": "incorrect", "confidence": 3, "attempted_at": old_time.isoformat()},
            {"self_correct": "correct", "confidence": 3, "attempted_at": now.isoformat()},
        ]

        # Set B: old correct (wt 0.25), new incorrect (wt 1.0)
        attempts_declining = [
            {"self_correct": "correct", "confidence": 3, "attempted_at": old_time.isoformat()},
            {"self_correct": "incorrect", "confidence": 3, "attempted_at": now.isoformat()},
        ]

        result_improving = chapter_mastery_score(attempts_improving, current_time=now)
        result_declining = chapter_mastery_score(attempts_declining, current_time=now)

        assert result_improving is not None
        assert result_declining is not None

        # Both have identical unweighted correctness (50%) and confidence (50%)
        self.assertEqual(result_improving["correctness_rate"], result_declining["correctness_rate"])
        self.assertEqual(result_improving["mean_confidence"], result_declining["mean_confidence"])

        # But improving set has higher recency-weighted correctness and higher overall mastery
        self.assertGreater(
            result_improving["recency_weighted_correctness"],
            result_declining["recency_weighted_correctness"],
        )
        self.assertGreater(
            result_improving["mastery_score"],
            result_declining["mastery_score"],
        )

    def test_recency_delta_t_calculated_relative_to_current_time(self) -> None:
        """Confirm elapsed time delta is evaluated against current_time at computation."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        attempt = {"self_correct": "correct", "confidence": 5, "attempted_at": t0.isoformat()}

        # Evaluated at t0: dt = 0 -> weight = 1.0, score = 100.0
        score_at_t0 = chapter_mastery_score([attempt], current_time=t0)
        assert score_at_t0 is not None
        self.assertEqual(score_at_t0["mastery_score"], 100.0)

        # Evaluated 14 days later (1 half-life later) with no new attempts
        t_later = t0 + timedelta(days=RECENCY_HALF_LIFE_DAYS)
        score_later = chapter_mastery_score([attempt], current_time=t_later)
        assert score_later is not None
        # With single attempt, recency_weighted_correctness is (weight * 1.0) / weight = 1.0
        self.assertEqual(score_later["recency_weighted_correctness"], 1.0)


if __name__ == "__main__":
    unittest.main()
