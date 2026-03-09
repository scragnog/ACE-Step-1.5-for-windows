"""Unit tests for the scheduler registry."""

import unittest

from acestep.core.generation.schedulers import (
    SCHEDULERS,
    SCHEDULER_INFO,
    VALID_SCHEDULERS,
    get_scheduler,
    linear_schedule,
    ddim_uniform_schedule,
    sgm_uniform_schedule,
    bong_tangent_schedule,
    linear_quadratic_schedule,
)


class TestSchedulerRegistry(unittest.TestCase):
    """Tests for the registry lookup and metadata."""

    def test_all_schedulers_have_info(self):
        for name in SCHEDULERS:
            self.assertIn(name, SCHEDULER_INFO, f"Missing SCHEDULER_INFO for '{name}'")

    def test_valid_schedulers_matches_registry(self):
        self.assertEqual(VALID_SCHEDULERS, set(SCHEDULERS.keys()))

    def test_get_scheduler_returns_callable(self):
        for name in SCHEDULERS:
            fn = get_scheduler(name)
            self.assertTrue(callable(fn))

    def test_get_scheduler_case_insensitive(self):
        fn = get_scheduler("LINEAR")
        self.assertEqual(fn, linear_schedule)

    def test_get_scheduler_strips_whitespace(self):
        fn = get_scheduler("  linear  ")
        self.assertEqual(fn, linear_schedule)

    def test_get_scheduler_invalid_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_scheduler("nonexistent_scheduler")
        self.assertIn("nonexistent_scheduler", str(ctx.exception))


class TestScheduleOutput(unittest.TestCase):
    """Tests for individual schedule functions."""

    ALL_SCHEDULERS = [
        ("linear", linear_schedule),
        ("ddim_uniform", ddim_uniform_schedule),
        ("sgm_uniform", sgm_uniform_schedule),
        ("bong_tangent", bong_tangent_schedule),
        ("linear_quadratic", linear_quadratic_schedule),
    ]

    def test_correct_length(self):
        for name, fn in self.ALL_SCHEDULERS:
            for n in [5, 8, 20, 50]:
                result = fn(n, shift=1.0)
                self.assertEqual(len(result), n, f"{name} with n={n}: got {len(result)}")

    def test_descending(self):
        for name, fn in self.ALL_SCHEDULERS:
            result = fn(20, shift=1.0)
            for i in range(len(result) - 1):
                self.assertGreaterEqual(
                    result[i], result[i + 1],
                    f"{name} not descending at index {i}: {result[i]} < {result[i + 1]}",
                )

    def test_values_in_range(self):
        for name, fn in self.ALL_SCHEDULERS:
            result = fn(20, shift=1.0)
            for i, t in enumerate(result):
                self.assertGreater(t, 0.0, f"{name}[{i}] = {t} <= 0")
                self.assertLessEqual(t, 1.0, f"{name}[{i}] = {t} > 1")

    def test_first_value_near_one(self):
        """The first timestep should be close to 1.0 (full noise)."""
        for name, fn in self.ALL_SCHEDULERS:
            result = fn(20, shift=1.0)
            self.assertGreater(result[0], 0.8, f"{name} first value too low: {result[0]}")

    def test_shift_changes_values(self):
        """Non-unity shift should produce different timesteps."""
        for name, fn in self.ALL_SCHEDULERS:
            no_shift = fn(10, shift=1.0)
            with_shift = fn(10, shift=3.0)
            self.assertNotEqual(no_shift, with_shift, f"{name}: shift=3.0 had no effect")

    def test_linear_matches_legacy(self):
        """Linear schedule with shift=1.0 should match torch.linspace(1,0,N+1)[:-1]."""
        n = 10
        result = linear_schedule(n, shift=1.0)
        expected = [1.0 - i / n for i in range(n)]
        for i, (r, e) in enumerate(zip(result, expected)):
            self.assertAlmostEqual(r, e, places=10, msg=f"linear[{i}]: {r} != {e}")

    def test_single_step(self):
        """Edge case: 1 step should still work."""
        for name, fn in self.ALL_SCHEDULERS:
            result = fn(1, shift=1.0)
            self.assertEqual(len(result), 1, f"{name}: single step returned {len(result)}")
            self.assertGreater(result[0], 0.0)
            self.assertLessEqual(result[0], 1.0)


if __name__ == "__main__":
    unittest.main()
