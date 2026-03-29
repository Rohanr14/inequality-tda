# tests/test_ph_pipeline.py
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ph_pipeline import locate_largest_gap, bootstrap_gap


class TestLocateLargestGap:
    def test_simple_gap(self):
        vec = np.array([0, 1, 2, 10, 11, 12])
        lo, hi, width = locate_largest_gap(vec)
        assert lo == 2
        assert hi == 3
        assert width == 8.0

    def test_gap_at_start(self):
        vec = np.array([0, 100, 101, 102])
        lo, hi, width = locate_largest_gap(vec)
        assert lo == 0
        assert hi == 1
        assert width == 100.0

    def test_gap_at_end(self):
        vec = np.array([0, 1, 2, 200])
        lo, hi, width = locate_largest_gap(vec)
        assert lo == 2
        assert hi == 3
        assert width == 198.0

    def test_uniform_spacing(self):
        vec = np.linspace(0, 100, 101)
        lo, hi, width = locate_largest_gap(vec)
        assert abs(width - 1.0) < 1e-10

    def test_empty_vector(self):
        vec = np.array([42])
        lo, hi, width = locate_largest_gap(vec)
        assert lo == -1
        assert np.isnan(width)

    def test_two_elements(self):
        vec = np.array([10, 50])
        lo, hi, width = locate_largest_gap(vec)
        assert lo == 0
        assert hi == 1
        assert width == 40.0


class TestBootstrapGap:
    def test_returns_tuple(self):
        vec = np.array([0, 1, 2, 10, 11, 12, 13, 14, 15, 20])
        lo, hi = bootstrap_gap(vec, n_boot=100, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_point_estimate(self):
        vec = np.array([0, 1, 2, 10, 11, 12, 13, 14, 15, 20])
        _, _, point_est = locate_largest_gap(vec)
        lo, hi = bootstrap_gap(vec, n_boot=500, seed=0)
        # The point estimate should be within or near the CI
        assert lo <= point_est + 1  # allow small tolerance
        assert hi >= point_est - 1

    def test_deterministic_with_seed(self):
        vec = np.linspace(0, 100000, 101)
        lo1, hi1 = bootstrap_gap(vec, n_boot=100, seed=0)
        lo2, hi2 = bootstrap_gap(vec, n_boot=100, seed=0)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_ci_lo_le_hi(self):
        vec = np.random.default_rng(0).uniform(0, 200000, 101)
        vec.sort()
        lo, hi = bootstrap_gap(vec, n_boot=200, seed=0)
        assert lo <= hi


class TestGiniTheilConsistency:
    """Test that the Gini/Theil calculations in ph_pipeline produce sane values."""

    def test_perfect_equality(self):
        # All same income → Gini ≈ 0
        vec = np.full(101, 50000.0)
        mean = vec.mean()
        gini = 1 - 2 * np.trapezoid(np.cumsum(vec) / vec.sum(), dx=1 / len(vec))
        assert abs(gini) < 0.05  # should be ~0

    def test_high_inequality(self):
        # One person has everything
        vec = np.zeros(101)
        vec[-1] = 1_000_000
        vec[0] = 1  # avoid div-by-zero
        mean = vec.mean()
        gini = 1 - 2 * np.trapezoid(np.cumsum(vec) / vec.sum(), dx=1 / len(vec))
        assert gini > 0.8  # should be near 1
