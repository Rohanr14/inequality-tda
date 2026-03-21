# tests/test_data_loader.py
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import (
    _estimate_pareto_alpha,
    _calculate_percentiles_from_brackets,
    ACS_BRACKET_VARS,
    ACS_BRACKET_UPPER_BOUNDS,
    load_cpi,
)


# ---------------------------------------------------------------------------
# Pareto alpha estimation
# ---------------------------------------------------------------------------

class TestParetoAlpha:
    def test_known_alpha(self):
        """With a true Pareto(α=2), survival fractions should recover α≈2."""
        x_m_lo, x_m_hi = 150_000, 200_000
        alpha_true = 2.0
        f_lo = (x_m_lo / x_m_hi) ** alpha_true  # survival at lo / survival at hi
        # f_lo = fraction above lo, f_hi = fraction above hi
        # (x_lo/x_hi)^α = f_hi/f_lo ... rearranging:
        f_hi = 0.05  # 5% above 200k
        f_lo = f_hi * (x_m_hi / x_m_lo) ** alpha_true
        alpha_est = _estimate_pareto_alpha(f_lo, f_hi, x_m_lo, x_m_hi)
        assert abs(alpha_est - alpha_true) < 0.01

    def test_fallback_on_zero_fraction(self):
        alpha = _estimate_pareto_alpha(0.0, 0.0, 150_000, 200_000)
        assert alpha == 1.5  # default

    def test_fallback_on_impossible_fractions(self):
        # More people above 200k than above 150k → impossible
        alpha = _estimate_pareto_alpha(0.05, 0.10, 150_000, 200_000)
        assert alpha == 1.5

    def test_clamp_extreme_alpha(self):
        # Very extreme fractions that produce α outside [1.2, 5.0]
        alpha = _estimate_pareto_alpha(0.999, 0.998, 150_000, 200_000)
        assert alpha == 1.5  # should fall back


# ---------------------------------------------------------------------------
# Percentile interpolation with Pareto tail
# ---------------------------------------------------------------------------

def _make_fake_row(counts: list) -> pd.Series:
    """Create a fake ACS row with given bracket counts."""
    data = {'NAME': 'TestState'}
    for var, count in zip(ACS_BRACKET_VARS, counts):
        data[var] = count
    return pd.Series(data)


class TestPercentileInterpolation:
    def test_output_length(self):
        # Uniform distribution across brackets
        counts = [100] * 16
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is not None
        assert len(result) == 101

    def test_monotonically_increasing(self):
        counts = [100] * 16
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is not None
        assert np.all(np.diff(result) >= 0), "Percentiles must be non-decreasing"

    def test_pareto_tail_extends_beyond_200k(self):
        """With households in the $200k+ bracket, tail should exceed $200k."""
        counts = [100] * 15 + [50]  # 50 households above $200k
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is not None
        assert result[-1] > 200_000, (
            f"Pareto tail should push 100th percentile above $200k, got {result[-1]:,.0f}"
        )

    def test_no_flat_plateau_at_top(self):
        """The old bug: top percentiles were all capped at $200k. Now they shouldn't be."""
        counts = [200, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 60, 40, 30, 20]
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is not None
        top_diffs = np.diff(result[-10:])
        # At least some of the top percentile gaps should be non-trivial
        assert np.any(top_diffs > 100), (
            "Top percentiles should not be flat — Pareto tail should create variation"
        )

    def test_handles_zero_top_bracket(self):
        """If no one earns $200k+, tail should still work (no Pareto needed)."""
        counts = [100] * 15 + [0]
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is not None
        assert result[-1] <= 200_000

    def test_handles_all_nan(self):
        counts = [np.nan] * 16
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is None

    def test_handles_all_zero(self):
        counts = [0] * 16
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 101)
        assert result is None

    def test_201_points(self):
        counts = [100] * 16
        row = _make_fake_row(counts)
        result = _calculate_percentiles_from_brackets(row, 201)
        assert result is not None
        assert len(result) == 201


# ---------------------------------------------------------------------------
# CPI loader
# ---------------------------------------------------------------------------

class TestCPI:
    def test_load_cpi(self):
        cpi_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'cpi-u_annual.csv')
        df = load_cpi(cpi_path)
        assert 'year' in df.columns
        assert 'deflator' in df.columns
        # 2024 deflator should be 1.0 (base year)
        d2024 = df.loc[df['year'] == 2024, 'deflator'].values[0]
        assert abs(d2024 - 1.0) < 0.001
        # Older years should have deflator > 1 (inflating old dollars)
        d2010 = df.loc[df['year'] == 2010, 'deflator'].values[0]
        assert d2010 > 1.0
