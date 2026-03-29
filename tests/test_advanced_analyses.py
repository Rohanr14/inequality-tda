# tests/test_advanced_analyses.py
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGenuineTDA:
    def test_compute_persistence(self):
        from analysis.genuine_tda import _compute_persistence, _persistence_to_array
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        persistence = _compute_persistence(pts, max_dim=1)
        h0 = _persistence_to_array(persistence, 0)
        h1 = _persistence_to_array(persistence, 1)
        # 5 points → should have H₀ features (connected components merge)
        assert len(h0) > 0
        # All births should be ≤ deaths
        assert np.all(h0[:, 0] <= h0[:, 1])

    def test_persistence_empty_h1(self):
        """Collinear points should produce no H₁ features (no loops)."""
        from analysis.genuine_tda import _compute_persistence, _persistence_to_array
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        persistence = _compute_persistence(pts, max_dim=1)
        h1 = _persistence_to_array(persistence, 1)
        assert len(h1) == 0

    def test_persistence_square_h1(self):
        """A square arrangement should produce at least one H₁ feature (a loop)."""
        from analysis.genuine_tda import _compute_persistence, _persistence_to_array
        # Square with large separation → clear loop
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        persistence = _compute_persistence(pts, max_dim=1)
        h1 = _persistence_to_array(persistence, 1)
        assert len(h1) >= 1


class TestWasserstein:
    def test_identical_distributions(self):
        from analysis.wasserstein_analysis import _wasserstein_1d
        u = np.linspace(0, 100000, 101)
        assert _wasserstein_1d(u, u) == 0.0

    def test_shifted_distribution(self):
        from analysis.wasserstein_analysis import _wasserstein_1d
        u = np.linspace(0, 100000, 101)
        v = u + 5000
        d = _wasserstein_1d(u, v)
        assert abs(d - 5000) < 1.0

    def test_symmetry(self):
        from analysis.wasserstein_analysis import _wasserstein_1d
        rng = np.random.default_rng(42)
        u = np.sort(rng.uniform(0, 200000, 101))
        v = np.sort(rng.uniform(0, 200000, 101))
        assert abs(_wasserstein_1d(u, v) - _wasserstein_1d(v, u)) < 1e-10


class TestMapperGraph:
    def test_region_mapping(self):
        from analysis.mapper_graph import STATE_TO_REGION
        assert STATE_TO_REGION["California"] == "West"
        assert STATE_TO_REGION["New York"] == "Northeast"
        assert STATE_TO_REGION["Texas"] == "South"
        assert STATE_TO_REGION["Ohio"] == "Midwest"
        assert len(STATE_TO_REGION) == 51  # 50 states + DC


class TestMobilityData:
    def test_mobility_csv_loads(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'chetty_mobility_by_state.csv')
        df = pd.read_csv(csv_path)
        assert len(df) == 51  # 50 states + DC
        assert 'state' in df.columns
        assert 'absolute_upward_mobility' in df.columns
        # Mobility values should be in plausible range (roughly 35-50)
        assert df['absolute_upward_mobility'].min() > 30
        assert df['absolute_upward_mobility'].max() < 55

    def test_all_states_present(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'chetty_mobility_by_state.csv')
        df = pd.read_csv(csv_path)
        assert "District of Columbia" in df['state'].values
        assert "California" in df['state'].values
        assert "Wyoming" in df['state'].values


class TestTemporalHoldout:
    def test_period_gap(self):
        from analysis.temporal_holdout import _period_gap
        ts = pd.DataFrame({
            "year": [2010, 2011, 2018, 2019] * 2,
            "state": ["A"] * 4 + ["B"] * 4,
            "acs_longest_h0_lifespan_real": [100, 200, 300, 400, 500, 600, 700, 800],
        })
        early = _period_gap(ts, 2010, 2016)
        assert len(early) == 2
        assert early.loc[early["state"] == "A", "mean_gap_real"].values[0] == 150.0
        assert early.loc[early["state"] == "B", "mean_gap_real"].values[0] == 550.0

    def test_partial_corr_uncorrelated(self):
        from analysis.temporal_holdout import _partial_corr
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 100)
        x = z + rng.normal(0, 0.01, 100)
        y = z + rng.normal(0, 0.01, 100)
        r, p = _partial_corr(x, y, z)
        # After removing z's effect, residuals should have near-zero correlation
        assert abs(r) < 0.5


class TestCrossOutcomeValidation:
    def test_fallback_outcomes_load(self):
        from analysis.cross_outcome_validation import _FALLBACK_OUTCOMES
        assert len(_FALLBACK_OUTCOMES) == 51  # 50 states + DC
        assert "California" in _FALLBACK_OUTCOMES
        assert "District of Columbia" in _FALLBACK_OUTCOMES
        # Poverty rates should be in plausible range
        for state, (pov, med) in _FALLBACK_OUTCOMES.items():
            assert 0 < pov < 30, f"{state} poverty rate {pov} out of range"
            assert 30000 < med < 120000, f"{state} median income {med} out of range"

    def test_outcomes_csv_created(self):
        """Loading outcomes should create the CSV if absent."""
        from analysis.cross_outcome_validation import _load_outcomes
        df = _load_outcomes()
        assert len(df) == 51
        assert "poverty_rate" in df.columns
        assert "median_income" in df.columns


class TestFinancialSignal:
    def test_compute_signals_basic(self):
        """H₀ gap on synthetic industry returns."""
        from analysis.financial_signal import compute_signals
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        # 10 fake industries: uniform returns except one outlier
        data = {}
        for i in range(10):
            data[f"Ind{i}"] = [1.0 * i, 0.5 * i, -0.3 * i]
        df = pd.DataFrame(data, index=dates)
        signals = compute_signals(df)
        assert len(signals) == 3
        assert "h0_gap" in signals.columns
        assert "dispersion" in signals.columns
        assert (signals["h0_gap"] >= 0).all()

    def test_gap_detects_bimodal(self):
        """Gap should be larger for bimodal vs uniform returns."""
        from analysis.financial_signal import compute_signals
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        # Month 1: uniform spread (0, 1, 2, ..., 9)
        uniform = {f"I{i}": [float(i), 0.0] for i in range(10)}
        # Month 2: bimodal cluster (0,0,0,0,0, 10,10,10,10,10)
        bimodal = {f"I{i}": [0.0, 0.0 if i < 5 else 10.0] for i in range(10)}
        data = {k: [uniform[k][0], bimodal[k][1]] for k in uniform}
        df = pd.DataFrame(data, index=dates)
        signals = compute_signals(df)
        assert signals.iloc[1]["h0_gap"] > signals.iloc[0]["h0_gap"]

    def test_partial_corr(self):
        from analysis.financial_signal import _partial_corr
        rng = np.random.default_rng(99)
        z = rng.normal(0, 1, 200)
        x = 2 * z + rng.normal(0, 0.1, 200)
        y = -z + rng.normal(0, 0.1, 200)
        r, p = _partial_corr(x, y, z)
        # After removing z, residuals should be near-uncorrelated
        assert abs(r) < 0.3
