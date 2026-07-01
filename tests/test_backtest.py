"""Tests for core/backtest.py — historical IC backtesting."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    BacktestResult,
    _spearmanr,
    backtest_summary_table,
    run_ic_backtest,
)


class TestSpearmanR:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r = _spearmanr(x, y)
        assert 0.99 <= r <= 1.01

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        r = _spearmanr(x, y)
        assert -1.01 <= r <= -0.99

    def test_no_correlation(self):
        np.random.seed(99)
        x = np.random.randn(50)
        y = np.random.randn(50)
        r = _spearmanr(x, y)
        assert -0.5 < r < 0.5

    def test_too_few_points(self):
        assert _spearmanr(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0

    def test_constant_array(self):
        r = _spearmanr(np.array([1.0, 1.0, 1.0, 1.0]), np.array([2.0, 3.0, 4.0, 5.0]))
        assert r == 0.0


class TestICBacktest:
    @pytest.fixture
    def backtest_nb(self):
        """Build a northbound summary with a predictable score-return relationship."""
        dates = pd.date_range("2025-01-02", periods=150, freq="B")
        np.random.seed(123)
        # Trend up with noise
        trend = np.linspace(3500, 4200, 150)
        noise = np.random.randn(150) * 30
        csi300 = trend + noise
        market_value = 18000 + np.cumsum(np.random.randn(150) * 40)
        net_buy = np.diff(market_value, prepend=market_value[0])

        return pd.DataFrame({
            "date": [d.strftime("%Y%m%d") for d in dates],
            "market_value": market_value,
            "net_buy_amount": net_buy,
            "CSI300": csi300,
        })

    @pytest.fixture
    def backtest_margin(self):
        """Build margin macro data."""
        dates = pd.date_range("2025-01-02", periods=150, freq="B")
        np.random.seed(124)
        margin_bal = 14000 + np.cumsum(np.random.randn(150) * 25)
        short_bal = 700 + np.cumsum(np.random.randn(150) * 4)
        buy = np.random.uniform(350, 700, 150)
        rows = []
        for i, d in enumerate(dates):
            for mkt in ["sh", "sz"]:
                rows.append({
                    "date": d.strftime("%Y%m%d"),
                    "market": mkt,
                    "margin_balance": margin_bal[i] * 0.55,
                    "short_balance": short_bal[i] * 0.6,
                    "buy_on_margin_value": buy[i] * 0.5,
                })
        return pd.DataFrame(rows)

    def test_backtest_returns_result(self, backtest_nb, backtest_margin):
        result = run_ic_backtest(
            backtest_nb, backtest_margin,
            min_window=60, step_days=5,
        )
        assert isinstance(result, BacktestResult)
        assert result.n_periods > 0
        assert len(result.score_series) == result.n_periods
        assert len(result.score_series) == len(result.return_series)

    def test_backtest_ic_range(self, backtest_nb, backtest_margin):
        result = run_ic_backtest(
            backtest_nb, backtest_margin,
            min_window=60, step_days=5,
        )
        assert -1.0 <= result.mean_ic <= 1.0
        if result.n_periods >= 5:
            assert result.icir == result.mean_ic / result.ic_std if result.ic_std > 0 else 0.0

    def test_backtest_hit_rate_range(self, backtest_nb, backtest_margin):
        result = run_ic_backtest(
            backtest_nb, backtest_margin,
            min_window=60, step_days=5,
        )
        assert 0.0 <= result.hit_rate <= 1.0

    def test_backtest_insufficient_data(self):
        """Too few rows should raise ValueError with clear message."""
        nb = pd.DataFrame({
            "date": ["20250101", "20250102"],
            "market_value": [20000, 20100],
            "net_buy_amount": [0, 100],
            "CSI300": [4000, 4010],
        })
        mg = pd.DataFrame({
            "date": ["20250101", "20250102"],
            "market": ["sh", "sz"],
            "margin_balance": [15000, 15000],
            "short_balance": [500, 500],
            "buy_on_margin_value": [300, 300],
        })
        with pytest.raises(ValueError, match="Need at least"):
            run_ic_backtest(nb, mg, min_window=60)

    def test_backtest_summary_table(self, backtest_nb, backtest_margin):
        result = run_ic_backtest(backtest_nb, backtest_margin, min_window=60, step_days=5)
        table = backtest_summary_table(result)
        assert "Mean IC" in table
        assert "ICIR" in table
        assert "Hit Rate" in table
        assert f"{result.mean_ic:.4f}" in table

    def test_backtest_with_config(self, backtest_nb, backtest_margin):
        config = {"scoring": {"northbound_weight": 0.6, "margin_weight": 0.4}}
        result = run_ic_backtest(backtest_nb, backtest_margin, config=config, min_window=60, step_days=5)
        assert isinstance(result, BacktestResult)

    def test_dense_step(self, backtest_nb, backtest_margin):
        """step_days=1 should produce more evaluation periods."""
        result = run_ic_backtest(backtest_nb, backtest_margin, min_window=60, step_days=1)
        assert result.n_periods > 5
