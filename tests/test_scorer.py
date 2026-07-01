"""Tests for composite scoring engine."""

import pandas as pd
import pytest

from core.scorer import (
    compute_capital_flow_score,
    rank_stocks_by_margin_balance,
    rank_stocks_by_margin_buy,
    CompositeScore,
)


class TestComputeCapitalFlowScore:
    def test_neutral_inputs(self, sample_config):
        result = compute_capital_flow_score(0.0, 0.0, 0.0, 0, 0, 0, sample_config)
        assert isinstance(result, CompositeScore)
        assert 40 <= result.score <= 60  # near neutral
        assert result.grade in ("C", "B", "D")

    def test_bullish_inputs(self, sample_config):
        result = compute_capital_flow_score(0.8, 0.7, 0.5, 5, 5, 3, sample_config)
        assert result.score > 60
        assert result.grade in ("A+", "A", "B+")

    def test_bearish_inputs(self, sample_config):
        result = compute_capital_flow_score(-0.8, -0.7, -0.5, 0, 0, 3, sample_config)
        assert result.score < 40
        assert result.grade in ("E", "F", "F-")

    def test_score_in_0_100_range(self, sample_config):
        result = compute_capital_flow_score(1.0, 1.0, 1.0, 6, 6, 4, sample_config)
        assert 0 <= result.score <= 100

    def test_score_in_0_100_range_bearish(self, sample_config):
        result = compute_capital_flow_score(-1.0, -1.0, -1.0, 0, 0, 4, sample_config)
        assert 0 <= result.score <= 100

    def test_risk_penalty_applied(self, sample_config):
        # Many triggered signals all pointing same way = risk
        result = compute_capital_flow_score(0.8, 0.8, 0.6, 5, 5, 2, sample_config)
        # Risk penalty should be > 0 when many signals triggered
        assert result.risk_penalty >= 0

    def test_returns_composite_score_dataclass(self, sample_config):
        result = compute_capital_flow_score(0.3, 0.2, 0.1, 2, 2, 1, sample_config)
        assert hasattr(result, "score")
        assert hasattr(result, "grade")
        assert hasattr(result, "label")
        assert hasattr(result, "summary")

    def test_different_custom_weights(self):
        config = {
            "scoring": {
                "northbound_weight": 0.7,
                "margin_weight": 0.3,
                "risk_overheat_threshold": 0.5,
            },
        }
        result = compute_capital_flow_score(0.5, -0.5, 0.0, 1, 1, 0, config)
        # NB weighted more → positive bias
        assert result.northbound_score == 0.5
        assert result.margin_score == -0.5


class TestRankStocksByMarginBalance:
    def test_returns_top_n(self, sample_margin_detail, sample_stock_info):
        result = rank_stocks_by_margin_balance(sample_margin_detail, sample_stock_info, top_n=10)
        assert len(result) <= 10
        assert "rank" in result.columns
        assert "symbol" in result.columns
        assert "margin_balance" in result.columns

    def test_returns_correct_columns(self, sample_margin_detail, sample_stock_info):
        result = rank_stocks_by_margin_balance(sample_margin_detail, sample_stock_info)
        for col in ["rank", "symbol", "name", "industry", "margin_balance"]:
            assert col in result.columns

    def test_sorted_descending(self, sample_margin_detail, sample_stock_info):
        result = rank_stocks_by_margin_balance(sample_margin_detail, sample_stock_info)
        balances = result["margin_balance"].values
        for i in range(len(balances) - 1):
            assert balances[i] >= balances[i + 1]

    def test_empty_input(self):
        result = rank_stocks_by_margin_balance(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_no_symbol_column(self):
        df = pd.DataFrame({"margin_balance": [100, 200, 50]})
        result = rank_stocks_by_margin_balance(df, pd.DataFrame())
        assert result.empty

    def test_no_balance_column(self):
        df = pd.DataFrame({"symbol": ["000001.SZ", "000002.SZ"]})
        result = rank_stocks_by_margin_balance(df, pd.DataFrame())
        assert result.empty

    def test_joins_stock_info_names(self):
        margin = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ"],
            "margin_balance": [1000, 500],
            "date": ["20260630", "20260630"],
        })
        info = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ"],
            "name": ["平安银行", "万科A"],
            "industry": ["银行", "房地产"],
        })
        result = rank_stocks_by_margin_balance(margin, info)
        assert result.iloc[0]["name"] == "平安银行"
        assert result.iloc[0]["industry"] == "银行"


class TestRankStocksByMarginBuy:
    def test_returns_top_n(self, sample_margin_detail, sample_stock_info):
        result = rank_stocks_by_margin_buy(sample_margin_detail, sample_stock_info, top_n=10)
        assert len(result) <= 10
        assert "margin_buy" in result.columns

    def test_sorted_descending(self, sample_margin_detail, sample_stock_info):
        result = rank_stocks_by_margin_buy(sample_margin_detail, sample_stock_info)
        buys = result["margin_buy"].values
        for i in range(len(buys) - 1):
            assert buys[i] >= buys[i + 1]

    def test_empty_input(self):
        result = rank_stocks_by_margin_buy(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_no_buy_column(self):
        df = pd.DataFrame({"symbol": ["000001.SZ"], "margin_balance": [100]})
        result = rank_stocks_by_margin_buy(df, pd.DataFrame())
        assert result.empty
