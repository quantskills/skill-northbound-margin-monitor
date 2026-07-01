"""Composite scoring engine for the northbound + margin panorama.

Combines northbound signals, margin signals, and resonance patterns
into a single 0–100 capital flow composite score with risk penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    """Overall capital flow assessment."""
    score: float              # 0–100 composite score
    grade: str                # A+ to F
    label: str                # Chinese label (极度乐观 ~ 极度悲观)
    northbound_score: float   # -1 to 1
    margin_score: float       # -1 to 1
    futures_score: float      # -1 to 1
    resonance_score: float    # -1 to 1
    risk_penalty: float       # 0–1, higher = more risk
    summary: str


# ------------------------------------------------------------------
# Composite scoring
# ------------------------------------------------------------------

def compute_capital_flow_score(
    nb_composite: float,
    margin_composite: float,
    resonance_score: float,
    nb_triggered_count: int,
    margin_triggered_count: int,
    resonance_triggered_count: int,
    config: dict,
    futures_composite: float = 0.0,
    futures_triggered_count: int = 0,
) -> CompositeScore:
    """Compute the 0–100 capital flow composite score.

    Formula:
        base = nb_weight * nb_composite + margin_weight * margin_composite
               + futures_weight * futures_composite
        with_resonance = base * (1 + resonance_modifier)
        score_0_100 = (with_resonance + 1) * 50  (map -1..1 → 0..100)
        risk_penalty applied if triggered signals cross danger thresholds

    Args:
        nb_composite: Northbound composite score (-1 to 1).
        margin_composite: Margin composite score (-1 to 1).
        resonance_score: Resonance/divergence score (-1 to 1).
        nb_triggered_count: Number of triggered northbound signals.
        margin_triggered_count: Number of triggered margin signals.
        resonance_triggered_count: Number of triggered resonance patterns.
        config: Full runtime config dict.
        futures_composite: Futures composite score (-1 to 1). Default 0.0.
        futures_triggered_count: Number of triggered futures signals. Default 0.

    Returns:
        CompositeScore with 0–100 score and grade.
    """
    scoring_cfg = config.get("scoring", {})
    nb_weight = scoring_cfg.get("northbound_weight", 0.4)
    margin_weight = scoring_cfg.get("margin_weight", 0.4)
    futures_weight = scoring_cfg.get("futures_weight", 0.2)
    risk_threshold = scoring_cfg.get("risk_overheat_threshold", 0.7)

    # Base weighted score (3 dimensions)
    base = (
        nb_weight * nb_composite
        + margin_weight * margin_composite
        + futures_weight * futures_composite
    )

    # Resonance amplifier (configurable)
    resonance_mod = 0.0
    if resonance_triggered_count > 0:
        strong_threshold = scoring_cfg.get("resonance_strong_threshold", 0.3)
        if resonance_score > strong_threshold:
            resonance_mod = scoring_cfg.get("resonance_boost_strong", 0.15)
        elif resonance_score > 0:
            resonance_mod = scoring_cfg.get("resonance_boost_weak", 0.08)
        elif resonance_score < -strong_threshold:
            resonance_mod = scoring_cfg.get("resonance_dampen_strong", -0.10)
        elif resonance_score < 0:
            resonance_mod = scoring_cfg.get("resonance_dampen_weak", -0.05)

    with_resonance = base * (1.0 + resonance_mod)

    # Clamp to -1..1
    with_resonance = max(-1.0, min(1.0, with_resonance))

    # Map to 0–100
    score_0_100 = round((with_resonance + 1.0) * 50.0, 1)

    # Risk penalty: if many triggered signals all pointing same way,
    # it could indicate overheating or panic
    risk_penalty = 0.0
    total_triggered = nb_triggered_count + margin_triggered_count + futures_triggered_count
    high_trig = scoring_cfg.get("risk_penalty_high_triggered", 8)
    high_amount = scoring_cfg.get("risk_penalty_high_amount", 0.2)
    high_base = scoring_cfg.get("risk_penalty_high_base_threshold", 0.6)
    med_trig = scoring_cfg.get("risk_penalty_medium_triggered", 6)
    med_amount = scoring_cfg.get("risk_penalty_medium_amount", 0.1)
    med_base = scoring_cfg.get("risk_penalty_medium_base_threshold", 0.5)

    if total_triggered >= high_trig and abs(base) > high_base:
        risk_penalty = high_amount
    elif total_triggered >= med_trig and abs(base) > med_base:
        risk_penalty = med_amount

    if risk_penalty > 0:
        if base > 0:
            score_0_100 = max(0, score_0_100 - risk_penalty * 50)
        else:
            score_0_100 = min(100, score_0_100 + risk_penalty * 50)

    # Assign grade
    grade, label = _assign_grade(score_0_100)

    # Build summary
    if score_0_100 >= 70:
        summary = f"资金面{label}（{score_0_100:.0f}分），北向+融资双多，做多信号强烈"
    elif score_0_100 >= 55:
        summary = f"资金面{label}（{score_0_100:.0f}分），整体偏多，关注共振信号"
    elif score_0_100 >= 45:
        summary = f"资金面{label}（{score_0_100:.0f}分），多空均衡，等待方向确认"
    elif score_0_100 >= 30:
        summary = f"资金面{label}（{score_0_100:.0f}分），整体偏空，注意风险控制"
    else:
        summary = f"资金面{label}（{score_0_100:.0f}分），北向+融资双空，避险信号强烈"

    return CompositeScore(
        score=score_0_100,
        grade=grade,
        label=label,
        northbound_score=round(nb_composite, 4),
        margin_score=round(margin_composite, 4),
        futures_score=round(futures_composite, 4),
        resonance_score=round(resonance_score, 4),
        risk_penalty=round(risk_penalty, 4),
        summary=summary,
    )


def _assign_grade(score: float) -> tuple[str, str]:
    """Map 0–100 score to grade and Chinese label."""
    if score >= 85:
        return "A+", "极度乐观"
    elif score >= 75:
        return "A", "乐观"
    elif score >= 65:
        return "B+", "偏乐观"
    elif score >= 55:
        return "B", "中性偏多"
    elif score >= 45:
        return "C", "中性"
    elif score >= 35:
        return "D", "中性偏空"
    elif score >= 25:
        return "E", "偏悲观"
    elif score >= 15:
        return "F", "悲观"
    else:
        return "F-", "极度悲观"


# ------------------------------------------------------------------
# Stock ranking helpers
# ------------------------------------------------------------------

def rank_stocks_by_margin_balance(
    margin_detail: pd.DataFrame,
    stock_info: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank stocks by margin balance, joining with stock info for names.

    Args:
        margin_detail: Per-stock margin data.
        stock_info: Stock detail info with symbol, name, industry.
        top_n: Number of top stocks to return.

    Returns:
        DataFrame with columns: symbol, name, industry, margin_balance, rank.
    """
    if margin_detail.empty:
        return pd.DataFrame()

    # Resolve columns
    sym_col = None
    bal_col = None
    for c in ["symbol", "股票代码"]:
        if c in margin_detail.columns:
            sym_col = c
            break
    for c in ["margin_balance", "融资余额"]:
        if c in margin_detail.columns:
            bal_col = c
            break

    if sym_col is None or bal_col is None:
        logger.warning("Cannot rank by margin balance: missing columns")
        return pd.DataFrame()

    df = margin_detail.copy()
    df[bal_col] = pd.to_numeric(df[bal_col], errors="coerce")

    # Get latest per symbol
    if "date" in df.columns:
        df = df.sort_values("date")
        df = df.groupby(sym_col).last().reset_index()

    top = df.nlargest(top_n, bal_col)

    result = top[[sym_col, bal_col]].copy()
    result["rank"] = range(1, len(result) + 1)

    # Join with stock_info for names
    if not stock_info.empty:
        info_sym_col = None
        name_col = None
        for c in ["symbol", "股票代码"]:
            if c in stock_info.columns:
                info_sym_col = c
                break
        for c in ["name", "股票名称"]:
            if c in stock_info.columns:
                name_col = c
                break
        if info_sym_col is not None and name_col is not None:
            info_map = stock_info.set_index(info_sym_col)[name_col].to_dict()
            result["name"] = result[sym_col].map(info_map).fillna("")
        else:
            result["name"] = ""
    else:
        result["name"] = ""

    # Add industry if available
    if not stock_info.empty and "industry" in stock_info.columns:
        ind_col = "symbol" if "symbol" in stock_info.columns else "股票代码"
        if ind_col in stock_info.columns:
            ind_map = stock_info.set_index(ind_col)["industry"].to_dict()
            result["industry"] = result[sym_col].map(ind_map).fillna("")
        else:
            result["industry"] = ""
    else:
        result["industry"] = ""

    # Rename balance column
    result = result.rename(columns={bal_col: "margin_balance", sym_col: "symbol"})
    result = result[["rank", "symbol", "name", "industry", "margin_balance"]]

    return result


def rank_stocks_by_margin_buy(
    margin_detail: pd.DataFrame,
    stock_info: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank stocks by margin buy amount (当日融资买入额).

    Args:
        margin_detail: Per-stock margin data.
        stock_info: Stock detail info.
        top_n: Number of top stocks to return.

    Returns:
        DataFrame with columns: symbol, name, industry, buy_amount, rank.
    """
    if margin_detail.empty:
        return pd.DataFrame()

    sym_col = None
    buy_col = None
    for c in ["symbol", "股票代码"]:
        if c in margin_detail.columns:
            sym_col = c
            break
    for c in ["buy_on_margin_value", "融资买入额"]:
        if c in margin_detail.columns:
            buy_col = c
            break

    if sym_col is None or buy_col is None:
        logger.warning("Cannot rank by margin buy: missing columns")
        return pd.DataFrame()

    df = margin_detail.copy()
    df[buy_col] = pd.to_numeric(df[buy_col], errors="coerce")

    if "date" in df.columns:
        df = df.sort_values("date")
        df = df.groupby(sym_col).last().reset_index()

    top = df.nlargest(top_n, buy_col)
    result = top[[sym_col, buy_col]].copy()
    result["rank"] = range(1, len(result) + 1)

    # Join with stock_info
    if not stock_info.empty:
        info_sym = "symbol" if "symbol" in stock_info.columns else "股票代码"
        info_name = "name" if "name" in stock_info.columns else "股票名称"
        if info_sym in stock_info.columns and info_name in stock_info.columns:
            name_map = stock_info.set_index(info_sym)[info_name].to_dict()
            result["name"] = result[sym_col].map(name_map).fillna("")
        else:
            result["name"] = ""
        if "industry" in stock_info.columns:
            ind_map = stock_info.set_index(info_sym)["industry"].to_dict() if info_sym in stock_info.columns else {}
            result["industry"] = result[sym_col].map(ind_map).fillna("")
        else:
            result["industry"] = ""
    else:
        result["name"] = ""
        result["industry"] = ""

    result = result.rename(columns={buy_col: "margin_buy", sym_col: "symbol"})
    result = result[["rank", "symbol", "name", "industry", "margin_buy"]]

    return result
