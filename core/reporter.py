"""Report generator: Markdown daily report + JSON structured output.

Produces a 7-section Markdown report and a companion JSON file with
full signal breakdown for auditability and downstream consumption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ._types import SignalResult
from .resonance import ResonanceResult
from .scorer import CompositeScore

logger = logging.getLogger(__name__)


def _safe_fmt(val, fmt: str = ".2f") -> str:
    """Safely format a numeric value, returning '-' on NaN/None."""
    if val is None:
        return "-"
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return "-"
        return f"{f:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _provenance_label(source: str) -> str:
    """Map df.attrs source tag to Chinese provenance label."""
    labels = {
        "pandadata": "Pandadata (实时)",
        "eastmoney": "东方财富 (实时)",
        "degraded": "降级模式 (部分数据不可用)",
        "unknown": "未知来源",
        "none": "无数据",
    }
    return labels.get(source, source)


# ------------------------------------------------------------------
# Markdown report
# ------------------------------------------------------------------

def generate_markdown_report(
    trade_date: str,
    nb_signals: list[SignalResult],
    margin_signals: list[SignalResult],
    resonance_results: list[ResonanceResult],
    composite: CompositeScore,
    nb_summary: pd.DataFrame,
    margin_detail: pd.DataFrame,
    margin_macro: pd.DataFrame,
    top_margin_balance: pd.DataFrame,
    top_margin_buy: pd.DataFrame,
    stock_info: pd.DataFrame,
    fetch_time: str,
    config: dict,
    llm_analysis: str = "",
    llm_source: str = "none",
    futures_signals: list[SignalResult] | None = None,
    futures_data: pd.DataFrame | None = None,
) -> str:
    """Generate the full Markdown daily report.

    Args:
        trade_date: Trading day in YYYYMMDD format.
        nb_signals: All northbound signal results.
        margin_signals: All margin signal results.
        resonance_results: All resonance/divergence results.
        composite: Composite scoring result.
        nb_summary: Northbound daily summary DataFrame.
        margin_detail: Per-stock margin DataFrame.
        margin_macro: Aggregate margin history DataFrame.
        top_margin_balance: Top stocks by margin balance.
        top_margin_buy: Top stocks by margin buy amount.
        stock_info: Stock detail info DataFrame.
        fetch_time: ISO timestamp of data fetch.
        config: Full runtime config dict.
        futures_signals: Futures signal results (optional).
        futures_data: Index futures daily data (optional).

    Returns:
        Markdown report string.
    """
    if futures_signals is None:
        futures_signals = []
    if futures_data is None:
        futures_data = pd.DataFrame()

    dt = datetime.strptime(trade_date, "%Y%m%d")
    date_display = dt.strftime("%Y-%m-%d")

    lines: list[str] = []
    add = lines.append  # shorthand (avoid _ to prevent rebind in loops)

    # ── Header ──
    add(f"# 北向资金 + 融资融券全景监控 — {date_display}")
    add("")
    add(f"**生成时间**: {fetch_time} | **综合评分**: {composite.score:.0f}/100 ({composite.grade} — {composite.label})")
    add("")
    add("---")
    add("")

    # ── 1. Northbound overview ──
    add("## 一、北向资金概览")
    add("")

    nb_triggered = [s for s in nb_signals if s.triggered]
    nb_bullish = [s for s in nb_signals if s.direction == "bullish"]
    nb_bearish = [s for s in nb_signals if s.direction == "bearish"]

    add("| 指标 | 信号 | 方向 | 强度 | 解读 |")
    add("|------|------|------|------|------|")
    for s in nb_signals:
        direction_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(s.direction, "⚪")
        triggered_mark = "**触发**" if s.triggered else "未触发"
        add(f"| {s.label} | {triggered_mark} | {direction_icon} | {s.strength:+.2f} | {s.summary} |")
    add("")
    add(f"**北向综合评分**: {composite.northbound_score:+.3f} | "
      f"触发信号: {len(nb_triggered)}/6 | "
      f"看多: {len(nb_bullish)} | 看空: {len(nb_bearish)}")
    add("")

    # Northbound data summary
    if not nb_summary.empty:
        add("### 北向资金数据摘要")
        add("")
        mv_col = None
        for c in ["market_value", "持股市值"]:
            if c in nb_summary.columns:
                mv_col = c
                break
        net_col = None
        for c in ["net_buy_amount", "净买入", "资金净流入"]:
            if c in nb_summary.columns:
                net_col = c
                break

        if mv_col is not None:
            recent = pd.to_numeric(nb_summary[mv_col], errors="coerce").dropna()
            if len(recent) >= 2:
                latest_mv = recent.iloc[-1]
                prev_mv = recent.iloc[-6] if len(recent) >= 6 else recent.iloc[0]
                chg = (latest_mv - prev_mv) / prev_mv * 100 if prev_mv != 0 else 0
                add(f"- 最新持仓市值: {latest_mv/1e8:.0f} 亿元")
                add(f"- 5日市值变动: {chg:+.2f}%")
        if net_col is not None:
            recent_net = pd.to_numeric(nb_summary[net_col], errors="coerce").tail(5)
            net_5d = recent_net.sum()
            add(f"- 近5日净买入合计: {net_5d/1e8:+.1f} 亿元")
        add("")

    add("---")
    add("")

    # ── 2. Margin overview ──
    add("## 二、融资融券概览")
    add("")

    mg_triggered = [s for s in margin_signals if s.triggered]
    mg_bullish = [s for s in margin_signals if s.direction == "bullish"]
    mg_bearish = [s for s in margin_signals if s.direction == "bearish"]

    add("| 指标 | 信号 | 方向 | 强度 | 解读 |")
    add("|------|------|------|------|------|")
    for s in margin_signals:
        direction_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(s.direction, "⚪")
        triggered_mark = "**触发**" if s.triggered else "未触发"
        add(f"| {s.label} | {triggered_mark} | {direction_icon} | {s.strength:+.2f} | {s.summary} |")
    add("")
    add(f"**融资综合评分**: {composite.margin_score:+.3f} | "
      f"触发信号: {len(mg_triggered)}/6 | "
      f"看多: {len(mg_bullish)} | 看空: {len(mg_bearish)}")
    add("")

    # Margin macro summary
    if not margin_macro.empty:
        add("### 融资融券宏观数据")
        add("")
        # Show latest date's aggregate values, not all-history sum
        date_col = "date" if "date" in margin_macro.columns else "日期"
        if date_col in margin_macro.columns:
            latest_date = margin_macro[date_col].max()
            latest_data = margin_macro[margin_macro[date_col] == latest_date]
        else:
            latest_data = margin_macro

        for label, col_candidates in [
            ("融资余额", ["margin_balance", "融资余额"]),
            ("融券余额", ["short_balance", "融券余额"]),
            ("融资买入额", ["buy_on_margin_value", "融资买入额"]),
        ]:
            col = None
            for c in col_candidates:
                if c in latest_data.columns:
                    col = c
                    break
            if col is not None:
                total = pd.to_numeric(latest_data[col], errors="coerce").sum()
                if total > 0:
                    add(f"- {label}: {total/1e8:.0f} 亿元")
        add("")

    # Top margin stocks
    if not top_margin_balance.empty:
        add("### 融资余额 TOP10")
        add("")
        add("| 排名 | 代码 | 名称 | 行业 | 融资余额(亿) |")
        add("|------|------|------|------|-------------|")
        for _, row in top_margin_balance.head(10).iterrows():
            bal = float(row.get("margin_balance", 0)) / 1e8
            add(f"| {int(row['rank'])} | {row['symbol']} | {row.get('name', '-')} | {row.get('industry', '-')} | {bal:.1f} |")
        add("")

    if not top_margin_buy.empty:
        add("### 融资买入 TOP10")
        add("")
        add("| 排名 | 代码 | 名称 | 行业 | 融资买入(亿) |")
        add("|------|------|------|------|-------------|")
        for _, row in top_margin_buy.head(10).iterrows():
            buy = float(row.get("margin_buy", 0)) / 1e8
            add(f"| {int(row['rank'])} | {row['symbol']} | {row.get('name', '-')} | {row.get('industry', '-')} | {buy:.1f} |")
        add("")

    # ── 2.5. Futures overview ──
    if futures_signals:
        add("---")
        add("")
        add("## 三、股指期货信号")
        add("")

        f_triggered = [s for s in futures_signals if s.triggered]
        f_bullish = [s for s in futures_signals if s.direction == "bullish"]
        f_bearish = [s for s in futures_signals if s.direction == "bearish"]

        add("| 指标 | 信号 | 方向 | 强度 | 解读 |")
        add("|------|------|------|------|------|")
        for s in futures_signals:
            direction_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(s.direction, "⚪")
            triggered_mark = "**触发**" if s.triggered else "未触发"
            add(f"| {s.label} | {triggered_mark} | {direction_icon} | {s.strength:+.2f} | {s.summary} |")
        add("")
        add(f"**期货综合评分**: {composite.futures_score:+.3f} | "
            f"触发信号: {len(f_triggered)}/3 | "
            f"看多: {len(f_bullish)} | 看空: {len(f_bearish)}")
        add("")

        # Futures data summary
        if not futures_data.empty:
            add("### 期货数据摘要")
            add("")
            idx_col = None
            for c in ["index_name", "指数名称", "contract", "合约"]:
                if c in futures_data.columns:
                    idx_col = c
                    break
            date_col = None
            for c in ["date", "日期"]:
                if c in futures_data.columns:
                    date_col = c
                    break
            if idx_col is not None:
                for name, group in futures_data.groupby(idx_col):
                    if len(group) > 0:
                        last = group.sort_values(date_col).iloc[-1] if date_col else group.iloc[-1]
                        close = float(last.get("close", 0))
                        oi = float(last.get("open_interest", 0))
                        add(f"- {name}: 收盘 {close:.1f} | 持仓 {oi/10000:.0f}万手")
                add("")

    add("---")
    add("")

    # ── 3. Resonance analysis ──
    add("## 三、共振/背离分析")
    add("")

    triggered_resonance = [r for r in resonance_results if r.triggered]
    if triggered_resonance:
        for r in resonance_results:
            if r.triggered:
                icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(r.direction, "⚪")
                add(f"### {icon} {r.label}")
                add("")
                add(f"**强度**: {r.strength:+.3f}")
                add("")
                add(f"{r.description}")
                add("")
    else:
        add("*无共振/背离信号触发*")
        add("")
        add(f"共振评分: {composite.resonance_score:+.3f}")
        add("")

    add("---")
    add("")

    # ── 4. Sentiment summary ──
    add("## 四、综合情绪评估")
    add("")
    add(f"**综合评分**: {composite.score:.0f}/100 ({composite.grade})")
    add("")
    add(f"**评估**: {composite.label} — {composite.summary}")
    add("")
    add(f"**风险惩罚**: {composite.risk_penalty:.0%}")
    add("")

    add("### 分项得分")
    add("")
    add(f"| 维度 | 得分 | 说明 |")
    add(f"|------|------|------|")
    add(f"| 北向资金 | {composite.northbound_score:+.3f} | {len(nb_triggered)}/6 信号触发 |")
    add(f"| 融资融券 | {composite.margin_score:+.3f} | {len(mg_triggered)}/6 信号触发 |")
    if futures_signals:
        f_triggered_count = sum(1 for s in futures_signals if s.triggered)
        add(f"| 股指期货 | {composite.futures_score:+.3f} | {f_triggered_count}/3 信号触发 |")
    add(f"| 共振背离 | {composite.resonance_score:+.3f} | {len(triggered_resonance)}/4 模式触发 |")
    add(f"| **综合** | **{composite.score:.0f}/100** | **{composite.grade}** |")
    add("")

    add("---")
    add("")

    # ── 4.5. LLM Macro Analysis ──
    if llm_analysis and llm_source not in ("none",):
        add("## 四·五、AI 宏观研判")
        add("")
        if llm_source == "real":
            add(f"*由 {config.get('llm', {}).get('model', 'LLM')} 基于以上全部信号综合分析生成*")
        elif llm_source == "fallback":
            add("*⚠️ LLM API 不可用，以下为规则引擎生成的基础分析*")
        else:
            add(f"*⚠️ LLM 分析异常（{llm_source}）*")
        add("")
        add(llm_analysis)
        add("")

    add("---")
    add("")

    # ── 6. Sector flow ──
    add("## 六、板块资金流向")
    add("")

    if not stock_info.empty and "industry" in stock_info.columns:
        industry_counts = stock_info["industry"].value_counts()
        add(f"**A股行业分布** (共{len(industry_counts)}个行业):")
        add("")
        add("| 行业 | 上市公司数 |")
        add("|------|-----------|")
        for ind, count in industry_counts.head(10).items():
            add(f"| {ind} | {count} |")
        add("")
        add("*注: 个股北向持仓数据不可直接获取，板块资金流向基于汇总数据+行业分布推算。*")
    else:
        add("*行业分类数据不可用*")
    add("")

    add("---")
    add("")

    # ── 7. Risk alerts ──
    add("## 七、风险提示")
    add("")

    # Compile risk alerts from all signals
    risks: list[str] = []
    for s in nb_signals:
        if s.triggered and s.direction == "bearish" and abs(s.strength) > 0.5:
            risks.append(f"- [北向] {s.summary}")
    for s in margin_signals:
        if s.triggered and s.direction == "bearish" and abs(s.strength) > 0.5:
            risks.append(f"- [融资] {s.summary}")
    for r in resonance_results:
        if r.triggered and r.direction == "bearish":
            risks.append(f"- [共振] {r.summary}")

    if risks:
        add(f"**共{len(risks)}项风险提示:**")
        add("")
        for risk in risks:
            add(risk)
        add("")
    else:
        add("*当前无重大风险提示*")
        add("")

    add("---")
    add("")

    # ── 8. Data provenance ──
    add("## 八、数据溯源")
    add("")
    add("| 数据类别 | 数据源 | 状态 |")
    add("|----------|--------|------|")

    nb_source = nb_summary.attrs.get("source", "unknown") if not nb_summary.empty else "none"
    mg_detail_source = margin_detail.attrs.get("source", "unknown") if not margin_detail.empty else "none"
    mg_macro_source = margin_macro.attrs.get("source", "unknown") if not margin_macro.empty else "none"
    info_source = stock_info.attrs.get("source", "unknown") if not stock_info.empty else "none"

    add(f"| 北向资金汇总 | {_provenance_label(nb_source)} | {'✅' if nb_source != 'degraded' else '⚠️'} |")
    add(f"| 融资融券明细 | {_provenance_label(mg_detail_source)} | {'✅' if mg_detail_source != 'degraded' else '⚠️'} |")
    add(f"| 融资融券宏观 | {_provenance_label(mg_macro_source)} | {'✅' if mg_macro_source != 'degraded' else '⚠️'} |")
    add(f"| 股票信息 | {_provenance_label(info_source)} | {'✅' if info_source != 'degraded' else '⚠️'} |")
    fut_source = futures_data.attrs.get("source", "none") if not futures_data.empty else "none"
    add(f"| 股指期货 | {_provenance_label(fut_source)} | {'✅' if fut_source not in ('degraded', 'none') else '⚠️'} |")
    add("")

    add("---")
    add(f"*报告由 skill-northbound-margin-monitor 自动生成 | {fetch_time}*")

    return "\n".join(lines)


# ------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------

def _to_native(val):
    """Convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_to_native(v) for v in val.tolist()]
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(v) for v in val]
    return val


def generate_json_output(
    trade_date: str,
    nb_signals: list[SignalResult],
    margin_signals: list[SignalResult],
    resonance_results: list[ResonanceResult],
    composite: CompositeScore,
    top_margin_balance: pd.DataFrame,
    top_margin_buy: pd.DataFrame,
    fetch_time: str,
    nb_summary: pd.DataFrame,
    margin_detail: pd.DataFrame,
    margin_macro: pd.DataFrame,
    stock_info: pd.DataFrame,
    config: dict,
    llm_analysis: str = "",
    llm_source: str = "none",
    futures_signals: list[SignalResult] | None = None,
    futures_data: pd.DataFrame | None = None,
) -> dict:
    """Generate structured JSON output for downstream consumption.

    Returns:
        dict ready for json.dump().
    """
    if futures_signals is None:
        futures_signals = []
    if futures_data is None:
        futures_data = pd.DataFrame()
    def signal_to_dict(s):
        return {
            "key": s.key,
            "label": s.label,
            "triggered": s.triggered,
            "strength": s.strength,
            "direction": s.direction,
            "summary": s.summary,
            "detail": s.detail,
        }

    def resonance_to_dict(r):
        return {
            "pattern": r.pattern,
            "label": r.label,
            "triggered": r.triggered,
            "strength": r.strength,
            "direction": r.direction,
            "summary": r.summary,
            "description": r.description,
        }

    # Serialize top-N DataFrames
    top_bal = []
    if not top_margin_balance.empty:
        for _, row in top_margin_balance.head(10).iterrows():
            top_bal.append({
                "rank": int(row["rank"]),
                "symbol": str(row["symbol"]),
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
                "margin_balance": float(row.get("margin_balance", 0)),
            })

    top_buy = []
    if not top_margin_buy.empty:
        for _, row in top_margin_buy.head(10).iterrows():
            top_buy.append({
                "rank": int(row["rank"]),
                "symbol": str(row["symbol"]),
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
                "margin_buy": float(row.get("margin_buy", 0)),
            })

    # Provenance
    provenance = {
        "northbound_summary": nb_summary.attrs.get("source", "none") if not nb_summary.empty else "none",
        "margin_detail": margin_detail.attrs.get("source", "none") if not margin_detail.empty else "none",
        "margin_macro": margin_macro.attrs.get("source", "none") if not margin_macro.empty else "none",
        "futures": futures_data.attrs.get("source", "none") if not futures_data.empty else "none",
        "stock_info": stock_info.attrs.get("source", "none") if not stock_info.empty else "none",
    }

    result = {
        "meta": {
            "trade_date": trade_date,
            "fetch_time": fetch_time,
            "generator": "skill-northbound-margin-monitor",
            "version": "1.0.0",
        },
        "composite": {
            "score": composite.score,
            "grade": composite.grade,
            "label": composite.label,
            "northbound_score": composite.northbound_score,
            "margin_score": composite.margin_score,
            "futures_score": composite.futures_score,
            "resonance_score": composite.resonance_score,
            "risk_penalty": composite.risk_penalty,
            "summary": composite.summary,
        },
        "northbound": {
            "signals": [signal_to_dict(s) for s in nb_signals],
            "triggered_count": sum(1 for s in nb_signals if s.triggered),
            "bullish_count": sum(1 for s in nb_signals if s.direction == "bullish"),
            "bearish_count": sum(1 for s in nb_signals if s.direction == "bearish"),
        },
        "margin": {
            "signals": [signal_to_dict(s) for s in margin_signals],
            "triggered_count": sum(1 for s in margin_signals if s.triggered),
            "bullish_count": sum(1 for s in margin_signals if s.direction == "bullish"),
            "bearish_count": sum(1 for s in margin_signals if s.direction == "bearish"),
            "top_by_balance": top_bal,
            "top_by_buy": top_buy,
        },
        "futures": {
            "signals": [signal_to_dict(s) for s in futures_signals],
            "triggered_count": sum(1 for s in futures_signals if s.triggered),
            "bullish_count": sum(1 for s in futures_signals if s.direction == "bullish"),
            "bearish_count": sum(1 for s in futures_signals if s.direction == "bearish"),
        },
        "resonance": {
            "patterns": [resonance_to_dict(r) for r in resonance_results],
            "triggered_count": sum(1 for r in resonance_results if r.triggered),
        },
        "llm": {
            "source": llm_source,
            "analysis": llm_analysis,
            "model": config.get("llm", {}).get("model", "unknown"),
        },
        "provenance": provenance,
    }
    return _to_native(result)


# ------------------------------------------------------------------
# Write output files
# ------------------------------------------------------------------

def write_report(
    trade_date: str,
    markdown: str,
    json_data: dict,
    output_dir: Optional[str | Path] = None,
    config: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Write Markdown and JSON report files to the output directory.

    Output path: ``<output_dir>/YYYY-MM-DD/panorama_monitor_YYYYMMDD.{md,json}``

    Args:
        trade_date: Trading day in YYYYMMDD format.
        markdown: Markdown report string.
        json_data: JSON-serialisable dict.
        output_dir: Override output directory. Defaults to config.output.dir.
        config: Full runtime config dict.

    Returns:
        (md_path, json_path) as Path objects.
    """
    cfg = config or {}
    out_dir = Path(output_dir or cfg.get("output", {}).get("dir", "output"))

    dt = datetime.strptime(trade_date, "%Y%m%d")
    date_dir = out_dir / dt.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"panorama_monitor_{trade_date}"

    md_path = date_dir / f"{base_name}.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown report written: %s", md_path)

    json_path = date_dir / f"{base_name}.json"
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON report written: %s", json_path)

    return md_path, json_path
