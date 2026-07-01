"""LLM macro analyst: interprets panorama signals holistically via DeepSeek/Claude API.

Unlike the per-stock screener, this produces ONE macro-level analysis per run,
synthesising all 12 northbound + margin signals, 4 resonance patterns, and the
composite score into a comprehensive market regime assessment.

Supports Anthropic Claude and DeepSeek (via Anthropic-compatible endpoint).
Set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic for DeepSeek.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------

PANORAMA_PROMPT_TEMPLATE = """你是一位资深 A 股宏观策略分析师，专精于资金面和市场情绪分析。
以下是今日北向资金（沪深港通）+ 融资融券 + 股指期货全景监控的全部信号数据。
请基于这些信号，做一次专业的宏观资金面研判。

## 综合评分
{composite_summary}

## 北向资金信号（6项）
{nb_signals_table}

## 融资融券信号（6项）
{mg_signals_table}

## 股指期货信号
{futures_table}

## 共振/背离分析
{resonance_table}

## 关键宏观数据
{macro_data}

## 要求
1. **市场状态判断**：当前资金面处于什么阶段？（进攻/防守/观望）用一句话总结。
2. **三维交叉验证**：北向、融资、期货三个独立维度是否相互印证？如果有矛盾，哪个维度更可信？
3. **风险点识别**：列出 1-3 个最值得警惕的风险信号。
4. **机会点识别**：如果有积极信号，指出可能的机会方向。
5. **操作建议**：分别对短线交易者、中线投资者、长期配置资金给出差异化建议（各一句话）。
6. **次日观察清单**：列出明天需要重点关注的 2-3 个指标/数据。
7. 语言专业但不晦涩，用「可能」「倾向于」「值得关注」等概率化措辞，避免绝对判断。
8. 控制在 400-600 字。"""


def build_panorama_prompt(
    composite_score: float,
    composite_grade: str,
    composite_label: str,
    composite_summary: str,
    nb_signals: list,
    margin_signals: list,
    resonance_results: list,
    margin_balance_亿: float = 0,
    short_balance_亿: float = 0,
    margin_buy_亿: float = 0,
    top_industries: Optional[list[str]] = None,
    futures_signals: list | None = None,
) -> str:
    """Build the LLM prompt from panorama signal results.

    Args:
        composite_score: 0-100 composite score.
        composite_grade: Letter grade (A+ to F-).
        composite_label: Chinese label.
        composite_summary: One-line summary from scorer.
        nb_signals: List of northbound SignalResult.
        margin_signals: List of margin SignalResult.
        resonance_results: List of ResonanceResult.
        margin_balance_亿: Total margin balance in 亿 CNY.
        short_balance_亿: Total short balance in 亿 CNY.
        margin_buy_亿: Total margin buy amount in 亿 CNY.
        top_industries: Top industry names for context.
        futures_signals: List of futures SignalResult (optional).

    Returns:
        Formatted prompt string ready for the LLM.
    """
    # Composite summary line
    composite_line = (
        f"综合评分 {composite_score:.0f}/100（{composite_grade} — {composite_label}）\n"
        f"解读：{composite_summary}"
    )

    # Northbound signals table
    nb_lines: list[str] = []
    for s in nb_signals:
        direction_icon = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(s.direction, "中性")
        triggered = "触发" if s.triggered else "未触发"
        nb_lines.append(
            f"| {s.label} | {triggered} | {direction_icon} | 强度{s.strength:+.2f} | {s.summary} |"
        )
    nb_table = "\n".join(nb_lines) if nb_lines else "（无北向数据）"

    # Margin signals table
    mg_lines: list[str] = []
    for s in margin_signals:
        direction_icon = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(s.direction, "中性")
        triggered = "触发" if s.triggered else "未触发"
        mg_lines.append(
            f"| {s.label} | {triggered} | {direction_icon} | 强度{s.strength:+.2f} | {s.summary} |"
        )
    mg_table = "\n".join(mg_lines) if mg_lines else "（无融资融券数据）"

    # Resonance table
    res_lines: list[str] = []
    for r in resonance_results:
        direction_icon = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(r.direction, "中性")
        triggered = "触发" if r.triggered else "未触发"
        desc = r.description if r.description else r.summary
        res_lines.append(f"| {r.label} | {triggered} | {direction_icon} | {desc} |")
    resonance_table = "\n".join(res_lines) if res_lines else "（无共振/背离信号）"

    # Futures signals table
    futures_signals = futures_signals or []
    fut_lines: list[str] = []
    for s in futures_signals:
        direction_icon = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(s.direction, "中性")
        triggered = "触发" if s.triggered else "未触发"
        fut_lines.append(
            f"| {s.label} | {triggered} | {direction_icon} | 强度{s.strength:+.2f} | {s.summary} |"
        )
    futures_table = "\n".join(fut_lines) if fut_lines else "（无股指期货数据）"

    # Macro data
    industries_str = "、".join(top_industries[:5]) if top_industries else "数据不可用"
    macro_data = (
        f"- 融资余额：{margin_balance_亿:.0f} 亿元\n"
        f"- 融券余额：{short_balance_亿:.0f} 亿元\n"
        f"- 融资买入额：{margin_buy_亿:.0f} 亿元\n"
        + (f"- 融资/融券比：{margin_balance_亿 / short_balance_亿:.1f}\n" if short_balance_亿 > 0 else "")
        + f"- 重点行业：{industries_str}"
    )

    return PANORAMA_PROMPT_TEMPLATE.format(
        composite_summary=composite_line,
        nb_signals_table=nb_table,
        mg_signals_table=mg_table,
        resonance_table=resonance_table,
        futures_table=futures_table,
        macro_data=macro_data,
    )


# ------------------------------------------------------------------
# Fallback generation
# ------------------------------------------------------------------

def _generate_fallback(
    composite_score: float,
    composite_label: str,
    nb_triggered: int,
    mg_triggered: int,
    resonance_triggered: int,
    margin_balance_亿: float = 0,
) -> str:
    """Generate rule-based macro analysis when LLM is unavailable."""

    # Determine regime
    if composite_score >= 65:
        regime = "偏进攻"
        regime_detail = "资金面整体向好，北向与融资信号偏多"
    elif composite_score >= 45:
        regime = "观望偏多"
        regime_detail = "资金面中性，多空信号均衡，建议等待更明确的方向确认"
    elif composite_score >= 30:
        regime = "防御偏谨慎"
        regime_detail = "资金面偏空，存在需要关注的风险信号"
    else:
        regime = "防御"
        regime_detail = "资金面显著偏空，多项风险信号触发，建议降低仓位控制风险"

    # Risk identification
    risks: list[str] = []
    if mg_triggered >= 5:
        risks.append("融资融券多项信号触发，杠杆情绪活跃，需关注过度杠杆风险")
    if resonance_triggered > 0:
        risks.append(f"共振/背离信号触发（{resonance_triggered}项），跨市场信号需重点关注")
    if margin_balance_亿 > 30000:
        risks.append(f"融资余额{ margin_balance_亿:.0f}亿处于历史高位，去杠杆风险上升")

    risk_text = "\n".join(f"  - {r}" for r in risks) if risks else "  - 当前无显著极端风险信号"

    nb_info = f"{nb_triggered}/6 触发" if nb_triggered else "无触发"
    mg_info = f"{mg_triggered}/6 触发" if mg_triggered else "无触发"

    return (
        f"## 市场状态\n\n"
        f"当前资金面处于 **{regime}** 阶段。{regime_detail}。\n\n"
        f"## 信号概况\n\n"
        f"- 北向资金：{nb_info}\n"
        f"- 融资融券：{mg_info}\n"
        f"- 综合评分：{composite_score:.0f}/100（{composite_label}）\n\n"
        f"## 风险提示\n\n"
        f"{risk_text}\n\n"
        f"## 操作参考\n\n"
        f"- 短线：{'减少操作频率，等待信号明朗' if composite_score < 45 else '关注强势板块，快进快出'}\n"
        f"- 中线：{'控制仓位，等待右侧确认信号' if composite_score < 55 else '可适度参与，设好止盈止损'}\n"
        f"- 长线：{'分批布局优质标的，利用调整建仓' if composite_score >= 35 else '暂停加仓，等待系统性风险释放'}\n\n"
        f"> 此为规则引擎自动生成的基础分析，完整分析需配置 LLM 后运行。"
    )


# ------------------------------------------------------------------
# LLM Analyst
# ------------------------------------------------------------------

class LLMAnalyst:
    """Generate macro panorama analysis using Claude or DeepSeek API.

    Set ANTHROPIC_BASE_URL to use DeepSeek or other Anthropic-compatible endpoints.
    ANTHROPIC_MODEL selects the model (default: claude-sonnet-4-6).
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._model = (
            os.getenv("ANTHROPIC_MODEL")
            or self._config.get("llm", {}).get("model", "claude-sonnet-4-6")
        )
        self._max_tokens = int(
            self._config.get("llm", {}).get("max_tokens", 2048)
        )
        self._base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        self._client = None
        self._api_key: Optional[str] = None

    @property
    def client(self):
        """Lazy-init the Anthropic client.

        API key resolution (in order):
        1. ANTHROPIC_API_KEY env var
        2. ANTHROPIC_AUTH_TOKEN env var (used by DeepSeek deployments)
        3. llm.api_key in config.json
        """
        if self._client is not None:
            return self._client

        import anthropic

        self._api_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or self._config.get("llm", {}).get("api_key", "")
        )

        if not self._api_key or self._api_key == "sk-xxx":
            raise RuntimeError(
                "LLM API key not configured. Set one of:\n"
                "  - ANTHROPIC_API_KEY\n"
                "  - ANTHROPIC_AUTH_TOKEN\n"
                "  - llm.api_key in config.json"
            )

        client_kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def analyze(
        self,
        composite_score: float,
        composite_grade: str,
        composite_label: str,
        composite_summary: str,
        nb_signals: list,
        margin_signals: list,
        resonance_results: list,
        margin_balance_亿: float = 0,
        short_balance_亿: float = 0,
        margin_buy_億: float = 0,
        top_industries: Optional[list[str]] = None,
        futures_signals: list | None = None,
    ) -> tuple[str, str, str]:
        """Generate macro panorama analysis via LLM.

        Args:
            (all the signal + data inputs for the prompt)

        Returns:
            (analysis_text, source, model) where source is "real" or "fallback"
            and model is the actual model identifier used.
        """
        nb_triggered = sum(1 for s in nb_signals if s.triggered)
        mg_triggered = sum(1 for s in margin_signals if s.triggered)
        res_triggered = sum(1 for r in resonance_results if r.triggered)

        prompt = build_panorama_prompt(
            composite_score=composite_score,
            composite_grade=composite_grade,
            composite_label=composite_label,
            composite_summary=composite_summary,
            nb_signals=nb_signals,
            margin_signals=margin_signals,
            resonance_results=resonance_results,
            margin_balance_亿=margin_balance_亿,
            short_balance_亿=short_balance_亿,
            margin_buy_亿=margin_buy_億,
            top_industries=top_industries,
            futures_signals=futures_signals,
        )

        # Try API call
        try:
            client = self.client
        except RuntimeError as e:
            logger.warning("LLM not available: %s — using fallback", e)
            fallback = _generate_fallback(
                composite_score, composite_label,
                nb_triggered, mg_triggered, res_triggered,
                margin_balance_亿,
            )
            return fallback, "fallback", self._model

        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts: list[str] = []
            for block in resp.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)
            text = "".join(text_parts).strip()

            if not text:
                logger.warning("LLM returned empty response — using fallback")
                fallback = _generate_fallback(
                    composite_score, composite_label,
                    nb_triggered, mg_triggered, res_triggered,
                    margin_balance_亿,
                )
                return fallback, "fallback", self._model

            logger.info("LLM analysis generated (%d chars, model=%s)", len(text), self._model)
            return text, "real", self._model

        except Exception as e:
            logger.warning("LLM API call failed: %s — using fallback", e)
            fallback = _generate_fallback(
                composite_score, composite_label,
                nb_triggered, mg_triggered, res_triggered,
                margin_balance_亿,
            )
            return fallback, "fallback", self._model
