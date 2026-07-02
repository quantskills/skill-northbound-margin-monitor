"""Panorama pipeline: orchestrates data fetch → analysis → report generation.

Usage::

    from core.pipeline import PanoramaPipeline

    pipeline = PanoramaPipeline()
    result = pipeline.run()  # latest trading day
    result = pipeline.run(trade_date="20260630")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .cache import CacheManager
from .data_fetcher import DataFetcher
from .northbound import (
    run_all_detectors as run_nb_detectors,
    compute_composite_score as nb_composite_score,
    get_triggered_signals as nb_triggered,
    get_bullish_signals as nb_bullish,
    get_bearish_signals as nb_bearish,
)
from .margin import (
    run_all_detectors as run_margin_detectors,
    compute_composite_score as margin_composite_score,
    get_triggered_signals as mg_triggered,
    get_bullish_signals as mg_bullish,
    get_bearish_signals as mg_bearish,
)
from .futures import (
    run_all_detectors as run_futures_detectors,
    compute_composite_score as futures_composite_score,
    get_triggered_signals as futures_triggered,
    get_bullish_signals as futures_bullish,
    get_bearish_signals as futures_bearish,
)
from .resonance import (
    analyse_resonance,
    get_triggered_patterns,
    compute_resonance_score,
)
from .scorer import (
    compute_capital_flow_score,
    compute_industry_neutral_ranking,
    rank_stocks_by_margin_balance,
    rank_stocks_by_margin_buy,
)
from .reporter import (
    generate_markdown_report,
    generate_json_output,
    write_report,
)

try:
    from llm.analyst import LLMAnalyst as _LLMAnalyst
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    trade_date: str
    md_path: Path
    json_path: Path
    composite_score: float
    composite_grade: str
    nb_triggered: int
    margin_triggered: int
    resonance_triggered: int
    futures_triggered: int = 0
    llm_analysis: str = ""
    llm_source: str = "none"
    errors: list[str] = field(default_factory=list)


class PanoramaPipeline:
    """Full panorama pipeline: fetch → analyse → report."""

    def __init__(self, config: Optional[dict] = None, cache_root: str | Path = "cache"):
        import json as _json

        self._config = config
        if self._config is None:
            cfg_path = Path(__file__).resolve().parent.parent / "config.json"
            if cfg_path.exists():
                self._config = _json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                self._config = {}

        self._fetcher = DataFetcher(config=self._config)
        self._cache = CacheManager(cache_root=cache_root)
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cleanup_cache(self, keep_days: int = 30) -> int:
        """Remove cached data older than *keep_days*.

        Returns:
            Number of directories removed.
        """
        return self._cache.clear_old(keep_days=keep_days)

    def run(
        self,
        trade_date: Optional[str] = None,
        use_cache: bool = True,
        top_n: int = 20,
    ) -> PipelineResult:
        """Execute the full panorama pipeline.

        Args:
            trade_date: Target trading day (YYYYMMDD). Default: latest.
            use_cache: If True, use cached data when available.
            top_n: Number of stocks in TOP-N lists.

        Returns:
            PipelineResult with output paths and summary stats.
        """
        self._errors.clear()

        # 1. Resolve trade date
        if trade_date is None:
            try:
                self._fetcher.init_api()
                trade_date = self._fetcher.get_last_trade_date()
                logger.info("Using latest trade date: %s", trade_date)
            except Exception as e:
                logger.error("Cannot determine latest trade date: %s", e)
                self._errors.append(f"Date resolution failed: {e}")
                return self._empty_result("unknown")

        logger.info("=" * 60)
        logger.info("PanoramaPipeline starting for %s", trade_date)
        logger.info("=" * 60)

        # 2. Check cache
        if use_cache and self._cache.has(trade_date):
            logger.info("Cache hit for %s — loading cached data", trade_date)
            nb_summary, margin_macro, margin_detail, stock_info, futures_data = self._cache.load(trade_date)
            nb_flow = self._cache.load_nb_flow(trade_date)
            universe = pd.DataFrame()
            fetch_time = self._cache.load_meta(trade_date).get("fetch_time", "cached")
        else:
            # 3. Fetch all data
            try:
                self._fetcher.init_api()
            except Exception as e:
                logger.warning("Pandadata init failed (non-fatal): %s", e)
                self._errors.append(f"Pandadata init: {e}")

            try:
                data = self._fetcher.fetch_all_data(trade_date)
                nb_summary = data["northbound_summary"]
                nb_flow = data["northbound_flow"]
                margin_detail = data["margin_detail"]
                margin_macro = data["margin_macro"]
                futures_data = data.get("futures_data", pd.DataFrame())
                stock_info = data["stock_info"]
                universe = data["universe"]
                fetch_time = data["fetch_time"]
            except Exception as e:
                logger.error("Data fetch failed: %s", e)
                self._errors.append(f"Data fetch: {e}")
                return self._empty_result(trade_date)

            # 4. Save to cache
            if use_cache:
                try:
                    self._cache.save(
                        trade_date,
                        nb_summary,
                        margin_macro,
                        margin_detail,
                        stock_info,
                        northbound_flow_df=nb_flow,
                        futures_df=futures_data,
                    )
                except Exception as e:
                    logger.warning("Cache save failed (non-fatal): %s", e)

        # 5. Run northbound detectors
        logger.info("Running northbound detectors ...")
        nb_signals = run_nb_detectors(
            nb_summary, nb_flow, stock_info, self._config,
        )
        nb_composite = nb_composite_score(nb_signals)

        # 6. Run margin detectors
        logger.info("Running margin detectors ...")
        margin_signals = run_margin_detectors(
            margin_detail, margin_macro, stock_info, self._config,
        )
        margin_composite = margin_composite_score(margin_signals)

        # 6.5. Run futures detectors (3rd dimension)
        logger.info("Running futures detectors ...")
        futures_signals = run_futures_detectors(
            futures_data, self._config,
        )
        futures_composite = futures_composite_score(futures_signals)

        # 7. Resonance analysis
        logger.info("Running resonance analysis ...")
        resonance_results = analyse_resonance(
            nb_composite,
            margin_composite,
            sum(1 for s in nb_signals if s.direction == "bullish"),
            sum(1 for s in nb_signals if s.direction == "bearish"),
            sum(1 for s in margin_signals if s.direction == "bullish"),
            sum(1 for s in margin_signals if s.direction == "bearish"),
            self._config,
        )
        resonance_score = compute_resonance_score(resonance_results, self._config)

        # 8. Composite scoring
        logger.info("Computing composite score ...")
        composite = compute_capital_flow_score(
            nb_composite,
            margin_composite,
            resonance_score,
            sum(1 for s in nb_signals if s.triggered),
            sum(1 for s in margin_signals if s.triggered),
            sum(1 for r in resonance_results if r.triggered),
            self._config,
            futures_composite=futures_composite,
            futures_triggered_count=sum(1 for s in futures_signals if s.triggered),
        )

        # 9. Stock rankings
        logger.info("Ranking stocks ...")
        top_bal = rank_stocks_by_margin_balance(margin_detail, stock_info, top_n)
        top_buy = rank_stocks_by_margin_buy(margin_detail, stock_info, top_n)

        # 9.1. Industry-neutral sector ranking
        industry_ranking = compute_industry_neutral_ranking(margin_detail, stock_info, top_n=10)

        # 9.5. LLM macro analysis
        llm_analysis = ""
        llm_source = "none"
        llm_model = ""
        if _LLM_AVAILABLE:
            logger.info("Running LLM macro analysis ...")
            try:
                # Extract macro summary data for prompt
                mg_bal_亿 = 0.0
                mg_short_億 = 0.0
                mg_buy_億 = 0.0
                if not margin_macro.empty:
                    date_col = "date" if "date" in margin_macro.columns else "日期"
                    if date_col in margin_macro.columns:
                        latest_date = margin_macro[date_col].max()
                        latest_data = margin_macro[margin_macro[date_col] == latest_date]
                    else:
                        latest_data = margin_macro
                    for col in ["margin_balance", "融资余额"]:
                        if col in latest_data.columns:
                            mg_bal_亿 = pd.to_numeric(latest_data[col], errors="coerce").sum() / 1e8
                            break
                    for col in ["short_balance", "融券余额"]:
                        if col in latest_data.columns:
                            mg_short_億 = pd.to_numeric(latest_data[col], errors="coerce").sum() / 1e8
                            break
                    for col in ["buy_on_margin_value", "融资买入额"]:
                        if col in latest_data.columns:
                            mg_buy_億 = pd.to_numeric(latest_data[col], errors="coerce").sum() / 1e8
                            break

                top_industries = []
                if not stock_info.empty and "industry" in stock_info.columns:
                    top_industries = stock_info["industry"].value_counts().head(5).index.tolist()

                llm = _LLMAnalyst(config=self._config)
                llm_analysis, llm_source, llm_model = llm.analyze(
                    composite_score=composite.score,
                    composite_grade=composite.grade,
                    composite_label=composite.label,
                    composite_summary=composite.summary,
                    nb_signals=nb_signals,
                    margin_signals=margin_signals,
                    futures_signals=futures_signals,
                    resonance_results=resonance_results,
                    margin_balance_亿=mg_bal_亿,
                    short_balance_亿=mg_short_億,
                    margin_buy_億=mg_buy_億,
                    top_industries=top_industries,
                )
                logger.info("LLM analysis: model=%s, source=%s, length=%d chars", llm_model, llm_source, len(llm_analysis))
            except Exception as e:
                logger.warning("LLM analysis failed: %s", e)
                llm_analysis = f"（LLM 分析不可用: {e}）"
                llm_source = "error"
                llm_model = ""
        else:
            logger.info("LLM module not available — skipping macro analysis")

        # 10. Generate reports
        logger.info("Generating reports ...")
        markdown = generate_markdown_report(
            trade_date=trade_date,
            nb_signals=nb_signals,
            margin_signals=margin_signals,
            futures_signals=futures_signals,
            resonance_results=resonance_results,
            composite=composite,
            nb_summary=nb_summary,
            margin_detail=margin_detail,
            margin_macro=margin_macro,
            futures_data=futures_data,
            top_margin_balance=top_bal,
            top_margin_buy=top_buy,
            stock_info=stock_info,
            industry_ranking=industry_ranking,
            fetch_time=fetch_time,
            config=self._config,
            llm_analysis=llm_analysis,
            llm_source=llm_source,
            llm_model=llm_model,
        )

        json_data = generate_json_output(
            trade_date=trade_date,
            nb_signals=nb_signals,
            margin_signals=margin_signals,
            futures_signals=futures_signals,
            resonance_results=resonance_results,
            composite=composite,
            top_margin_balance=top_bal,
            top_margin_buy=top_buy,
            fetch_time=fetch_time,
            nb_summary=nb_summary,
            margin_detail=margin_detail,
            margin_macro=margin_macro,
            futures_data=futures_data,
            stock_info=stock_info,
            industry_ranking=industry_ranking,
            config=self._config,
            llm_analysis=llm_analysis,
            llm_source=llm_source,
            llm_model=llm_model,
        )

        md_path, json_path = write_report(
            trade_date, markdown, json_data, config=self._config,
        )

        # 11. Validate reports
        self._validate_reports(md_path, json_path)

        # 12. Result
        n_triggered = sum(1 for s in nb_signals if s.triggered)
        m_triggered = sum(1 for s in margin_signals if s.triggered)
        f_triggered = sum(1 for s in futures_signals if s.triggered)
        r_triggered = sum(1 for r in resonance_results if r.triggered)

        logger.info("=" * 60)
        logger.info(
            "Pipeline complete: score=%.0f (%s), nb=%d, margin=%d, futures=%d, resonance=%d, llm=%s",
            composite.score, composite.grade,
            n_triggered, m_triggered, f_triggered, r_triggered, llm_source,
        )
        logger.info("Markdown: %s", md_path)
        logger.info("JSON:     %s", json_path)
        if self._errors:
            logger.warning("Errors: %s", self._errors)
        logger.info("=" * 60)

        return PipelineResult(
            trade_date=trade_date,
            md_path=md_path,
            json_path=json_path,
            composite_score=composite.score,
            composite_grade=composite.grade,
            nb_triggered=n_triggered,
            margin_triggered=m_triggered,
            futures_triggered=f_triggered,
            resonance_triggered=r_triggered,
            llm_analysis=llm_analysis,
            llm_source=llm_source,
            errors=list(self._errors),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_reports(self, md_path: Path, json_path: Path) -> None:
        """Run post-generation validation on the reports.

        Issues are logged as warnings so they don't block pipeline completion,
        but are visible in the log output for manual review.
        """
        try:
            from scripts.validate_report import validate_md, validate_json
        except ImportError:
            return  # validation script not importable — skip silently

        issues: list[str] = []
        try:
            md_text = md_path.read_text(encoding="utf-8-sig")
            issues.extend(validate_md(md_text))
        except Exception as e:
            logger.warning("Report validation: cannot read markdown — %s", e)

        try:
            import json
            json_data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            issues.extend(validate_json(json_data))
        except Exception as e:
            logger.warning("Report validation: cannot read JSON — %s", e)

        if issues:
            logger.warning("Report validation found %d issue(s):", len(issues))
            for issue in issues[:10]:  # Cap to avoid log spam
                logger.warning("  - %s", issue)
            if len(issues) > 10:
                logger.warning("  ... and %d more", len(issues) - 10)
        else:
            logger.info("Report validation: OK")

    def _empty_result(self, trade_date: str) -> PipelineResult:
        return PipelineResult(
            trade_date=trade_date,
            md_path=Path(""),
            json_path=Path(""),
            composite_score=0.0,
            composite_grade="N/A",
            nb_triggered=0,
            margin_triggered=0,
            resonance_triggered=0,
            errors=list(self._errors),
        )
