"""Tests for Markdown + JSON report generation."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.reporter import (
    generate_markdown_report,
    generate_json_output,
    write_report,
    _safe_fmt,
    _provenance_label,
)
from core.northbound import run_all_detectors as run_nb
from core.northbound import compute_composite_score as nb_score
from core.margin import run_all_detectors as run_mg
from core.margin import compute_composite_score as mg_score
from core.resonance import analyse_resonance, compute_resonance_score
from core.scorer import (
    compute_capital_flow_score,
    rank_stocks_by_margin_balance,
    rank_stocks_by_margin_buy,
)


def _run_pipeline(
    nb_summary, nb_flow, margin_detail, margin_macro, stock_info, config
):
    """Mini pipeline to generate all report inputs."""
    nb_signals = run_nb(nb_summary, nb_flow, stock_info, config)
    mg_signals = run_mg(margin_detail, margin_macro, stock_info, config)
    nb_comp = nb_score(nb_signals)
    mg_comp = mg_score(mg_signals)
    resonance = analyse_resonance(
        nb_comp, mg_comp,
        sum(1 for s in nb_signals if s.direction == "bullish"),
        sum(1 for s in nb_signals if s.direction == "bearish"),
        sum(1 for s in mg_signals if s.direction == "bullish"),
        sum(1 for s in mg_signals if s.direction == "bearish"),
        config,
    )
    res_score = compute_resonance_score(resonance)
    composite = compute_capital_flow_score(
        nb_comp, mg_comp, res_score,
        sum(1 for s in nb_signals if s.triggered),
        sum(1 for s in mg_signals if s.triggered),
        sum(1 for r in resonance if r.triggered),
        config,
    )
    top_bal = rank_stocks_by_margin_balance(margin_detail, stock_info, 10)
    top_buy = rank_stocks_by_margin_buy(margin_detail, stock_info, 10)
    return nb_signals, mg_signals, resonance, composite, top_bal, top_buy


class TestSafeFmt:
    def test_normal_float(self):
        assert _safe_fmt(3.14159, ".2f") == "3.14"

    def test_none(self):
        assert _safe_fmt(None) == "-"

    def test_nan(self):
        import numpy as np
        assert _safe_fmt(np.nan) == "-"

    def test_string(self):
        assert _safe_fmt("hello") == "hello"


class TestProvenanceLabel:
    def test_pandadata(self):
        assert "Pandadata" in _provenance_label("pandadata")

    def test_eastmoney(self):
        assert "东方财富" in _provenance_label("eastmoney")

    def test_degraded(self):
        assert "降级" in _provenance_label("degraded")

    def test_unknown(self):
        assert "未知" in _provenance_label("unknown")

    def test_none(self):
        assert "无数据" in _provenance_label("none")


class TestMarkdownReport:
    def test_generates_valid_markdown(
        self, sample_nb_summary, sample_nb_flow,
        sample_margin_detail, sample_margin_macro,
        sample_stock_info, sample_config,
    ):
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            sample_nb_summary, sample_nb_flow,
            sample_margin_detail, sample_margin_macro,
            sample_stock_info, sample_config,
        )
        md = generate_markdown_report(
            trade_date="20260630",
            nb_signals=nb_s,
            margin_signals=mg_s,
            resonance_results=res,
            composite=comp,
            nb_summary=sample_nb_summary,
            margin_detail=sample_margin_detail,
            margin_macro=sample_margin_macro,
            top_margin_balance=top_bal,
            top_margin_buy=top_buy,
            stock_info=sample_stock_info,
            fetch_time="2026-06-30T15:00:00",
            config=sample_config,
        )
        assert isinstance(md, str)
        assert len(md) > 500
        # Check all 7 sections present
        assert "北向资金概览" in md
        assert "融资融券概览" in md
        assert "共振/背离分析" in md
        assert "综合情绪评估" in md
        assert "板块资金流向" in md
        assert "风险提示" in md
        assert "数据溯源" in md

    def test_handles_empty_data(self, sample_config):
        empty = pd.DataFrame()
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            empty, empty, empty, empty, empty, sample_config,
        )
        md = generate_markdown_report(
            trade_date="20260630",
            nb_signals=nb_s,
            margin_signals=mg_s,
            resonance_results=res,
            composite=comp,
            nb_summary=empty,
            margin_detail=empty,
            margin_macro=empty,
            top_margin_balance=pd.DataFrame(),
            top_margin_buy=pd.DataFrame(),
            stock_info=empty,
            fetch_time="2026-06-30T15:00:00",
            config=sample_config,
        )
        assert isinstance(md, str)
        assert len(md) > 0

    def test_includes_date(self, sample_config):
        empty = pd.DataFrame()
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            empty, empty, empty, empty, empty, sample_config,
        )
        md = generate_markdown_report(
            trade_date="20260715",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            nb_summary=empty, margin_detail=empty, margin_macro=empty,
            top_margin_balance=pd.DataFrame(), top_margin_buy=pd.DataFrame(),
            stock_info=empty,
            fetch_time="2026-07-15T15:00:00",
            config=sample_config,
        )
        assert "2026-07-15" in md

    def test_triggered_signals_marked(self, sample_config):
        """Triggered signals should be marked with **触发**."""
        empty = pd.DataFrame()
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            empty, empty, empty, empty, empty, sample_config,
        )
        md = generate_markdown_report(
            trade_date="20260630",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            nb_summary=empty, margin_detail=empty, margin_macro=empty,
            top_margin_balance=pd.DataFrame(), top_margin_buy=pd.DataFrame(),
            stock_info=empty,
            fetch_time="2026-06-30T15:00:00",
            config=sample_config,
        )
        assert "未触发" in md


class TestJsonOutput:
    def test_generates_valid_json(
        self, sample_nb_summary, sample_nb_flow,
        sample_margin_detail, sample_margin_macro,
        sample_stock_info, sample_config,
    ):
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            sample_nb_summary, sample_nb_flow,
            sample_margin_detail, sample_margin_macro,
            sample_stock_info, sample_config,
        )
        data = generate_json_output(
            trade_date="20260630",
            nb_signals=nb_s,
            margin_signals=mg_s,
            resonance_results=res,
            composite=comp,
            top_margin_balance=top_bal,
            top_margin_buy=top_buy,
            fetch_time="2026-06-30T15:00:00",
            nb_summary=sample_nb_summary,
            margin_detail=sample_margin_detail,
            margin_macro=sample_margin_macro,
            stock_info=sample_stock_info,
            config=sample_config,
        )
        assert isinstance(data, dict)
        # Check structure
        assert "meta" in data
        assert "composite" in data
        assert "northbound" in data
        assert "margin" in data
        assert "resonance" in data
        assert "provenance" in data
        # Verify serializable
        json.dumps(data, ensure_ascii=False)

    def test_meta_fields(self, sample_config):
        empty = pd.DataFrame()
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            empty, empty, empty, empty, empty, sample_config,
        )
        data = generate_json_output(
            trade_date="20260630",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            top_margin_balance=pd.DataFrame(), top_margin_buy=pd.DataFrame(),
            fetch_time="2026-06-30T15:00:00",
            nb_summary=empty, margin_detail=empty, margin_macro=empty,
            stock_info=empty,
            config=sample_config,
        )
        assert data["meta"]["trade_date"] == "20260630"
        assert data["meta"]["generator"] == "skill-northbound-margin-monitor"

    def test_includes_provenance(self, sample_config):
        empty = pd.DataFrame()
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            empty, empty, empty, empty, empty, sample_config,
        )
        data = generate_json_output(
            trade_date="20260630",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            top_margin_balance=pd.DataFrame(), top_margin_buy=pd.DataFrame(),
            fetch_time="2026-06-30T15:00:00",
            nb_summary=empty, margin_detail=empty, margin_macro=empty,
            stock_info=empty,
            config=sample_config,
        )
        assert "northbound_summary" in data["provenance"]
        assert "margin_detail" in data["provenance"]


class TestWriteReport:
    def test_writes_md_and_json(
        self, sample_nb_summary, sample_nb_flow,
        sample_margin_detail, sample_margin_macro,
        sample_stock_info, sample_config,
    ):
        nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
            sample_nb_summary, sample_nb_flow,
            sample_margin_detail, sample_margin_macro,
            sample_stock_info, sample_config,
        )
        md = generate_markdown_report(
            trade_date="20260630",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            nb_summary=sample_nb_summary,
            margin_detail=sample_margin_detail,
            margin_macro=sample_margin_macro,
            top_margin_balance=top_bal, top_margin_buy=top_buy,
            stock_info=sample_stock_info,
            fetch_time="2026-06-30T15:00:00",
            config=sample_config,
        )
        json_data = generate_json_output(
            trade_date="20260630",
            nb_signals=nb_s, margin_signals=mg_s, resonance_results=res,
            composite=comp,
            top_margin_balance=top_bal, top_margin_buy=top_buy,
            fetch_time="2026-06-30T15:00:00",
            nb_summary=sample_nb_summary,
            margin_detail=sample_margin_detail,
            margin_macro=sample_margin_macro,
            stock_info=sample_stock_info,
            config=sample_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            md_path, json_path = write_report(
                "20260630", md, json_data, output_dir=tmp,
            )
            assert md_path.exists()
            assert json_path.exists()
            assert md_path.suffix == ".md"
            assert json_path.suffix == ".json"
            # Check content
            content = md_path.read_text(encoding="utf-8")
            assert "北向资金" in content
            json_content = json.loads(json_path.read_text(encoding="utf-8"))
            assert "composite" in json_content

    def test_output_dir_from_config(self, sample_config):
        with tempfile.TemporaryDirectory() as tmp:
            config = {**sample_config, "output": {"dir": tmp}}
            nb_s, mg_s, res, comp, top_bal, top_buy = _run_pipeline(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), config,
            )
            md = "test"
            js = {"test": True}
            md_path, json_path = write_report(
                "20260630", md, js, config=config,
            )
            assert str(tmp) in str(md_path)
