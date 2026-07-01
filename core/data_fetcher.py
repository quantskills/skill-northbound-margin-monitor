"""Data fetcher: Pandadata (margin + stock info) + AKShare/East Money (north-bound).

Data source strategy (verified 2026-06-30):
  - Margin trading:  Pandadata ``get_margin`` (primary) → AKShare east money (fallback)
  - North-bound:     AKShare ``stock_hsgt_hist_em`` (primary) → fund_flow_summary (degraded)
  - Stock info:      Pandadata ``get_trade_list`` + ``get_stock_detail``

Every DataFrame carries ``df.attrs["source"]`` provenance metadata.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Load .env file if present (for DEFAULT_USERNAME, DEFAULT_PASSWORD, etc.)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

logger = logging.getLogger(__name__)

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load config.json with in-memory caching to avoid repeated disk reads."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if config_path.exists():
        _config_cache = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        _config_cache = {}
    return _config_cache


def _retry_api_call(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    description: str = "API call",
    **kwargs,
):
    """Call *func* with retry and exponential backoff.

    Args:
        func: Callable to invoke.
        max_retries: Maximum retry attempts (3 = 1 initial + 2 retries).
        base_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for successive retry delays.
        description: Human-readable label for log messages.
    """
    import time as _time

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description, attempt + 1, max_retries + 1, e, delay,
                )
                _time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    description, max_retries + 1, e,
                )
                raise


class DataFetcher:
    """Unified data acquisition for the northbound + margin panorama monitor.

    Pandadata is used for margin trading data and stock reference info.
    AKShare (wrapping East Money) is used for north-bound capital flow data
    because Pandadata ``get_hsgt_hold`` is stale (last data: 2025-06-30).
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._initialized = False

    # ------------------------------------------------------------------
    # Pandadata API init
    # ------------------------------------------------------------------

    def init_api(self) -> None:
        """Log into Pandadata service.

        Credentials are resolved in order:
        1. Environment variables: DEFAULT_USERNAME / DEFAULT_PASSWORD
        2. config.json: pandadata.username / pandadata.password

        Note: username must use "86" prefix (e.g. 8618046753943).
        """
        if self._initialized:
            return

        try:
            import panda_data as pdd
        except ImportError:
            raise RuntimeError(
                "panda_data SDK not installed. Install with: pip install panda_data>=0.0.9"
            )

        pd_cfg = self._config.get("pandadata", {})

        username = os.getenv("DEFAULT_USERNAME") or pd_cfg.get("username", "")
        password = os.getenv("DEFAULT_PASSWORD") or pd_cfg.get("password", "")
        base_url = pd_cfg.get("base_url", "http://pandadata.pandaaiquant.com")

        if username and not username.startswith("86"):
            username = "86" + username
            logger.info("Auto-added 86 prefix: %s", username)

        if not username or not password:
            raise RuntimeError(
                "Pandadata credentials not configured. Set either:\n"
                "  - Environment variables: DEFAULT_USERNAME / DEFAULT_PASSWORD\n"
                "  - config.json: pandadata.username / pandadata.password\n"
                "Note: username must be 86+phone (e.g. 8618046753943)."
            )

        pdd.init_token(username=username, password=password, base_url=base_url)
        self._initialized = True
        logger.info("Pandadata API initialized (base_url=%s)", base_url)

    # ------------------------------------------------------------------
    # Trading calendar (Pandadata)
    # ------------------------------------------------------------------

    def get_last_trade_date(self) -> str:
        """Return the latest completed A-share trading day as 'YYYYMMDD'."""
        import panda_data as pdd
        return pdd.get_last_trade_date(exchange="sh")

    def is_trading_day(self, date_str: str) -> bool:
        """Check if *date_str* is an A-share trading day."""
        import panda_data as pdd
        cal = pdd.get_trade_cal(
            start_date=date_str, end_date=date_str, exchange="sh"
        )
        if cal.empty:
            return False
        val = cal.iloc[0].get("is_trade", cal.iloc[0].get("is_trading_day", "0"))
        return val in ("1", 1, True)

    # ------------------------------------------------------------------
    # Stock universe (Pandadata)
    # ------------------------------------------------------------------

    def fetch_stock_universe(self, date_str: str) -> pd.DataFrame:
        """Return tradable A-share stocks on *date_str* via Pandadata.

        Columns include: symbol, name, market_cap, industry, list_status.
        """
        import panda_data as pdd
        raw = _retry_api_call(
            pdd.get_trade_list,
            date=date_str,
            description="Stock universe",
        )
        if raw.empty:
            return raw
        raw.attrs["source"] = "pandadata"
        return raw

    def fetch_stock_details_batch(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch stock basic info: name, industry, list_status via Pandadata.

        Chunks of 200 symbols per request.
        Returns DataFrame with columns: symbol, name, industry, list_status.
        """
        if not symbols:
            return pd.DataFrame()

        import panda_data as pdd

        chunk_size = 200
        frames: list[pd.DataFrame] = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                df = _retry_api_call(
                    pdd.get_stock_detail,
                    symbol=chunk,
                    description=f"Stock detail chunk {i // chunk_size}",
                )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(
                    "Stock detail fetch failed for chunk %d: %s", i // chunk_size, e
                )

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        # Remap columns to our expected names
        col_map = {"sector_code_name": "industry", "status": "list_status"}
        for old, new in col_map.items():
            if old in result.columns and new not in result.columns:
                result[new] = result[old]

        for col in ["name", "industry", "list_status"]:
            if col not in result.columns:
                result[col] = "未知"

        keep = ["symbol", "name", "industry", "list_status"]
        result = result[[c for c in keep if c in result.columns]]
        result.attrs["source"] = "pandadata"
        logger.info("Stock details: %d stocks", len(result))
        return result

    def fetch_stock_universe_with_info(self, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch stock universe + detail info for *date_str*.

        Returns:
            (universe_df, info_df)
        """
        universe = self.fetch_stock_universe(date_str)
        if universe.empty:
            logger.warning("Empty stock universe for %s", date_str)
            return pd.DataFrame(), pd.DataFrame()

        if "symbol" in universe.columns:
            symbols = universe["symbol"].dropna().unique().tolist()
        else:
            symbols = universe.iloc[:, 0].dropna().unique().tolist()

        logger.info("Universe: %d stocks", len(symbols))
        info_df = self.fetch_stock_details_batch(symbols)
        return universe, info_df

    # ------------------------------------------------------------------
    # Margin data — Pandadata (primary)
    # ------------------------------------------------------------------

    def fetch_margin_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch margin trading data for *symbols* via Pandadata ``get_margin``.

        Chunks of 200 symbols, 4 ThreadPoolExecutor workers.
        Columns: symbol, date, margin_balance, margin_repayment, short_balance,
        short_sell_quantity, buy_on_margin_value, total_balance, margin_type,
        short_repayment_quantity, short_balance_quantity.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not symbols:
            return pd.DataFrame()

        import panda_data as pdd

        chunk_size = 200
        frames: list[pd.DataFrame] = []

        def _fetch_chunk(chunk_symbols: list[str], chunk_idx: int) -> pd.DataFrame:
            try:
                df = _retry_api_call(
                    pdd.get_margin,
                    symbol=chunk_symbols,
                    start_date=start_date,
                    end_date=end_date,
                    description=f"Margin chunk {chunk_idx}",
                )
                if not df.empty:
                    return df
                return pd.DataFrame()
            except Exception as e:
                logger.warning(
                    "Margin fetch failed for chunk %d after retries: %s", chunk_idx, e
                )
                return pd.DataFrame()

        chunks = [
            (symbols[i : i + chunk_size], i // chunk_size)
            for i in range(0, len(symbols), chunk_size)
        ]

        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_chunk, chunk, idx): idx
                for chunk, idx in chunks
            }
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    frames.append(df)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.attrs["source"] = "pandadata"
            logger.info("Pandadata margin: %d rows for %d stocks", len(result), len(symbols))
            return result

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Margin data — AKShare / East Money (fallback)
    # ------------------------------------------------------------------

    def _fetch_margin_detail_em(self, trade_date: str) -> pd.DataFrame:
        """AKShare fallback: per-stock margin detail from SSE + SZSE.

        Returns combined DataFrame with ~4000 stocks.
        Columns are normalised to match Pandadata schema as closely as possible.
        """
        import akshare as ak

        frames: list[pd.DataFrame] = []
        for market, fetch_fn in [
            ("sh", ak.stock_margin_detail_sse),
            ("sz", ak.stock_margin_detail_szse),
        ]:
            try:
                df = _retry_api_call(
                    fetch_fn,
                    date=trade_date,
                    max_retries=2,
                    description=f"Margin detail {market.upper()} (EM fallback)",
                )
                if df is not None and not df.empty and len(df.columns) > 0:
                    df = df.copy()
                    df["market"] = market
                    frames.append(df)
                    logger.info(
                        "EM margin detail %s: %d stocks", market.upper(), len(df)
                    )
                else:
                    logger.warning(
                        "EM margin detail %s: empty response (data not yet available for %s)",
                        market.upper(), trade_date,
                    )
            except Exception as e:
                logger.warning("EM margin detail %s fallback failed: %s", market.upper(), e)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # Normalise column names: AKShare uses Chinese column names
        _col_map = {
            "股票代码": "symbol",
            "股票名称": "name",
            "融资余额": "margin_balance",
            "融资买入额": "buy_on_margin_value",
            "融资偿还额": "margin_repayment",
            "融券余量": "short_balance_quantity",
            "融券卖出量": "short_sell_quantity",
            "融券偿还量": "short_repayment_quantity",
            "融券余额": "short_balance",
            "日期": "date",
        }
        result = result.rename(columns=_col_map)
        # Add missing columns with NaN
        for col in ["margin_type", "total_balance"]:
            if col not in result.columns:
                result[col] = None
        result.attrs["source"] = "eastmoney"
        return result

    def _fetch_margin_macro_em(self) -> pd.DataFrame:
        """AKShare macro margin totals (SH + SZ aggregate history).

        Returns DataFrame with columns: date, margin_balance, short_balance, ...
        for both Shanghai and Shenzhen markets.
        """
        import akshare as ak

        rows: list[dict] = []
        for market, fetch_fn in [
            ("sh", ak.macro_china_market_margin_sh),
            ("sz", ak.macro_china_market_margin_sz),
        ]:
            try:
                df = _retry_api_call(
                    fetch_fn,
                    max_retries=2,
                    description=f"Macro margin {market.upper()} (EM)",
                )
                if df is not None and not df.empty:
                    # AKShare macro columns are Chinese
                    _col_map = {
                        "融资余额": "margin_balance",
                        "融资买入额": "buy_on_margin_value",
                        "融券余额": "short_balance",
                        "融券卖出量": "short_sell_quantity",
                        "日期": "date",
                        "市场": "market",
                    }
                    df = df.rename(columns={
                        k: v for k, v in _col_map.items() if k in df.columns
                    })
                    df["market"] = market
                    rows.append(df)
                    logger.info("EM macro margin %s: %d rows", market.upper(), len(df))
            except Exception as e:
                logger.warning("EM macro margin %s failed: %s", market.upper(), e)

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        result.attrs["source"] = "eastmoney"
        return result

    # ------------------------------------------------------------------
    # Margin data — high-level with fallback
    # ------------------------------------------------------------------

    def fetch_margin_data(
        self,
        symbols: list[str],
        trade_date: str,
        lookback_days: int = 30,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch margin trading data with Pandadata-primary → AKShare-fallback.

        Returns:
            (margin_detail_df, margin_macro_df)

        margin_detail_df: per-stock margin data for the most recent date.
        margin_macro_df: aggregate SH+SZ margin history (for trend analysis).

        Data source is recorded in ``df.attrs["source"]``.
        """
        dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = dt - timedelta(days=lookback_days * 2)
        start_date = start_dt.strftime("%Y%m%d")

        margin_detail = pd.DataFrame()
        margin_macro = pd.DataFrame()

        # --- Primary: Pandadata ---
        if self._initialized and symbols:
            logger.info("Fetching margin data via Pandadata (primary) ...")
            try:
                margin_detail = self.fetch_margin_batch(symbols, start_date, trade_date)
            except Exception as e:
                logger.warning("Pandadata margin batch failed: %s", e)

        # --- Fallback: AKShare / East Money ---
        if margin_detail.empty:
            logger.info("Pandadata margin unavailable — falling back to East Money")
            margin_detail = self._fetch_margin_detail_em(trade_date)
            # Macro data is only available via East Money
            margin_macro = self._fetch_margin_macro_em()
        else:
            # Still fetch macro for trend analysis (East Money macro is reliable)
            margin_macro = self._fetch_margin_macro_em()

        if margin_detail.empty:
            logger.warning("No margin detail data available from any source")
        else:
            logger.info(
                "Margin detail: %d rows [source=%s]",
                len(margin_detail),
                margin_detail.attrs.get("source", "unknown"),
            )

        return margin_detail, margin_macro

    # ------------------------------------------------------------------
    # North-bound data — AKShare / East Money (primary)
    # ------------------------------------------------------------------

    def _fetch_northbound_summary_em(self) -> pd.DataFrame:
        """AKShare ``stock_hsgt_hist_em`` — north-bound daily summary.

        Returns historical daily data with remapped English column names:
        date, market_value (持股市值), net_buy_amount (当日成交净买额),
        csi300 (沪深300), csi300_change (沪深300-涨跌幅), etc.

        Note: ``net_buy_amount`` has been NaN for ~2 weeks on this endpoint.
        ``market_value`` is 0 for the last ~3 months.
        The pipeline should prefer flow-direction data for recent dates.
        """
        import akshare as ak

        try:
            df = _retry_api_call(
                ak.stock_hsgt_hist_em,
                max_retries=2,
                description="Northbound summary (EM)",
            )
            if df is not None and not df.empty:
                df = df.copy()
                df.attrs["source"] = "eastmoney"
                # Remap Chinese column names to English
                _col_map = {
                    "日期": "date_raw",
                    "当日成交净买额": "net_buy_amount",
                    "买入成交额": "buy_turnover",
                    "卖出成交额": "sell_turnover",
                    "历史累计净买额": "cumulative_net_buy",
                    "当日资金流入": "daily_inflow",
                    "当日余额": "daily_balance",
                    "持股市值": "market_value",
                    "领涨股": "top_gainer_name",
                    "领涨股-涨跌幅": "top_gainer_pct",
                    "沪深300": "csi300",
                    "沪深300-涨跌幅": "csi300_change",
                    "领涨股-代码": "top_gainer_code",
                }
                df = df.rename(columns=_col_map)
                # Build standardised date column
                date_col = "date_raw" if "date_raw" in df.columns else "date"
                df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d")
                logger.info("Northbound summary: %d rows", len(df))
                return df
        except Exception as e:
            logger.warning("Northbound summary (EM) failed: %s", e)

        return pd.DataFrame()

    def _fetch_northbound_flow_direction_em(self) -> pd.DataFrame:
        """AKShare ``stock_hsgt_fund_flow_summary_em`` — flow direction by market.

        Returns 4 rows: 沪股通/深股通 × 资金流向(流入/流出).
        Columns include market direction, index performance, fund flow indicators.
        This is the reliable source for recent north-bound flow direction.
        """
        import akshare as ak

        try:
            df = _retry_api_call(
                ak.stock_hsgt_fund_flow_summary_em,
                max_retries=2,
                description="Northbound flow direction (EM)",
            )
            if df is not None and not df.empty:
                df.attrs["source"] = "eastmoney"
                logger.info("Northbound flow direction: %d rows", len(df))
                return df
        except Exception as e:
            logger.warning("Northbound flow direction (EM) failed: %s", e)

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # North-bound data — high-level with graceful degradation
    # ------------------------------------------------------------------

    def fetch_northbound_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch north-bound capital flow data from East Money.

        Primary: ``stock_hsgt_hist_em`` (daily summary history).
        Supplemental: ``stock_hsgt_fund_flow_summary_em`` (flow direction).

        Graceful degradation:
        1. Both succeed → full data
        2. Only summary succeeds → summary only, flag degraded
        3. Only flow direction succeeds → flow direction only, flag degraded
        4. Both fail → empty DataFrames with source="degraded"

        Returns:
            (summary_df, flow_direction_df)
        """
        summary_df = self._fetch_northbound_summary_em()
        flow_df = self._fetch_northbound_flow_direction_em()

        if summary_df.empty and flow_df.empty:
            logger.warning(
                "No northbound data available — all sources failed. "
                "Report will show 'data unavailable' for northbound section."
            )
            empty = pd.DataFrame()
            empty.attrs["source"] = "degraded"
            return empty, empty

        if summary_df.empty:
            logger.warning("Northbound summary unavailable — using flow direction only")
            flow_df.attrs["source"] = "degraded"
        elif flow_df.empty:
            logger.warning("Northbound flow direction unavailable — using summary only")
            summary_df.attrs["source"] = "degraded"

        return summary_df, flow_df

    # ------------------------------------------------------------------
    # Index futures data — AKShare / East Money
    # ------------------------------------------------------------------

    def _fetch_index_futures_em(self) -> pd.DataFrame:
        """Fetch stock index futures daily data from Sina + AKShare.

        Uses ``futures_main_sina`` for continuous-contract futures OHLCV data
        and ``stock_zh_index_daily`` for the underlying spot index prices.
        Merges both to produce a normalised DataFrame with columns:
        date, close, spot_close, open_interest, volume, index_name.

        Falls back gracefully if data is unavailable.
        """
        import akshare as ak

        # Mapping: futures symbol → (label, spot-index symbol)
        INDEX_MAP = {
            "IF0": ("CSI300", "sh000300"),
            "IH0": ("SSE50", "sh000016"),
            "IC0": ("CSI500", "sh000905"),
        }

        frames: list[pd.DataFrame] = []

        for fut_sym, (label, spot_sym) in INDEX_MAP.items():
            try:
                # 1. Fetch futures daily data (main continuous contract)
                fut_df = _retry_api_call(
                    ak.futures_main_sina,
                    symbol=fut_sym,
                    max_retries=2,
                    description=f"Futures {fut_sym} (Sina)",
                )
                if fut_df is None or fut_df.empty:
                    logger.debug("Futures %s: no data returned", fut_sym)
                    continue

                fut_df = fut_df.copy()

                # Normalise Chinese column names → English
                _fut_col_map = {
                    "日期": "date",
                    "收盘价": "close",
                    "开盘价": "open",
                    "最高价": "high",
                    "最低价": "low",
                    "成交量": "volume",
                    "持仓量": "open_interest",
                }
                fut_df = fut_df.rename(columns=_fut_col_map)
                # Keep only the columns we need
                keep_cols = ["date", "close", "open", "high", "low", "volume", "open_interest"]
                fut_df = fut_df[[c for c in keep_cols if c in fut_df.columns]].copy()

                # 2. Fetch spot index daily data
                spot_df = _retry_api_call(
                    ak.stock_zh_index_daily,
                    symbol=spot_sym,
                    max_retries=2,
                    description=f"Spot index {spot_sym}",
                )
                if spot_df is None or spot_df.empty:
                    logger.debug("Spot index %s: no data returned", spot_sym)
                    continue

                spot_df = spot_df.copy()
                spot_col_map = {
                    "date": "date",
                    "close": "spot_close",
                    "open": "spot_open",
                    "high": "spot_high",
                    "low": "spot_low",
                }
                spot_df = spot_df.rename(columns=spot_col_map)
                spot_keep = ["date", "spot_close"]
                spot_df = spot_df[[c for c in spot_keep if c in spot_df.columns]].copy()

                # 3. Merge futures + spot on date
                fut_df["date"] = pd.to_datetime(fut_df["date"])
                spot_df["date"] = pd.to_datetime(spot_df["date"])
                merged = fut_df.merge(spot_df, on="date", how="inner")
                merged = merged.sort_values("date")

                if merged.empty:
                    logger.debug("Futures %s: merge with spot resulted in 0 rows", fut_sym)
                    continue

                # Convert numeric columns
                for col in ["close", "open_interest", "volume", "spot_close"]:
                    if col in merged.columns:
                        merged[col] = pd.to_numeric(merged[col], errors="coerce")

                merged["date"] = merged["date"].dt.strftime("%Y%m%d")
                merged["index_name"] = label
                merged["contract"] = fut_sym

                frames.append(merged)
                logger.info(
                    "Futures %s (%s): %d rows, spot %s",
                    fut_sym, label, len(merged), spot_sym,
                )
            except Exception as e:
                logger.debug("Index futures %s (%s) unavailable: %s", fut_sym, label, e)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.attrs["source"] = "sina_eastmoney"
            logger.info(
                "Index futures: %d rows across %d indices",
                len(result),
                result["index_name"].nunique() if "index_name" in result.columns else 0,
            )
            return result

        return pd.DataFrame()

    def fetch_futures_data(self) -> pd.DataFrame:
        """Fetch stock index futures data with graceful degradation.

        Returns:
            DataFrame with columns: date, contract, close, spot_close,
            open_interest, volume, index_name.
            Empty DataFrame with source="degraded" if all sources fail.
        """
        logger.info("Fetching index futures data ...")
        df = self._fetch_index_futures_em()

        if df.empty:
            logger.warning(
                "No index futures data available — futures signals will be neutral. "
                "Install akshare and ensure network access for futures data."
            )
            empty = pd.DataFrame()
            empty.attrs["source"] = "degraded"
            return empty

        logger.info(
            "Futures data: %d rows [source=%s]",
            len(df), df.attrs.get("source", "unknown"),
        )
        return df

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def fetch_all_data(
        self,
        trade_date: str,
        lookback_days: int = 60,
    ) -> dict:
        """Orchestrate all data fetching for the panorama pipeline.

        Args:
            trade_date: Target trading day in YYYYMMDD format.
            lookback_days: Calendar days to look back for margin trend data.

        Returns:
            dict with keys:
              - universe: stock universe DataFrame
              - stock_info: stock detail info DataFrame
              - northbound_summary: northbound daily summary
              - northbound_flow: northbound flow direction
              - margin_detail: per-stock margin data
              - margin_macro: aggregate margin history
              - trade_date: the trade date used
              - fetch_time: ISO timestamp of fetch
        """
        logger.info("=" * 60)
        logger.info("DataFetcher: starting data acquisition for %s", trade_date)
        logger.info("=" * 60)

        # 1. Stock universe + info (Pandadata)
        universe, stock_info = self.fetch_stock_universe_with_info(trade_date)

        if "symbol" in universe.columns:
            symbols = universe["symbol"].dropna().unique().tolist()
        elif not universe.empty:
            symbols = universe.iloc[:, 0].dropna().unique().tolist()
        else:
            symbols = []

        # 2. Northbound data (East Money via AKShare)
        logger.info("Fetching northbound data (East Money) ...")
        nb_summary, nb_flow = self.fetch_northbound_data()

        # 3. Margin data (Pandadata primary → East Money fallback)
        logger.info("Fetching margin data ...")
        margin_detail, margin_macro = self.fetch_margin_data(
            symbols, trade_date, lookback_days
        )

        # 4. Index futures data (East Money via AKShare)
        logger.info("Fetching index futures data ...")
        futures_data = self.fetch_futures_data()

        fetch_time = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info(
            "Data acquisition complete: universe=%d, stock_info=%d, "
            "nb_summary=%d, nb_flow=%d, margin_detail=%d, margin_macro=%d, "
            "futures=%d",
            len(universe), len(stock_info),
            len(nb_summary), len(nb_flow),
            len(margin_detail), len(margin_macro),
            len(futures_data),
        )
        logger.info("=" * 60)

        return {
            "universe": universe,
            "stock_info": stock_info,
            "northbound_summary": nb_summary,
            "northbound_flow": nb_flow,
            "margin_detail": margin_detail,
            "margin_macro": margin_macro,
            "futures_data": futures_data,
            "trade_date": trade_date,
            "fetch_time": fetch_time,
        }
