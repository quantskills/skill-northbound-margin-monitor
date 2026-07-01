"""Local Parquet cache for north-bound, margin, and stock info data.

Avoids re-fetching data on every run during the same day.
Cache is keyed by trade date (YYYYMMDD).

Cache directory::

    cache/
      20260630/
        northbound.parquet
        margin.parquet
        margin_detail.parquet
        stock_info.parquet
        .cache_meta.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

META_FILENAME = ".cache_meta.json"


class CacheManager:
    """Date-keyed Parquet cache for panorama monitor data."""

    def __init__(self, cache_root: str | Path = "cache") -> None:
        self._root = Path(cache_root)

    # -- public API --------------------------------------------------

    def has(self, trade_date: str) -> bool:
        """Check if cached data exists for *trade_date*."""
        if not self._root.exists():
            return False
        date_dir = self._root / trade_date
        return date_dir.is_dir() and (date_dir / "northbound.parquet").exists()

    def load(self, trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached (northbound, margin, margin_detail, stock_info, futures) for *trade_date*.

        Returns empty DataFrames on corrupt or missing cache files
        rather than crashing the pipeline.
        """
        date_dir = self._root / trade_date

        def _safe_read(name: str) -> pd.DataFrame:
            try:
                return pd.read_parquet(date_dir / name)
            except Exception:
                logger.warning("Corrupt or unreadable %s cache for %s, ignoring", name, trade_date)
                return pd.DataFrame()

        nb = _safe_read("northbound.parquet")
        mg = _safe_read("margin.parquet")
        md = _safe_read("margin_detail.parquet")
        info = _safe_read("stock_info.parquet")
        fut = _safe_read("futures.parquet")
        logger.info("Cache hit: %s (%d northbound rows)", trade_date, len(nb))
        return nb, mg, md, info, fut

    def load_meta(self, trade_date: str) -> dict:
        """Load cache metadata without reading parquet files."""
        date_dir = self._root / trade_date
        meta_path = date_dir / META_FILENAME
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save(
        self,
        trade_date: str,
        northbound_df: pd.DataFrame,
        margin_df: pd.DataFrame,
        margin_detail_df: pd.DataFrame,
        stock_info_df: pd.DataFrame,
        northbound_flow_df: pd.DataFrame | None = None,
        futures_df: pd.DataFrame | None = None,
    ) -> None:
        """Persist data to cache."""
        date_dir = self._root / trade_date
        date_dir.mkdir(parents=True, exist_ok=True)

        def _safe_parquet(df: pd.DataFrame, path: Path) -> None:
            """Write parquet, cleaning NaN strings in object columns first."""
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == object:
                    # Replace string "NaN" / "nan" with None (parquet-compatible)
                    mask = df[col].astype(str).str.lower().isin(["nan", "<na>"])
                    df.loc[mask, col] = None
            df.to_parquet(path, index=False)

        _safe_parquet(northbound_df, date_dir / "northbound.parquet")
        _safe_parquet(margin_df, date_dir / "margin.parquet")
        _safe_parquet(margin_detail_df, date_dir / "margin_detail.parquet")
        _safe_parquet(stock_info_df, date_dir / "stock_info.parquet")
        if northbound_flow_df is not None and not northbound_flow_df.empty:
            _safe_parquet(northbound_flow_df, date_dir / "northbound_flow.parquet")
        if futures_df is not None and not futures_df.empty:
            _safe_parquet(futures_df, date_dir / "futures.parquet")

        nb_source = northbound_df.attrs.get("source", "unknown") if not northbound_df.empty else "none"
        mg_source = margin_df.attrs.get("source", "unknown") if not margin_df.empty else "none"
        meta = {
            "trade_date": trade_date,
            "northbound_rows": len(northbound_df),
            "margin_rows": len(margin_df),
            "margin_detail_rows": len(margin_detail_df),
            "stock_info_rows": len(stock_info_df),
            "northbound_source": nb_source,
            "margin_source": mg_source,
        }
        (date_dir / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Cache saved: %s", date_dir)

    def load_nb_flow(self, trade_date: str) -> pd.DataFrame:
        """Load cached northbound flow data for *trade_date*."""
        date_dir = self._root / trade_date
        flow_path = date_dir / "northbound_flow.parquet"
        if flow_path.exists():
            try:
                return pd.read_parquet(flow_path)
            except Exception:
                logger.warning("Corrupt northbound_flow cache for %s", trade_date)
        return pd.DataFrame()

    def clear_old(self, keep_days: int = 30) -> int:
        """Remove cached dates older than *keep_days* calendar days.

        Returns:
            Number of directories removed.
        """
        from datetime import datetime, timedelta

        if not self._root.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            try:
                dt = datetime.strptime(child.name, "%Y%m%d")
                if dt < cutoff:
                    for f in child.iterdir():
                        f.unlink()
                    child.rmdir()
                    removed += 1
                    logger.info("Cache cleanup: removed %s", child.name)
            except ValueError:
                continue
        return removed

    @property
    def cached_dates(self) -> list[str]:
        """Return sorted list of cached dates."""
        if not self._root.exists():
            return []
        dates = []
        for child in self._root.iterdir():
            if child.is_dir() and (child / "northbound.parquet").exists():
                dates.append(child.name)
        return sorted(dates)
