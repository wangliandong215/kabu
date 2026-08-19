"""
research/fundamentals.py — Category 7 (company quality) best-effort fetch.

engine/fundamental.py already found several moomoo OpenD indicator endpoints
broken on this account/tier (get_rating_change, get_earnings_beat_rank — see
that module's docstring). This module only uses endpoints already proven
working elsewhere in this codebase (get_market_snapshot, same call
data/fetcher.py's get_price() uses) or documented/exampled in
.claude/skills/moomooapi (get_financials_statements). Every field that has
no working data source stays None with fundamentals_status[...] =
"NOT_IMPLEMENTED" — never fabricated. Never raises.

Fundamentals change slowly, so results are disk-cached
(config.MR_FUNDAMENTALS_CACHE_TTL_DAYS, default 7d) — the daily scan
shouldn't re-hit moomoo's valuation/financials endpoints for the same
watchlist stock every day.
"""
import pickle
import time
from pathlib import Path
from typing import Optional

import common
import config

_CACHE_DIR = Path(config.MR_DB_PATH).parent / "fundamentals_cache"

# Income Statement (statement_type=1) line items we look for by keyword,
# case-insensitive substring match against display_name — field IDs aren't
# documented anywhere in this codebase/account, so structure_list is walked
# fresh each call rather than hardcoding IDs that could differ by market.
_REVENUE_KEYWORDS = ("total revenue", "revenue")
_EPS_KEYWORDS = ("diluted eps", "eps", "earnings per share")

# MainIndex (statement_type=4) line items for quality/margin fields.
_ROE_KEYWORDS = ("roe", "return on equity")
_GROSS_MARGIN_KEYWORDS = ("gross margin", "gross profit margin")
_NET_MARGIN_KEYWORDS = ("net margin", "net profit margin")
_DEBT_KEYWORDS = ("debt to asset", "debt ratio", "asset-liability")


def _cache_path(code: str) -> Path:
    return _CACHE_DIR / f"{code.replace('.', '_')}.pkl"


def _load_cache(code: str) -> Optional[dict]:
    path = _cache_path(code)
    if not path.exists():
        return None
    ttl = config.MR_FUNDAMENTALS_CACHE_TTL_DAYS * 86400
    if (time.time() - path.stat().st_mtime) >= ttl:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(code: str, record: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_cache_path(code), "wb") as f:
            pickle.dump(record, f)
    except OSError:
        pass


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None   # filter NaN
    except (TypeError, ValueError):
        return None


def _find_item_value(report_list: list, structure_list: list, keywords: tuple,
                      field: str = "data") -> Optional[float]:
    """Walk the most recent report's item_list for a display_name matching
    any of `keywords` (case-insensitive substring), return item[field]
    ('data' for the raw value, 'yoy' for year-over-year %). None if no
    report/match."""
    if not report_list:
        return None
    id_to_name = {e["field_id"]: (e.get("display_name") or "").lower()
                  for e in structure_list}
    report = report_list[0]
    for item in report.get("item_list", []):
        name = id_to_name.get(item.get("field_id"), "")
        if any(kw in name for kw in keywords):
            return _safe_float(item.get(field))
    return None


def _fetch_snapshot(ctx, code: str, status: dict) -> dict:
    from moomoo import RET_OK
    try:
        ret, data = ctx.get_market_snapshot([code])
        if ret == RET_OK and data is not None and len(data):
            row = data.iloc[0]
            status["snapshot"] = "OK"
            return {
                "market_cap": _safe_float(row.get("total_market_val")),
                "pe_ttm": _safe_float(row.get("pe_ttm_ratio")),
            }
    except Exception:
        pass
    status["snapshot"] = "UNAVAILABLE"
    return {"market_cap": None, "pe_ttm": None}


def _fetch_income_growth(ctx, code: str, status: dict) -> dict:
    from moomoo import RET_OK
    try:
        ret, data = ctx.get_financials_statements(
            code, statement_type=1, financial_type=7, num=2)
        if ret == RET_OK and isinstance(data, dict) and data.get("report_list"):
            revenue_growth = _find_item_value(
                data["report_list"], data.get("structure_list", []),
                _REVENUE_KEYWORDS, field="yoy")
            eps_growth = _find_item_value(
                data["report_list"], data.get("structure_list", []),
                _EPS_KEYWORDS, field="yoy")
            status["income_growth"] = "OK"
            return {"revenue_growth_yoy": revenue_growth, "eps_growth_yoy": eps_growth}
    except Exception:
        pass
    status["income_growth"] = "UNAVAILABLE"
    return {"revenue_growth_yoy": None, "eps_growth_yoy": None}


def _fetch_main_index(ctx, code: str, status: dict) -> dict:
    from moomoo import RET_OK
    try:
        ret, data = ctx.get_financials_statements(
            code, statement_type=4, financial_type=7, num=2)
        if ret == RET_OK and isinstance(data, dict) and data.get("report_list"):
            reports, structure = data["report_list"], data.get("structure_list", [])
            roe = _find_item_value(reports, structure, _ROE_KEYWORDS)
            gross_margin = _find_item_value(reports, structure, _GROSS_MARGIN_KEYWORDS)
            net_margin = _find_item_value(reports, structure, _NET_MARGIN_KEYWORDS)
            debt_ratio = _find_item_value(reports, structure, _DEBT_KEYWORDS)
            status["main_index"] = "OK"
            return {"roe": roe, "profit_margin": net_margin,
                    "gross_margin": gross_margin, "debt_ratio": debt_ratio}
    except Exception:
        pass
    status["main_index"] = "UNAVAILABLE"
    return {"roe": None, "profit_margin": None, "gross_margin": None, "debt_ratio": None}


def fetch_fundamentals(code: str) -> dict:
    """Best-effort company-quality snapshot for `code`. Never raises —
    returns a dict with whatever subset was obtainable, remaining fields
    None, plus a fundamentals_status dict (per-source OK/UNAVAILABLE/
    NOT_IMPLEMENTED) so callers/analysis can tell "genuinely zero" apart
    from "couldn't fetch this run". Disk-cached (see module docstring)."""
    cached = _load_cache(code)
    if cached is not None:
        return cached

    status = {}
    record = {
        "market_cap": None, "pe_ttm": None,
        "revenue_growth_yoy": None, "eps_growth_yoy": None,
        "roe": None, "profit_margin": None, "gross_margin": None,
        "debt_ratio": None,
        "roic": None, "free_cash_flow": None, "total_debt": None,
        "earnings_estimate_revision": None,
    }
    status["roic"] = "NOT_IMPLEMENTED"
    status["free_cash_flow"] = "NOT_IMPLEMENTED"
    status["total_debt"] = "NOT_IMPLEMENTED"
    status["earnings_estimate_revision"] = "NOT_IMPLEMENTED"

    ctx = None
    try:
        ctx = common.make_quote_ctx()
        record.update(_fetch_snapshot(ctx, code, status))
        record.update(_fetch_income_growth(ctx, code, status))
        record.update(_fetch_main_index(ctx, code, status))
    except Exception:
        pass
    finally:
        common.safe_close(ctx)

    record["sector"] = config.SECTOR_MAP.get(code)
    record["fundamentals_status"] = status
    _save_cache(code, record)
    return record
