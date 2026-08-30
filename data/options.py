"""
data/options.py — Options market research data via moomoo OpenD.

Thin data-acquisition layer only — no risk judgment, no trading decisions,
never imported by engine/runner.py or risk/*. Consumed by
research/collect_market_features.py.

Field names confirmed against the installed moomoo_api SDK source
(get_option_underlying_overview / get_option_market_statistic docstrings)
at implementation time — see each function's docstring for the exact
columns each OpenD call returns.

Timezone note (v2.11.1 audit): get_option_underlying_overview() returns a
same-moment snapshot with no embedded date field at all — its `iv`/`hv_*`/
volume figures are "as of the call", dated entirely by this module's own
caller-supplied `observed_at`. get_option_market_statistic()'s `time` field
is a plain 'YYYY-MM-DD' string (US trading day, matching how the earlier
real-data check showed rows for '2026-08-28' — the most recently completed
US session at query time), no separate timezone conversion needed.
"""
from typing import List, Optional

import common


def get_underlying_overview(codes: List[str]) -> List[dict]:
    """
    Per-symbol IV/HV/volume snapshot via get_option_underlying_overview().
    Row shape: {code, name, call_volume, put_volume, call_open_interest,
    put_open_interest, iv, iv_rank, iv_percentile, pre_iv, hv_30d,
    hv_30d_percentile, hv_60d, hv_60d_percentile, hv_90d, hv_90d_percentile,
    hv_120d, hv_120d_percentile, hv_365d, hv_365d_percentile}.

    put_call_ratio (per symbol) = put_volume / call_volume — not returned
    directly by this call, so callers should derive it from the two volume
    fields (call_volume == 0 -> None, matching how the market-wide ratio in
    get_option_market_statistic() reports N/A).
    """
    return _get_underlying_overview(codes)


def get_market_put_call_ratio(market: str = "US_SECURITY", begin_time: Optional[str] = None,
                               end_time: Optional[str] = None) -> List[dict]:
    """
    Market-wide (not per-symbol) Put/Call ratio time series via
    get_option_market_statistic(data_type=VOLUME).
    Row shape: {time, timestamp, call_value, put_value, total_value, ratio}.
    """
    return _get_market_put_call_ratio(market, begin_time, end_time)


# ── Internal ──────────────────────────────────────────────────────────────────

@common.retry()
def _get_underlying_overview(codes: List[str]) -> List[dict]:
    import moomoo as ft
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_option_underlying_overview(codes)
        if ret != ft.RET_OK or data is None or data.empty:
            return []
        return data.to_dict("records")
    finally:
        common.safe_close(ctx)


@common.retry()
def _get_market_put_call_ratio(market: str, begin_time, end_time) -> List[dict]:
    import moomoo as ft
    market_enum = getattr(ft.OptionMarket, market.upper(), ft.OptionMarket.US_SECURITY)
    ctx = common.make_quote_ctx()
    rows: List[dict] = []
    try:
        page_key = None
        while True:
            ret, data, page_key = ctx.get_option_market_statistic(
                market_enum, ft.OptionStatisticDataType.VOLUME,
                begin_time=begin_time, end_time=end_time, page_req_key=page_key,
            )
            if ret != ft.RET_OK or data is None:
                break
            if not data.empty:
                rows.extend(data.to_dict("records"))
            if page_key is None:
                break
    finally:
        common.safe_close(ctx)
    return rows
