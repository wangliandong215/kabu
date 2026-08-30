"""
data/macro.py — Macro indicator / FedWatch data fetching via moomoo OpenD.

Thin data-acquisition layer only (no caching beyond what the caller does,
no risk judgment, no trading decisions) — same idiom as data/earnings.py:
common.make_quote_ctx() + @common.retry(), never imported by engine/runner.py
or risk/*. Consumed by research/collect_market_features.py.

Timezone note (v2.11.1 audit): `data_time`/`release_time` (macro indicators)
and `meeting_date` (FedWatch) are plain date/datetime strings with no
timezone marker from moomoo — treated as US Eastern, matching how the
underlying US government/Fed releases are dated by every public source.

Revision-vintage limitation (v2.11.1 audit — documented, not fixable here):
moomoo's get_macro_indicator_history() returns each indicator's CURRENT
value for a given data_time, not the as-originally-reported value at first
release. If a government agency later revises a historical figure, a
re-fetch of the same data_time will return the REVISED number — this
module has no way to recover the original vintage from the API. What
research/market_features_store.py's macro_snapshots table CAN guarantee is
that every day this collector actually runs, whatever value the API
returned on THAT DAY is preserved forever under its own observed_at (see
that module's docstring — macro_snapshots' primary key includes
observed_at specifically so a later revision creates a new row instead of
overwriting the earlier one). So: "what moomoo reported as of day X" is
preserved exactly; "what was true in objective reality as of day X" is
only as accurate as moomoo's own historical revision handling, which this
codebase cannot inspect or correct.
"""
from typing import List, Optional

import common


def get_macro_indicator_list(region: str) -> List[dict]:
    """List of available macro indicators for `region` (e.g. "US").
    Row shape: {category_name, indicator_id, name}."""
    return _get_macro_indicator_list(region)


def get_macro_indicator_history(indicator_id: str, time: Optional[str] = None,
                                 max_count: Optional[int] = None) -> List[dict]:
    """Historical values for one indicator_id.
    Row shape: {data_time, release_time, value, predict_value, previous_value, unit_type}."""
    return _get_macro_indicator_history(indicator_id, time, max_count)


def get_fed_watch_target_rate() -> List[dict]:
    """Row shape: {meeting_date, target_range, probability}."""
    return _get_fed_watch_target_rate()


def get_fed_watch_dot_plot() -> List[dict]:
    """Row shape: {year, rate, vote_count, is_median, median_rate, current_rate}."""
    return _get_fed_watch_dot_plot()


# ── Internal ──────────────────────────────────────────────────────────────────

@common.retry()
def _get_macro_indicator_list(region: str) -> List[dict]:
    import moomoo as ft
    region_enum = getattr(ft.MacroRegion, region.upper(), None)
    if region_enum is None:
        raise ValueError(f"unknown macro region: {region}")
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_macro_indicator_list(region=region_enum)
        if ret != ft.RET_OK or data is None or data.empty:
            return []
        return data.to_dict("records")
    finally:
        common.safe_close(ctx)


@common.retry()
def _get_macro_indicator_history(indicator_id: str, time, max_count) -> List[dict]:
    import moomoo as ft
    ctx = common.make_quote_ctx()
    try:
        kwargs = {}
        if time is not None:
            kwargs["time"] = time
        if max_count is not None:
            kwargs["max_count"] = max_count
        ret, data = ctx.get_macro_indicator_history(indicator_id=indicator_id, **kwargs)
        if ret != ft.RET_OK or data is None or data.empty:
            return []
        return data.to_dict("records")
    finally:
        common.safe_close(ctx)


@common.retry()
def _get_fed_watch_target_rate() -> List[dict]:
    import moomoo as ft
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_fed_watch_target_rate()
        if ret != ft.RET_OK or data is None or data.empty:
            return []
        return data.to_dict("records")
    finally:
        common.safe_close(ctx)


@common.retry()
def _get_fed_watch_dot_plot() -> List[dict]:
    import moomoo as ft
    ctx = common.make_quote_ctx()
    try:
        ret, data = ctx.get_fed_watch_dot_plot()
        if ret != ft.RET_OK or data is None or data.empty:
            return []
        return data.to_dict("records")
    finally:
        common.safe_close(ctx)
