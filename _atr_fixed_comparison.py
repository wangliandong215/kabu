"""
_atr_fixed_comparison.py -- clean 3-way fixed-ATR_TRAIL_MULT comparison over
the full 2015-2026 history, following the dynamic-ATR walk-forward result
(see engine/regime.py's "REJECTED 2026-07-04" note and
_atr_dynamic_walkforward_ab.py): dynamic ATR lost to a plain fixed ATR=5.5
on every headline metric, so the next question is whether the current
locked default (ATR_TRAIL_MULT / config.ATR_MULT_BASE = 4.0) should simply
move up toward 5.0/5.5.

Single fetch (2015-01-01..2026-07-03, full BACKTEST_STOCKS), reused across
all 3 fixed-ATR variants via simulate_from_prepared() -- no walk-forward,
no grid search, just the plain full-history backtest each variant would
have gotten from `python backtest_portfolio.py` directly, run 3 times.

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr_fixed_comparison.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

ATR_VALUES = [4.0, 5.0, 5.5, 6.0, 6.5]

LOG_PATH = "_atr_fixed_comparison.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once "
          f"(shared across all {len(ATR_VALUES)} fixed-ATR variants)...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    rows = []
    for atr in ATR_VALUES:
        print(f"Simulating ATR_TRAIL_MULT={atr} ...")
        bp.ATR_TRAIL_MULT = atr
        with contextlib.redirect_stdout(log_f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
        if not r:
            continue
        calmar = (r["annualized_return_pct"] / abs(r["max_drawdown_pct"])
                  if r["max_drawdown_pct"] not in (0, None) else float("nan"))
        rows.append({
            "ATR": atr,
            "总收益率%": r["total_return_pct"],
            "年化收益率%": r["annualized_return_pct"],
            "夏普比率": r["sharpe"],
            "最大回撤%": r["max_drawdown_pct"],
            "卡玛比率": round(calmar, 3),
            "总交易笔数": r["num_trades"],
            "胜率%": r["win_rate_pct"],
            "基准QQQ%": r["benchmark_qqq_pct"],
        })

    log_f.close()

    df = pd.DataFrame(rows).set_index("ATR")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\n{'='*100}")
    print(f"固定ATR全历史对比 ({FULL_START} ~ {FULL_END}, {len(prepared['stocks'])}只股票池)")
    print(f"{'='*100}")
    print(df.to_string())

    df.to_csv("_atr_fixed_comparison_result.csv", encoding="utf-8-sig")
    print(f"\n结果已保存: _atr_fixed_comparison_result.csv")


if __name__ == "__main__":
    main()
