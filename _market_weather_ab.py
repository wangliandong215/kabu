"""
_market_weather_ab.py -- V1/V2 full-history comparison for the "Market
Weather" regime classifier (engine/market_weather.py) after the 2026-07-04
refactor: RISK_PER_TRADE_PCT is now scaled directly by
config.MARKET_WEATHER_RISK_MULTIPLIER instead of adding a second, parallel
ATR-based risk cap (the first version double-counted against the existing
risk cap for ATR-stop strategies and cost ~25% of baseline return for no
drawdown benefit -- see config.py comment for the full writeup).

Single fetch (2015-01-01..2026-07-03, full BACKTEST_STOCKS), reused across
both variants via simulate_from_prepared() -- same pattern as
_atr_fixed_comparison.py.

Variants (only config.MARKET_WEATHER_RISK_MULTIPLIER differs):
  V1 -- {2:1.0, 1:1.0, 0:0.0}  locked default: state-0 hard-blocks new entries
        (existing positions still exit normally), state-1 does NOT reduce
        risk (this is the answer to "does halving on state-1 help" -- it
        doesn't, see V2 below), state-2 is full risk. Should closely
        reproduce the pre-Market-Weather locked baseline (return 277.18%,
        Sharpe 0.920, MDD -19.85%, 873 trades) since the risk formula itself
        is now byte-for-byte the old one when weather_mult=1.0; any
        remaining gap is state-0 hard-blocking days beyond what
        qqq_macro_halt alone already blocked.
  V2 -- {2:1.0, 1:0.5, 0:0.0}  state-1 halves the risk budget, re-tested
        against the corrected (non-double-counting) implementation to
        confirm the earlier finding still holds.

V2 vs V1 isolates the effect of state-1 half-sizing specifically -- the
question the user asked to re-test (a structurally similar mechanism,
qqq_risk_off_series, was tried and rolled back in this project for costing
28.5% of baseline return; this is a different trigger definition and a
genuinely new experiment, not a resurrection).

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _market_weather_ab.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

VARIANTS = {
    "V1_locked_default":  {2: 1.0, 1: 1.0, 0: 0.0},
    "V2_state1_half":     {2: 1.0, 1: 0.5, 0: 0.0},
}

LOG_PATH = "_market_weather_ab.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once "
          f"(shared across all {len(VARIANTS)} variants)...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    rows = []
    for name, mult in VARIANTS.items():
        print(f"Simulating {name}  MARKET_WEATHER_RISK_MULTIPLIER={mult} ...")
        config.MARKET_WEATHER_RISK_MULTIPLIER = mult
        with contextlib.redirect_stdout(log_f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
        if not r:
            continue
        calmar = (r["annualized_return_pct"] / abs(r["max_drawdown_pct"])
                  if r["max_drawdown_pct"] not in (0, None) else float("nan"))
        rows.append({
            "variant": name,
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

    df = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\n{'='*100}")
    print(f"Market Weather V0/V1/V2 全历史对比 ({FULL_START} ~ {FULL_END}, {len(prepared['stocks'])}只股票池)")
    print(f"{'='*100}")
    print(df.to_string())

    df.to_csv("_market_weather_ab_result.csv", encoding="utf-8-sig")
    print(f"\n结果已保存: _market_weather_ab_result.csv")


if __name__ == "__main__":
    main()
