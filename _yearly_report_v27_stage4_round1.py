"""
_yearly_report_v27_stage4_round1.py -- V2.7 Stage 4, Round 1 experiment:
"Bear regime -> suppress new entries" A/B test against the locked V2.6/V2.7
baseline, on the full 139-stock config.WATCHLIST_EUROPE_US universe.

Methodology (single-variable discipline, per user instruction):
  - Fetches `prepared` data ONCE (identical to _yearly_report_v27_locked.py's
    invocation: same START/END/CASH/watchlist), then calls
    simulate_from_prepared() TWICE against that same prepared dict:
      baseline   : regime_lookup=None (byte-identical to the locked V2.6/V2.7
                   baseline -- this is the single control variable's "off"
                   state, not a re-derivation).
      experiment : regime_lookup=<8-stock causal HMM regime series> (the
                   ONLY thing that differs from baseline).
  - regime_lookup only covers the 8 pilot stocks that have a trained
    engine/hmm_regime.py model (see project memory, V2.7 Stage 3). All other
    131 stocks in the 139-stock universe are absent from regime_lookup and
    therefore completely unaffected by the gate -- this keeps the portfolio-
    level comparison apples-to-apples (same universe, same capacity/sector
    dynamics) while being honest that the gate's effect can only show up in
    those 8 stocks' own trades.
  - Regime decoding uses engine.hmm_regime.decode_regime_series_causal(), a
    walk-forward no-look-ahead decoder -- NOT MLRegimeClassifier.classify_series()
    batch-called over the whole history, which would leak future information
    into past days' regime labels inside a backtest (see that function's
    docstring for why).

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v27_stage4_round1.py
"""
import contextlib

import pandas as pd

import backtest
import backtest_portfolio as bp
import config
from _yearly_report import _build_table
from engine.hmm_regime import decode_regime_series_causal, load_stock_hmm, prepare_ohlcv

START = "2015-01-01"
END = "2026-07-07"
CASH = 50_000.0

PILOT_CODES = [
    "US.AAPL", "US.QQQ", "US.MSFT", "US.NVDA",
    "US.AMZN", "US.GOOGL", "US.TSLA", "US.SPY",
]

LOG_PATH = "_yearly_report_v27_stage4_round1.log"


def build_regime_lookup(codes, start, end) -> dict:
    """{code: DataFrame(regime, confidence, duration)} -- decode_regime_series_causal()
    returns a DataFrame as of V2.7 Stage 4's Regime Analytics phase (2026-07-08),
    not a bare label Series (see engine/hmm_regime.py)."""
    lookup = {}
    for code in codes:
        payload = load_stock_hmm(code)
        if payload is None:
            print(f"  [skip] {code}: no trained HMM model")
            continue
        raw = backtest.fetch_kline(code, start, end)
        df = prepare_ohlcv(raw)
        regime_df = decode_regime_series_causal(payload, df)
        lookup[code] = regime_df
        counts = regime_df["regime"].dropna().value_counts().to_dict()
        print(f"  [ok] {code}: {counts}")
    return lookup


def _sell_trades_df(result: dict) -> pd.DataFrame:
    sell = [t for t in result["trade_log"] if t["side"] == "SELL"]
    df = pd.DataFrame(sell)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["pnl"] = df["pnl"].astype(float)
        df["holding_days"] = (df["date"] - df["entry_date"]).dt.days
    return df


def _regime_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Only meaningful for the PILOT_CODES trades (others have
    regime_at_entry=None -- grouped together as "N/A / ungated")."""
    if len(trades_df) == 0:
        return pd.DataFrame()
    df = trades_df.copy()
    df["regime_at_entry"] = df["regime_at_entry"].fillna("N/A (ungated stock)")
    rows = {}
    for label, sub in df.groupby("regime_at_entry"):
        n = len(sub)
        wins = sub[sub["pnl"] > 0]
        rows[label] = {
            "trades": n,
            "win_rate_%": round(len(wins) / n * 100, 1) if n else 0.0,
            "total_pnl": round(sub["pnl"].sum(), 2),
            "gross_wins": round(sub.loc[sub["pnl"] > 0, "pnl"].sum(), 2),
            "gross_losses": round(sub.loc[sub["pnl"] < 0, "pnl"].sum(), 2),
            "avg_holding_days": round(sub["holding_days"].mean(), 1),
        }
    return pd.DataFrame(rows).T


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US，"
          f"{len(config.WATCHLIST_EUROPE_US)}只）历史数据（baseline与experiment共用同一份fetch）...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print(f"\n为 {len(PILOT_CODES)} 只试点股票构建因果（无未来函数）regime序列...")
    regime_lookup = build_regime_lookup(PILOT_CODES, START, END)

    print(f"\n[1/2] 跑基线（regime_lookup=None，与V2.6/V2.7锁定基线逻辑完全一致）...")
    with contextlib.redirect_stdout(log_f):
        result_base = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")

    print(f"[2/2] 跑实验（Bear regime 禁止新开仓，仅 {len(regime_lookup)} 只试点股票受影响）...")
    # enable_bear_gate=True: this script reproduces Round 1's actual rule
    # experiment (paused as of 2026-07-08, see project memory) -- the flag
    # defaults to False (tag-only, no gating) since simulate_from_prepared()
    # was refactored for the Regime Analytics phase; must be explicit here.
    with contextlib.redirect_stdout(log_f):
        result_exp = bp.simulate_from_prepared(
            prepared, cash=CASH, currency="$",
            regime_lookup=regime_lookup, enable_bear_gate=True,
        )
    log_f.close()

    df_base = _build_table(result_base, prepared)
    df_exp = _build_table(result_exp, prepared)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*100}")
    print("全期汇总对比：基线（V2.6/V2.7）vs 实验（Bear禁止新开仓，Round 1）")
    print(f"{'='*100}")
    compare = pd.concat(
        [df_base.loc[["全期汇总"]].T, df_exp.loc[["全期汇总"]].T], axis=1
    )
    compare.columns = ["基线(V2.6)", "实验(Bear禁止新开仓)"]
    print(compare.to_string())

    base_trades = _sell_trades_df(result_base)
    exp_trades = _sell_trades_df(result_exp)

    print(f"\n{'='*100}")
    print(f"基线：全部平仓交易数={len(base_trades)}, 平均持仓天数="
          f"{base_trades['holding_days'].mean():.1f}")
    print(f"实验：全部平仓交易数={len(exp_trades)}, 平均持仓天数="
          f"{exp_trades['holding_days'].mean():.1f}")
    print(f"{'='*100}")

    print("\n基线 -- 按入场时 Regime 分组（8只试点股票的交易才有非N/A标签）：")
    print(_regime_breakdown(base_trades).to_string())

    print("\n实验 -- 按入场时 Regime 分组：")
    print(_regime_breakdown(exp_trades).to_string())

    # ── 精确统计"被拦截的开仓次数"：baseline 里 8 只试点股票、且入场日
    # regime_lookup 判定为 Bear 的 BUY 记录，就是本该在 experiment 里
    # 被拦截、且确实没有出现的那些交易（experiment 里同 code/date 不应
    # 再出现 BUY）。用 BUY 侧（不是 SELL 侧）trade_log 直接核对，不依赖
    # trade_log 自带的 regime_at_entry 字段（baseline 跑的时候
    # regime_lookup=None，这个字段全是 None，见函数顶部 docstring）。
    base_buys = pd.DataFrame([t for t in result_base["trade_log"] if t["side"] == "BUY"])
    exp_buys = pd.DataFrame([t for t in result_exp["trade_log"] if t["side"] == "BUY"])
    base_buys["date"] = pd.to_datetime(base_buys["date"])
    exp_buys["date"] = pd.to_datetime(exp_buys["date"])

    blocked_rows = []
    for code, regime_df in regime_lookup.items():
        pilot_buys = base_buys[base_buys["code"] == code]
        for _, row in pilot_buys.iterrows():
            d = row["date"]
            if d in regime_df.index and regime_df.loc[d, "regime"] == "Bear":
                still_present = (
                    (exp_buys["code"] == code) & (exp_buys["date"] == d)
                ).any()
                blocked_rows.append(
                    {"code": code, "date": d, "reason": row["reason"],
                     "strength": row.get("strength"),
                     "still_present_in_experiment": bool(still_present)}
                )
    blocked_df = pd.DataFrame(blocked_rows)

    print(f"\n{'='*100}")
    print(f"Bear regime 拦截明细：baseline 中 8 只试点股票在 Bear 状态下本会开仓的次数 "
          f"= {len(blocked_df)}")
    print(f"{'='*100}")
    if len(blocked_df):
        print(blocked_df.to_string(index=False))
        leaked = blocked_df["still_present_in_experiment"].sum()
        print(f"\n门禁校验：{len(blocked_df)} 次里有 {leaked} 次在 experiment 里仍然出现"
              f"（应为0，非0说明门禁逻辑有漏洞）")
    else:
        print("（这8只试点股票在baseline里，从未在HMM判定为Bear的当天产生过BUY信号——"
              "说明门禁在Round 1样本里几乎没有实际拦截到任何交易）")

    # 8只试点股票各自：baseline / experiment 的开仓次数对比（含QQQ底仓等
    # 非strength门控的BUY会两边都有，属正常，不是门禁效果）
    print(f"\n8只试点股票 baseline vs experiment 开仓次数对比：")
    for code in PILOT_CODES:
        b = len(base_buys[base_buys["code"] == code])
        e = len(exp_buys[exp_buys["code"] == code])
        print(f"  {code}: baseline={b}, experiment={e}, 差值={b - e}")

    df_base.to_csv("_yearly_report_v27_stage4_round1_baseline.csv", encoding="utf-8-sig")
    df_exp.to_csv("_yearly_report_v27_stage4_round1_experiment.csv", encoding="utf-8-sig")
    base_trades.to_csv("_yearly_report_v27_stage4_round1_baseline_trades.csv",
                        index=False, encoding="utf-8-sig")
    exp_trades.to_csv("_yearly_report_v27_stage4_round1_experiment_trades.csv",
                       index=False, encoding="utf-8-sig")
    base_buys.to_csv("_yearly_report_v27_stage4_round1_baseline_buys.csv",
                      index=False, encoding="utf-8-sig")
    exp_buys.to_csv("_yearly_report_v27_stage4_round1_experiment_buys.csv",
                     index=False, encoding="utf-8-sig")
    if len(blocked_df):
        blocked_df.to_csv("_yearly_report_v27_stage4_round1_blocked.csv",
                           index=False, encoding="utf-8-sig")
    print("\n结果已保存：_yearly_report_v27_stage4_round1_*.csv")


if __name__ == "__main__":
    main()
