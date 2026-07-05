"""
_v22_observation_ablation.py -- v2.2 消融实验：把 v2.1 的"总分<60放弃开仓"
硬过滤拆成 SKIP(<40) / OBSERVATION(40-59，3%观察仓) 两档后，量化：
  1) 组合层面对比：v2.1 baseline（沿用旧的<60直接SKIP） vs v2.2（新四档状态机）
     的总收益/Sharpe/最大回撤/交易笔数/胜率——验证"收益改善是否来自
     Observation档位本身，而非其他代码变动"（唯一变量就是这条决策规则）。
  2) 交易层面：从 v2.2 的 trade_log 里，用 total_score 反推每笔 BUY 的
     label（40<=total<60 即 OBSERVATION），配对同一 (code, entry_date) 的
     SELL 记录，统计这批"恢复的观察仓交易"自己的：新增交易数量/平均收益率/
     胜率/Profit Factor/对总收益(JPY)的直接贡献。

v2.1 baseline 复现方式：monkeypatch engine.scoring.compute_total_score，把
真实计算出的 LABEL_OBSERVATION 结果强制改写成 LABEL_SKIP（position_scale=0），
这精确对应 v2.1 决策层"total<60即SKIP"的行为——不改动其余任何逻辑（PARTIAL
折算规则、天气二元否决等原样不动)，确保两组之间的唯一变量就是这条被测规则。

注意（近似）：观察仓交易的"收益率%"用配对到的**首次建仓**price*qty作为成本
基准；如果该笔交易后续被 TRENDING_EARLY 晋升机制加过仓（PROMOTE），加仓部分
的成本不计入分母——这是离线分析脚本的简化近似，不影响 pnl（美元/日元）本身
的准确性（pnl直接来自trade_log的真实实现盈亏），只影响"收益率%"这一项指标。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _v22_observation_ablation.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import engine.scoring as scoring

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

LOG_PATH = "_v22_observation_ablation.log"


def _run(prepared, label):
    print(f"Simulating {label} ...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
    return r


def _summ(name, r):
    calmar = (r["annualized_return_pct"] / abs(r["max_drawdown_pct"])
              if r["max_drawdown_pct"] not in (0, None) else float("nan"))
    return {
        "variant": name,
        "总收益率%": r["total_return_pct"],
        "年化收益率%": r["annualized_return_pct"],
        "夏普比率": r["sharpe"],
        "最大回撤%": r["max_drawdown_pct"],
        "卡玛比率": round(calmar, 3),
        "总交易笔数": r["num_trades"],
        "胜率%": r["win_rate_pct"],
        "基准QQQ%": r["benchmark_qqq_pct"],
    }


def _classify_observation_trades(trade_log):
    buy_map = {}
    for t in trade_log:
        if t["side"] == "BUY" and t.get("reason") == "SIGNAL" and t.get("total_score") is not None:
            buy_map[(t["code"], t["date"])] = {
                "total_score": t["total_score"],
                "price": t["price"],
                "qty": t["qty"],
            }

    obs_trades = []
    for t in trade_log:
        if t["side"] != "SELL":
            continue
        key = (t["code"], t.get("entry_date"))
        entry = buy_map.get(key)
        if entry is None:
            continue
        ts = entry["total_score"]
        if not (scoring.OBSERVATION_THRESHOLD <= ts < scoring.PARTIAL_THRESHOLD):
            continue
        cost_basis = entry["price"] * entry["qty"]
        pnl = t["pnl"] or 0.0
        obs_trades.append({
            "code": t["code"], "entry_date": t["entry_date"], "exit_date": t["date"],
            "total_score": ts, "pnl": pnl,
            "return_pct": (pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0,
        })
    return obs_trades


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once (shared across variants)...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    # ── variant 1: v2.1 baseline（OBSERVATION 强制退化回 SKIP）───────────────
    _real_compute = scoring.compute_total_score

    def _v21_wrapper(trend_strength, weather_code, fundamental_score=None, news_score=None):
        r = _real_compute(trend_strength, weather_code, fundamental_score, news_score)
        if r.label == scoring.LABEL_OBSERVATION:
            return r._replace(label=scoring.LABEL_SKIP, position_scale=0.0)
        return r

    scoring.compute_total_score = _v21_wrapper
    try:
        r_v21 = _run(prepared, "v2.1_baseline (OBSERVATION forced back to SKIP)")
    finally:
        scoring.compute_total_score = _real_compute

    # ── variant 2: v2.2（当前默认代码，真实四档状态机）──────────────────────
    r_v22 = _run(prepared, "v2.2_default (FULL/PARTIAL/OBSERVATION/SKIP)")

    rows = [_summ("v2.1_baseline", r_v21), _summ("v2.2_default", r_v22)]
    df = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*100}")
    print(f"v2.2 消融实验：OBSERVATION 档位对全历史表现的影响 "
          f"({FULL_START} ~ {FULL_END}, {len(prepared['stocks'])}只股票池)")
    print(f"{'='*100}")
    print(df.to_string())
    df.to_csv("_v22_observation_ablation_result.csv", encoding="utf-8-sig")

    # ── 交易层面：v2.2 里被恢复的 OBSERVATION 交易 ──────────────────────────
    obs_trades = _classify_observation_trades(r_v22["trade_log"])
    n = len(obs_trades)
    print(f"\n{'='*100}")
    print(f"v2.2 中 OBSERVATION 档位（40<=total<60）已平仓交易明细统计（n={n}）")
    print(f"{'='*100}")
    if n == 0:
        print("没有任何 OBSERVATION 交易被触发/平仓。")
    else:
        pnl_list = [t["pnl"] for t in obs_trades]
        ret_list = [t["return_pct"] for t in obs_trades]
        wins   = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        win_rate = len(wins) / n * 100.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        total_pnl = sum(pnl_list)
        avg_ret = sum(ret_list) / n

        print(f"新增交易数量: {n}")
        print(f"平均单笔收益率: {avg_ret:.2f}%")
        print(f"胜率: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"累计已实现盈亏 (JPY): {total_pnl:,.0f}")
        print(f"v2.1->v2.2 总交易笔数变化: {r_v21['num_trades']} -> {r_v22['num_trades']} "
              f"(+{r_v22['num_trades'] - r_v21['num_trades']})")

        obs_df = pd.DataFrame(obs_trades)
        obs_df.to_csv("_v22_observation_trades_detail.csv", index=False, encoding="utf-8-sig")
        print(f"\n逐笔明细已保存: _v22_observation_trades_detail.csv")

    print(f"\n结果已保存: _v22_observation_ablation_result.csv")


if __name__ == "__main__":
    main()
