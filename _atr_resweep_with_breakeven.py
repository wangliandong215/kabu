"""
_atr_resweep_with_breakeven.py -- 2026-07-06 保本止损锁修复后的 ATR
跟踪止损倍数重新扫描。

背景：SSoT审计发现 backtest_portfolio.py 的ATR跟踪止损从未实现
risk/guard.py::update_trailing_stop()（实盘/模拟盘用）里已有的保本止损锁
（浮盈超过1×entry_ATR后止损线锁定到不低于成本价）——不是数值巧合掩盖，
是真的缺失。已用 risk/guard.py::breakeven_lock_floor() 共享实现补上（见
backtest_portfolio.py该段落注释），补上后82只池全历史(v2.4当前配置：
new_score>=95+same_sector拦截)从251.03%/Sharpe0.926/-18.20%/837笔
暴跌到180.49%/Sharpe0.762/-21.60%/1318笔——说明当初锁定ATR_TRAIL_MULT=
5.5那轮walk-forward验证（v1.1-RELEASE，2026-07-04）全程是在"没有保本锁"
的假设下做的，保本锁生效后原最优值很可能不再是最优，需要重新扫描。

方法论：跟 _atr_fixed_comparison.py 同款单次fetch+复用prepared数据、
纯网格扫描（不是完整的4窗口walk-forward+相关性分析那一整套研究），但
换成v2.3/v2.4现在统一使用的82只纯美股池+$50,000起点（不是那次脚本用的
142只混合池+JPY换算），且v2.4的Replace过滤门（new_score>=95+
same_sector拦截）保持开启，扫描的是在"这就是要上模拟盘的完整配置"这个
前提下，ATR_TRAIL_MULT该设多少。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _atr_resweep_with_breakeven.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0
POOL  = config.WATCHLIST_EUROPE_US

ATR_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]

LOG_PATH = "_atr_resweep_with_breakeven.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"确认当前v2.4配置：REPLACEMENT_MIN_NEW_SCORE={config.REPLACEMENT_MIN_NEW_SCORE}, "
          f"REPLACEMENT_BLOCK_SAME_SECTOR={config.REPLACEMENT_BLOCK_SAME_SECTOR}")
    print(f"拉取 {START} ~ {END}，纯美股池（{len(POOL)}只），仅拉取一次...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(POOL, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    orig_atr = bp.ATR_TRAIL_MULT
    rows = []
    try:
        for atr in ATR_VALUES:
            print(f"模拟 ATR_TRAIL_MULT={atr} ...")
            bp.ATR_TRAIL_MULT = atr
            with contextlib.redirect_stdout(log_f):
                r = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
            if not r:
                continue
            calmar = (r["annualized_return_pct"] / abs(r["max_drawdown_pct"])
                      if r["max_drawdown_pct"] not in (0, None) else float("nan"))
            rows.append({
                "ATR_TRAIL_MULT": atr,
                "总收益%": r["total_return_pct"],
                "年化收益%": r["annualized_return_pct"],
                "Sharpe": r["sharpe"],
                "最大回撤%": r["max_drawdown_pct"],
                "Calmar": round(calmar, 3),
                "交易数": r["num_trades"],
                "胜率%": r["win_rate_pct"],
            })
    finally:
        bp.ATR_TRAIL_MULT = orig_atr

    log_f.close()

    df = pd.DataFrame(rows).set_index("ATR_TRAIL_MULT")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\n{'='*110}")
    print(f"ATR_TRAIL_MULT 重新扫描（含保本止损锁修复 + v2.4 Replace过滤门），"
          f"{START} ~ {END}，{len(prepared['stocks'])}只纯美股，${CASH:,.0f}起点")
    print(f"{'='*110}")
    print(df.to_string())

    best_sharpe = df["Sharpe"].idxmax()
    best_return = df["总收益%"].idxmax()
    best_calmar = df["Calmar"].idxmax()
    print(f"\n最高Sharpe: ATR={best_sharpe} ({df.loc[best_sharpe, 'Sharpe']})")
    print(f"最高总收益: ATR={best_return} ({df.loc[best_return, '总收益%']}%)")
    print(f"最高Calmar: ATR={best_calmar} ({df.loc[best_calmar, 'Calmar']})")

    df.to_csv("_atr_resweep_with_breakeven_result.csv", encoding="utf-8-sig")
    print(f"\n结果已保存: _atr_resweep_with_breakeven_result.csv")


if __name__ == "__main__":
    main()
