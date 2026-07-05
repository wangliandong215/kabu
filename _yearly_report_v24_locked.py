"""
_yearly_report_v24_locked.py -- 逐年完整指标表，配置=当前生产锁定配置
（2026-07-05起：v2.3主动置换基础上 + REPLACEMENT_MIN_NEW_SCORE=95.0 +
REPLACEMENT_BLOCK_SAME_SECTOR=True，经消融验证后转正的v2.4 Replace
Engine默认配置，RSL整层仍关闭）。

复用 _yearly_report.py::_build_table()/_period_metrics() 的既有指标口径
（收益率/QQQ/简单Alpha/期末金额/交易数/胜率/Sharpe/Sortino/MDD/Calmar/
Profit Factor/Beta/CAPM Alpha），不重新实现，格式跟 _yearly_report_
v23_locked.py 完全一致，方便逐年对照。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v24_locked.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config
from _yearly_report import _build_table

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0

LOG_PATH = "_yearly_report_v24_locked.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print(f"模拟 当前生产锁定配置（REPLACEMENT_MIN_NEW_SCORE="
          f"{config.REPLACEMENT_MIN_NEW_SCORE}, REPLACEMENT_BLOCK_SAME_SECTOR="
          f"{config.REPLACEMENT_BLOCK_SAME_SECTOR}, ENABLE_REPLACEMENT_STABILIZATION="
          f"{config.ENABLE_REPLACEMENT_STABILIZATION}，cash=${CASH:,.0f}）...")
    with contextlib.redirect_stdout(log_f):
        result = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    log_f.close()
    if not result:
        print("NO RESULT -- aborting")
        return
    df = _build_table(result, prepared)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*140}")
    print(f"v2.4（生产已锁定为此配置，new_score+same_sector过滤，RSL已关闭） "
          f"{START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股 "
          f"WATCHLIST_EUROPE_US，{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df.to_string())
    df.to_csv("_yearly_report_v24_locked_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v24_locked_result.csv")


if __name__ == "__main__":
    main()
