"""
_yearly_report_v23_locked.py -- 逐年完整指标表，配置=当前生产锁定配置
（2026-07-04起：ENABLE_ACTIVE_REPLACEMENT=True + ENABLE_REPLACEMENT_
STABILIZATION=False，即纯v2.3主动置换，RSL整层关闭）。

复用 _yearly_report.py::_build_table()/_period_metrics() 的既有指标口径
（收益率/QQQ/简单Alpha/期末金额/交易数/胜率/Sharpe/Sortino/MDD/Calmar/
Profit Factor/Beta/CAPM Alpha），不重新实现——具体定义见该文件的模块
docstring。本脚本只负责按当前config.py的真实默认值跑一次回测并出表，
不做任何A/B对比（跟v2.0/v2.4的对比见analytics/replacement_stability_
report.py）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v23_locked.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config
from _yearly_report import _build_table

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0

LOG_PATH = "_yearly_report_v23_locked.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    print(f"模拟 当前生产锁定配置（ENABLE_ACTIVE_REPLACEMENT="
          f"{config.ENABLE_ACTIVE_REPLACEMENT}, ENABLE_REPLACEMENT_STABILIZATION="
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
    print(f"v2.3（当前生产锁定配置，RSL关闭） {START} ~ {END} 逐年完整指标"
          f"（${CASH:,.0f}起点，纯美股 WATCHLIST_EUROPE_US，{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df.to_string())
    df.to_csv("_yearly_report_v23_locked_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v23_locked_result.csv")


if __name__ == "__main__":
    main()
