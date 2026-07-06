"""
_yearly_report_v26_locked.py -- 逐年完整指标表，配置=当前生产锁定配置
（同 v2.4/v2.5 锁定配置：REPLACEMENT_MIN_NEW_SCORE=95.0 +
REPLACEMENT_BLOCK_SAME_SECTOR=True + ATR_MULT=14.0，RSL整层仍关闭）。

v2.6 新增 engine/trade_tracker.py（Trade Intelligence Database）——只挂钩
engine/runner.py 的实盘/模拟盘开平仓点位，backtest_portfolio.py 一行代码
都没有改动（Active Replacement 的 _undo_replacement 回滚路径决定了内联
挂钩不安全，改用回测跑完后对 trade_log 做事后配对入库，见
engine/trade_tracker.py::ingest_backtest_trade_log() docstring）。这份脚本
的目的是验证"v2.6 对回测路径零侵入"这个断言——预期这张表应与
_yearly_report_v24_locked_result.csv / _yearly_report_v25_locked_result.csv
逐位一致。

复用 _yearly_report.py::_build_table()/_period_metrics() 的既有指标口径，
格式跟 _yearly_report_v25_locked.py 完全一致，方便逐年对照。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v26_locked.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config
from _yearly_report import _build_table

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0

LOG_PATH = "_yearly_report_v26_locked.log"


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
    print(f"v2.6（Trade Intelligence Database上线，backtest_portfolio.py零改动，生产回测配置沿用v2.4锁定值） "
          f"{START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股 "
          f"WATCHLIST_EUROPE_US，{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df.to_string())
    df.to_csv("_yearly_report_v26_locked_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v26_locked_result.csv")


if __name__ == "__main__":
    main()
