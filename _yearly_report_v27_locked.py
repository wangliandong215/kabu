"""
_yearly_report_v27_locked.py -- 逐年完整指标表，配置=当前生产锁定配置
（同 v2.4/v2.5/v2.6 锁定配置：REPLACEMENT_MIN_NEW_SCORE=95.0 +
REPLACEMENT_BLOCK_SAME_SECTOR=True + ATR_MULT=14.0，RSL整层仍关闭）。

V2.7 分支目前只新增了独立的 v27_hmm/ 研究模块（HMM市场状态识别，
不接入 engine/runner.py 或 backtest_portfolio.py 任何调用点），
backtest_portfolio.py 本身零改动。这份脚本的目的是确认这一断言：
交易逻辑层面 v2.7 与 v2.6 应该完全一致，任何数字差异只可能来自
watchlist 本身的变化（v2.6 收尾当天 config.WATCHLIST_EUROPE_US
从82只扩容到139只，与代码逻辑改动无关，见 _yearly_report_v26_locked.py
82只结果 vs 本次139只结果的对比）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v27_locked.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config
from _yearly_report import _build_table

START = "2015-01-01"
END   = "2026-07-07"
CASH  = 50_000.0

LOG_PATH = "_yearly_report_v27_locked.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US，{len(config.WATCHLIST_EUROPE_US)}只）历史数据...")
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
    print(f"v2.7（新增独立 v27_hmm/ 研究模块，backtest_portfolio.py零改动，生产回测配置沿用v2.4锁定值） "
          f"{START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股 "
          f"WATCHLIST_EUROPE_US，{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df.to_string())
    df.to_csv("_yearly_report_v27_locked_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v27_locked_result.csv")


if __name__ == "__main__":
    main()
