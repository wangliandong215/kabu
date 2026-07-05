"""
_yearly_report_v22.py -- 同 _yearly_report_v21.py 的逐年指标格式，对比
v2.0（升级前，链式过滤器+strength阈值分档仓位，永远FULL/position_scale=1.0
的monkeypatch，跟 _v21_old_vs_new_diff.py 验证过的同一套手法）和 v2.2
（当前代码原样：四档状态机 FULL/PARTIAL/OBSERVATION/SKIP，含本次新增的
OBSERVATION 3%观察仓）。

复用 _yearly_report.py 的 _build_table/_period_metrics，纯美股
config.WATCHLIST_EUROPE_US（当前82只——2026-07-03做过区域拆分重构后的
数量，跟用户手头某张截图里"纯美股94只"的旧报告不是同一份股票池，是当时
拆分前的旧列表；这里用现在代码库里实际存在的池子，方法论一致，但绝对数字
不会跟那张旧截图逐位对上，只跟当前脚本自己的v2.0基线可比）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v22.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config
import engine.scoring as scoring
from _yearly_report import _build_table

START = "2015-01-01"
END   = "2026-07-03"
CASH  = 50_000.0

LOG_PATH = "_yearly_report_v22.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US，"
          f"{len(config.WATCHLIST_EUROPE_US)}只）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    # ── v2.0（升级前：链式过滤器 + strength阈值分档仓位）───────────────────────
    print(f"模拟 v2.0（升级前系统，总分模型关闭，cash=${CASH:,.0f}）...")
    _real_compute = scoring.compute_total_score
    _real_size    = bp._size

    def _fake_compute_total_score(trend_strength, weather_code,
                                   fundamental_score=None, news_score=None):
        trend_score = max(0.0, min(1.0, trend_strength)) * 100.0
        weather_score = 0.0 if weather_code == 0 else 100.0
        return scoring.TotalScore(100.0, 1.0, scoring.LABEL_FULL,
                                   trend_score, fundamental_score, news_score, weather_score)

    def _old_size(*args, **kwargs):
        kwargs.pop("score_label", None)
        return _real_size(*args, **kwargs)

    scoring.compute_total_score = _fake_compute_total_score
    bp._size = _old_size
    try:
        with contextlib.redirect_stdout(log_f):
            result_v20 = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    finally:
        scoring.compute_total_score = _real_compute
        bp._size = _real_size
    if not result_v20:
        print("NO RESULT (v2.0) -- aborting")
        return
    df_v20 = _build_table(result_v20, prepared)

    # ── v2.2（当前代码原样：四档状态机，含 OBSERVATION）────────────────────────
    print(f"模拟 v2.2（当前代码，四档状态机 FULL/PARTIAL/OBSERVATION/SKIP，"
          f"cash=${CASH:,.0f}）...")
    with contextlib.redirect_stdout(log_f):
        result_v22 = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    log_f.close()
    if not result_v22:
        print("NO RESULT (v2.2) -- aborting")
        return
    df_v22 = _build_table(result_v22, prepared)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*140}")
    print(f"v2.0（升级前，链式过滤器） {START} ~ {END} 逐年完整指标"
          f"（${CASH:,.0f}起点，纯美股{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df_v20.to_string())
    df_v20.to_csv("_yearly_report_v20_vs_v22_v20_result.csv", encoding="utf-8-sig")

    print(f"\n{'='*140}")
    print(f"v2.2（四档状态机+OBSERVATION观察仓） {START} ~ {END} 逐年完整指标"
          f"（${CASH:,.0f}起点，纯美股{len(config.WATCHLIST_EUROPE_US)}只）")
    print(f"{'='*140}")
    print(df_v22.to_string())
    df_v22.to_csv("_yearly_report_v20_vs_v22_v22_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v20_vs_v22_v20_result.csv / "
          f"_yearly_report_v20_vs_v22_v22_result.csv")


if __name__ == "__main__":
    main()
