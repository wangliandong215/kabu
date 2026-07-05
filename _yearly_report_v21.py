"""
_yearly_report_v21.py -- 同 _yearly_report.py 的逐年指标格式（收益率/QQQ/
简单Alpha/期末金额/交易数/胜率/Sharpe/Sortino/MDD/Calmar/Profit Factor/
Beta/CAPM Alpha），对比 v2.0（升级前，链式过滤器+strength阈值分档仓位）
和 v2.1（横向多因子总分模型）两个版本。复用 _yearly_report.py 的
_build_table/_period_metrics 等函数，不重复实现指标计算逻辑。

V2.0（旧版）= 当前 config.py 锁定配置（大盘天气+动态仓位都开启）叠加
v2.1 total-score 模型被 monkeypatch 关闭——用 engine.scoring.
compute_total_score 永远返回 FULL/position_scale=1.0（不SKIP、不做总分
连续折算）+ backtest_portfolio._size 忽略 score_label 参数（回退到原始
signal_strength 阈值分档：weak<0.4观察仓3%/medium 0.4-0.7为10%/
strong>=0.7为20-30%）精确复现——这是 _v21_old_vs_new_diff.py 已经验证过
「跟当前锁定基线240.91%/0.891/-20.37%/871笔逐位一致」的同一套 monkeypatch。

V2.1（新版）= 当前代码原样跑（无 --news-file，回测里 fund/news 均为
None，PARTIAL档不折算——最终修正版）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _yearly_report_v21.py
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

LOG_PATH = "_yearly_report_v21.log"


def main():
    log_f = open(LOG_PATH, "w", encoding="utf-8")

    print(f"拉取 {START} ~ {END} 纯美股（config.WATCHLIST_EUROPE_US）历史数据...")
    with contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(config.WATCHLIST_EUROPE_US, START, END, usd_to_jpy=None)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    # ── v2.0（升级前：链式过滤器 + strength阈值分档仓位）───────────────────────
    print(f"模拟 v2.0（升级前系统，v2.1总分模型关闭，cash=${CASH:,.0f}）...")
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

    # ── v2.1（当前代码原样：横向多因子总分模型）────────────────────────────────
    print(f"模拟 v2.1（当前代码，横向多因子总分模型，cash=${CASH:,.0f}）...")
    with contextlib.redirect_stdout(log_f):
        result_v21 = bp.simulate_from_prepared(prepared, cash=CASH, currency="$")
    log_f.close()
    if not result_v21:
        print("NO RESULT (v2.1) -- aborting")
        return
    df_v21 = _build_table(result_v21, prepared)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{'='*140}")
    print(f"v2.0（升级前，链式过滤器） {START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股）")
    print(f"{'='*140}")
    print(df_v20.to_string())
    df_v20.to_csv("_yearly_report_v20_result.csv", encoding="utf-8-sig")

    print(f"\n{'='*140}")
    print(f"v2.1（横向多因子总分模型） {START} ~ {END} 逐年完整指标（${CASH:,.0f}起点，纯美股）")
    print(f"{'='*140}")
    print(df_v21.to_string())
    df_v21.to_csv("_yearly_report_v21_result.csv", encoding="utf-8-sig")

    print(f"\n结果已保存: _yearly_report_v20_result.csv / _yearly_report_v21_result.csv")


if __name__ == "__main__":
    main()
