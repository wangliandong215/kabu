"""
_v21_restore_observation_tier.py -- 如果 v2.1 总分模型对"总分<60"不再直接
放弃开仓、而是像旧系统一样恢复3%观察仓，整体系统全历史表现会怎样？

做法：monkeypatch engine.scoring.compute_total_score，只拦截"总分<60"这一
种SKIP（天气state0的硬拦截不受影响——那发生在更早的4c-macro阶段，候选
列表在到达这里之前就已经清空，不会流到这个函数）。把这类候选的 label 改
成 None（而不是"SKIP"），position_scale 强制1.0：
  - label=None 会让 backtest_portfolio.py 的 `if result.label ==
    scoring.LABEL_SKIP: continue` 判断为 False，候选不再被丢弃；
  - 后面传给 _size() 的 score_label=None，会让 _size() 回退到原始
    signal_strength 阈值分档——trend<0.4 对应的正是旧系统的 WEAK(3%) 档，
    精确复现"观察仓"行为；
  - position_scale=1.0 保证没有总分模型的额外折算。
其余候选（total>=60，PARTIAL/FULL）完全不受影响，走当前v2.1的真实逻辑
（包括"fund/news都缺失时PARTIAL不折算"这条已经修正过的规则）。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _v21_restore_observation_tier.py
"""
import contextlib

import backtest_portfolio as bp
import engine.scoring as scoring

FULL_START = "2015-01-01"
FULL_END   = "2026-07-04"
CASH       = 7_000_000.0
USD_JPY    = 140.0

LOG_PATH = "_v21_restore_observation_tier.log"


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    print(f"Fetching full history {FULL_START} ~ {FULL_END} ...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    _real_compute = scoring.compute_total_score

    def _restore_observation_wrapper(trend_strength, weather_code,
                                      fundamental_score=None, news_score=None):
        result = _real_compute(trend_strength, weather_code, fundamental_score, news_score)
        if result.label == scoring.LABEL_SKIP and weather_code != 0:
            return result._replace(label=None, position_scale=1.0)
        return result

    scoring.compute_total_score = _restore_observation_wrapper
    try:
        print("Simulating v2.1 + restored 3% observation tier ...")
        with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
            r = bp.simulate_from_prepared(prepared, cash=CASH, currency="JPY ")
    finally:
        scoring.compute_total_score = _real_compute

    print(f"\n=== v2.1 + 恢复3%观察仓（弱信号total<60不再放弃开仓）===")
    print(f"总收益: {r['total_return_pct']:.2f}%")
    print(f"年化收益: {r['annualized_return_pct']:.2f}%")
    print(f"Sharpe: {r['sharpe']:.3f}")
    print(f"最大回撤: {r['max_drawdown_pct']:.2f}%")
    print(f"交易笔数: {r['num_trades']}")
    print(f"胜率: {r['win_rate_pct']:.1f}%")


if __name__ == "__main__":
    main()
