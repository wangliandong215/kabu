"""
_v21_ablation_ab.py -- v2.1 消融实验：量化"大盘天气过滤"(模块一,
engine/market_weather.py + config.MARKET_WEATHER_RISK_MULTIPLIER) 和
"RSI-14动态仓位缩放"(risk/sizing.py::rsi_multiplier + config.DYNAMIC_SIZING_*)
各自对全历史收益/风控的独立贡献。这两个机制在项目memory里都归在同一次
"2026-07-04 V2升级"会话下，但代码层面是两个独立开关，可以分别关闭对比。

三组，单次 fetch 复用（同 _market_weather_ab.py 的模式）：
  BASELINE        当前 v1.1-RELEASE-FINAL 锁定配置原样跑一遍，核对是否
                  复现 277.18%/Sharpe 0.920/-19.85%/873笔（基线核验）。
  A_weather_only  保留天气过滤不变，把动态仓位关掉——
                  DYNAMIC_SIZING_MULT_STRONG/WEAK 都设为 1.0（等价于
                  risk/sizing.py::rsi_multiplier() 恒返回 1.0，不生效）。
  B_dynamic_only  保留动态仓位不变，把天气过滤强制关掉——qqq_weather 序列
                  整体替换为恒定 code=2（安全/全面进攻），MARKET_WEATHER_
                  RISK_MULTIPLIER[2]=1.0x 且不触发状态0/1判断。

注意：qqq_macro_halt（v1.0-RELEASE-FINAL 就存在的"跌破MA200+MA20动量陡降"
硬熔断）在这三组里都保持原样不动——它是天气模块的前置、更早锁定的独立机制，
不是这次要拆解的"V2天气过滤"本身（天气模块的状态0只是复用了它作为危机判定
的一部分）。B组"关闭天气过滤"只关闭天气模块自己的状态0/1判断和风险乘数，
不会连带关掉这条更早的硬熔断。

Usage:
  cd D:\\workspace\\moomoo\\kabu
  python _v21_ablation_ab.py
"""
import contextlib

import pandas as pd

import backtest_portfolio as bp
import config

FULL_START = "2015-01-01"
FULL_END   = "2026-07-03"
CASH       = 7_000_000.0
USD_JPY    = 140.0

LOG_PATH = "_v21_ablation_ab.log"

_ORIG_MULT_STRONG = config.DYNAMIC_SIZING_MULT_STRONG   # 1.5
_ORIG_MULT_WEAK   = config.DYNAMIC_SIZING_MULT_WEAK      # 0.6


class _ConstWeather:
    """qqq_weather 的替身：.get(ts, default) 恒返回 2，用于B组强制关闭天气过滤。"""
    def get(self, _ts, _default=None):
        return 2


def _run_variant(prepared, name, dynamic_mult_strong, dynamic_mult_weak, force_weather_off):
    config.DYNAMIC_SIZING_MULT_STRONG = dynamic_mult_strong
    config.DYNAMIC_SIZING_MULT_WEAK   = dynamic_mult_weak

    variant_prepared = dict(prepared)
    if force_weather_off:
        variant_prepared["qqq_weather"] = _ConstWeather()

    print(f"Simulating {name}  "
          f"DYNAMIC_SIZING_MULT(strong/weak)={dynamic_mult_strong}/{dynamic_mult_weak}  "
          f"weather_forced_off={force_weather_off} ...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        r = bp.simulate_from_prepared(variant_prepared, cash=CASH, currency="JPY ")
    return r


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()   # truncate

    print(f"Fetching full history {FULL_START} ~ {FULL_END} once "
          f"(shared across all variants)...")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f):
        prepared = bp._prepare_backtest_data(bp.BACKTEST_STOCKS, FULL_START, FULL_END, usd_to_jpy=USD_JPY)
    if prepared is None:
        print("NO DATA -- aborting")
        return

    variants = [
        ("BASELINE_v1.1",       _ORIG_MULT_STRONG, _ORIG_MULT_WEAK, False),
        ("A_weather_only",      1.0,                1.0,              False),
        ("B_dynamic_only",      _ORIG_MULT_STRONG, _ORIG_MULT_WEAK, True),
    ]

    rows = []
    for name, mult_strong, mult_weak, weather_off in variants:
        r = _run_variant(prepared, name, mult_strong, mult_weak, weather_off)
        if not r:
            continue
        calmar = (r["annualized_return_pct"] / abs(r["max_drawdown_pct"])
                  if r["max_drawdown_pct"] not in (0, None) else float("nan"))
        rows.append({
            "variant": name,
            "总收益率%": r["total_return_pct"],
            "年化收益率%": r["annualized_return_pct"],
            "夏普比率": r["sharpe"],
            "最大回撤%": r["max_drawdown_pct"],
            "卡玛比率": round(calmar, 3),
            "总交易笔数": r["num_trades"],
            "胜率%": r["win_rate_pct"],
            "基准QQQ%": r["benchmark_qqq_pct"],
        })

    # 恢复原值，避免脚本进程若被复用时残留污染
    config.DYNAMIC_SIZING_MULT_STRONG = _ORIG_MULT_STRONG
    config.DYNAMIC_SIZING_MULT_WEAK   = _ORIG_MULT_WEAK

    df = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\n{'='*100}")
    print(f"v2.1 消融实验：天气过滤 vs 动态仓位 各自贡献 ({FULL_START} ~ {FULL_END}, "
          f"{len(prepared['stocks'])}只股票池)")
    print(f"{'='*100}")
    print(df.to_string())

    df.to_csv("_v21_ablation_ab_result.csv", encoding="utf-8-sig")
    print(f"\n结果已保存: _v21_ablation_ab_result.csv")


if __name__ == "__main__":
    main()
