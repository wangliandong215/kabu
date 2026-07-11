"""
engine/fundamental.py — 基本面评分（v2.1 多因子总分模型"基本面(20%)"分量）。

**范围收窄声明（2026-07-04 实测确认，做决策前已用真实 OpenD 连接验证过）：**
moomoo 的 `get_rating_change`（评级变动事件流）和 `get_earnings_beat_rank`
（盈利超/不及预期排名）两个接口，用当前账号/OpenD 实测都返回"未知的协议
ID"错误——跟项目里已经踩过的 JP 行情权限坑是同一类问题（账号/数据订阅
等级不够，不是代码 bug），所以这两个更直接的信号源目前拿不到。

真正能用的只有 `get_research_rating_summary`（机构逐条历史评级快照，含
数值 `rating` 字段 + `target_price`）。用它做"评级趋势"判断：同一机构最近
一次评级相比该机构自己上一次评级如果下降，记一次"评级下调"。

**方向假设已用真实数据验证（2026-07-04）**：`rating` 字段实测取值出现过
0-4，超出了 moomoo 自己 `RatingLevel` 枚举文档的 0-3 范围（`to_string()`:
0=N/A, 1=SELL, 2=HOLD, 3=BUY，4 会报错"cannot be converted to SortField
Type"——这个报错来自评级→排序字段的转换函数，不代表 4 不是合法评级值）。
这里采用"数值越大越看多"的方向假设，已用 15 只真实美股（AAPL/NVDA/INTC/
BA/TSLA/MSFT/META/AMZN/GOOGL/NFLX/AMD/QCOM/PYPL/DIS/NKE）、53 条真实机构
评级变动记录做交叉验证：把"评级变动方向"和同一条记录里"目标价变动方向"
比对，45/53（约85%）方向一致（评级数值上升伴随目标价上调，反之亦然），
只有8条不一致、2条目标价持平——单条机构评级本身就有噪音（目标价还受
其他因素影响），85%量级的一致性足以支撑"数值越大越看多"这个方向假设，
不需要再假设。

Tier1（黑天鹅一票否决）**不从本模块产生**——fraud/delisting/SEC调查等
黑天鹅事件的检测完全交给 engine/news_filter.py 的 Tier1 关键词分类器
（新闻评分分量已覆盖这一类事件），本模块只输出 Tier2/Tier3。
`get_financial_unusual` 实测返回的是成交量/换手率异动的文本描述，不是
财务造假/退市类信号，不适合拿来冒充 Tier1 检测器，本模块不使用它。

env="backtest"：返回 score=None（moomoo 这几个接口都不支持按历史日期
查询，历史基本面数据完全没有真实来源，编造出来就是 GIGO，见项目 memory
里新闻模块的同一条教训）。

env="paper"/"live"：调用 get_research_rating_summary，API 失败/无数据
同样返回 score=None（不是硬阻断，也不是塞一个固定占位分——网络抖动不该
让候选票被误判为"有利空"，也不该被当成"确认无风险"，这两种都是在编造
不存在的信息）。

**2026-07-04 修正（用户指出并采纳，Method A）**：score=None 交给
engine/scoring.py::compute_total_score() 处理——"缺失数据"应该直接从
加权平均里剔除、剩余权重重新归一化，而不是塞一个固定分（70或100）参与
平均。塞固定分参与平均，本质上都是"用编造的信息去乘一个非零权重"，
区别只是编的分数偏乐观还是偏悲观，仍然会用不存在的信息影响仓位决策。
第一版试过塞70分（回测全历史收益240.91%->207.09%，明显被拖累）和100分
（207.09%->219.75%，非常接近基线但概念上仍不对，"缺失数据=满分参与
平均"只是这次权重占比恰好让副作用不明显），两版都不是正确做法。改成
剔除+重新归一化后，"只有趋势+天气"参与加权（回测无--news-file的常态）
时，几乎精确复现了旧链式过滤器 strength>=0.4/0.7 的强弱信号分档边界
（见 engine/scoring.py 的 boundary 单元测试），这不是巧合，是加权归一化
数学上的自然结果，进一步印证这是正确的修正方向。
"""
from typing import Optional

import notify.alert as alert

TIER2_SCORE = 50.0   # 预期差恶化：评级下调
TIER3_SCORE = 100.0  # 正常/正面：评级持平或上调，或无足够历史判断趋势

# 判断"评级下调"的回溯窗口——只看最近这么多天内发生的下调，避免几年前的
# 一次性下调被反复计入"当前扣分"
LOOKBACK_DAYS = 90


def score(code: str, env: str) -> dict:
    """
    Returns {"score": float|None, "tier": 2|3|None, "reason": str}.
    score=None (tier=None) 表示当前没有真实基本面数据，调用方应该把它原样
    传给 engine.scoring.compute_total_score()，让该维度被剔除、不参与加权
    ——不要在这里或调用方把 None 替换成任何固定数值。

    env : "backtest" -> 始终 None（没有历史数据源，见模块docstring）。
          "paper"/"live" -> 真实调用 get_research_rating_summary；
          API失败/无数据同样是 None。
    """
    if env == "backtest":
        return {"score": None, "tier": None, "reason": "backtest_no_data_source"}

    downgrade = _recent_downgrade(code)
    if downgrade is None:
        return {"score": None, "tier": None, "reason": "no_data_or_error"}
    if downgrade:
        return {"score": TIER2_SCORE, "tier": 2, "reason": f"rating_downgrade:{downgrade}"}
    return {"score": TIER3_SCORE, "tier": 3, "reason": "rating_stable_or_up"}


def _recent_downgrade(code: str) -> Optional[str]:
    """
    Returns a description string if any institution's most recent rating is
    lower than that same institution's previous rating within LOOKBACK_DAYS,
    None if no downgrade found, or None (treated as "no signal") on any
    API/data error -- never raises.
    """
    import time
    from datetime import datetime, timedelta

    import moomoo as ft

    ticker = code.split(".")[-1] if "." in code else code
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)

    try:
        ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            ret, data = ctx.get_research_rating_summary(
                f"US.{ticker}" if not code.startswith(("US.", "HK.", "SH.", "SZ.")) else code,
                rating_dimension_type=1,
                num=20,
            )
            if ret != ft.RET_OK or not data or not data.get("inst_rating_summary_list"):
                return None

            for inst in data["inst_rating_summary_list"]:
                items = inst.get("rating_item_list") or []
                # 按 recommendation_date 降序，确保 items[0] 是该机构最新一次评级
                items = sorted(items, key=lambda x: x.get("recommendation_date", 0), reverse=True)
                if len(items) < 2:
                    continue
                latest, previous = items[0], items[1]
                latest_dt = datetime.fromtimestamp(latest.get("recommendation_date", 0))
                if latest_dt < cutoff:
                    continue   # 最近一次评级本身就在窗口之外，跳过这家机构
                if latest.get("rating", 0) < previous.get("rating", 0):
                    inst_name = inst.get("institution_info", {}).get(
                        "institution_en_name") or inst.get("institution_info", {}).get("institution_name", "?")
                    return f"{inst_name} {previous.get('rating')}->{latest.get('rating')}"
            return ""
        finally:
            ctx.close()
    except Exception as exc:
        alert.warn(f"{ticker} 基本面评级获取失败 — {exc}")
        return None
