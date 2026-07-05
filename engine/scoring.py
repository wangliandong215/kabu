"""
engine/scoring.py — v2.1 横向多因子总分模型：把"趋势/基本面/新闻/天气"四个
独立评分源合成一个 0-100 总分，再换算成喂给 risk/sizing.py::calculate() 的
position_scale。

这是对旧"链式过滤器"（strength 分档 -> 天气乘数 -> 新闻硬阈值/加成 -> 风险
上限取 min）的替代——但只替代"信号准入/评分"这一层，不改动
sizing.calculate() 内部的风险上限/策略上限逻辑（RISK_PER_TRADE_PCT、
MARKET_WEATHER_MAX_POSITION_PCT 等不变）。这里算出的 position_scale 直接
传给 calculate() 已有的同名参数（文档写明"trims the result, never expands
it"），如果同时命中 TRENDING_EARLY 之类的既有仓位系数，两者相乘，不新增
平行的裁剪路径。

评分权重：趋势40% + 基本面20% + 新闻20% + 天气20%。

**缺失数据处理（2026-07-04 修正，用户指出并采纳 Method A）**：
fundamental_score / news_score 传 None 表示"这个维度当前没有真实数据"
（回测环境没有历史基本面/新闻数据源；paper/live 环境下 API 调用失败或无
数据），此时**直接把该维度从加权平均里剔除，剩余权重按比例重新归一化**，
而不是塞一个固定占位分参与平均。第一版试过固定70分/固定100分，两者都
不对：70分让"缺失数据"数学上等价于"轻微利空"，系统性拖累仓位；100分
虽然让回测数字更接近基线，但"缺失数据当满分参与平均"同样是编造不存在
的信息，只是这次权重占比恰好让副作用不明显。

这个修正后的公式在"只有趋势+天气"时（回测无 --news-file 的常态）几乎
精确复现旧链式过滤器 strength=0.4/0.7 的分档边界——但**这只是边界位置
巧合对齐，不代表行为对齐**：实测发现即使总分模型算出的 FULL/PARTIAL 标签
拿去替代旧的 strength 阈值选档位，回测数字与"仍用旧阈值选档位"完全一样
（byte-for-byte），因为两者在只有趋势的情况下本来就选中同一个档位——真正
让回测收益从240.91%掉到148.37%的，是下面这条：PARTIAL档(60-79分)继续用
`total/100` 做连续折算，而 total 在只有趋势+天气时纯粹是 trend_strength
的线性变换，等于又把 trend_strength 自己打了一次折。这不是"总分模型综合
判断后决定轻仓试错"，只是"trend不够强所以打折trend"——不是PRD想要的东西。

**折算生效条件（2026-07-04 二次修正，用户确认）**：PARTIAL档的连续折算
只在 fundamental_score 或 news_score **至少有一项真实参与加权**时才生效
（`has_real_signal`）；两项都缺失时（回测没有 --news-file 的常态）
position_scale 固定为 1.0，等同旧 medium 档满仓，不做任何折算。只有当
非趋势信息真实存在、且总分因此被拉进 PARTIAL 区间时，才是名副其实的
"信号带瑕疵、轻仓试错"。paper/live 环境下 news 几乎总是有真实数据（news_
filter.classify_code 只要能连上 moomoo 就会返回真实tier，不会是 None），
所以这条例外主要只在回测里生效。

天气只做二元否决（用户已确认，2026-07-04）：state0=0分，state1/state2都是
满分100，不对state1(震荡)做连续减分——项目历史上"状态1减仓"/"Risk-Off
半仓"两次回测都证明"不划算"（代价20-28%收益换取轻微回撤改善），把天气
做成连续三档评分会在数学上变相重新引入这个已被否决的机制，所以这里刻意
只用二元。天气这一维永远参与加权（不会是 None）——weather_code 总是能算
出来（fetch/compute 失败时 engine/runner.py 自己会 fallback 到保守的
code=1，不会传 None 进来）。

**v2.2 新增 OBSERVATION 档（2026-07-04，按用户PRD实现）**：v2.1 的"总分<60
放弃开仓"这条硬过滤，实测让v2.0里本来会以3%观察仓下单的弱技术面信号
（fund/news缺失、trend<0.4）完全消失，回测显示这批交易历史上整体是净
正贡献（见 project memory "v2.1：横向多因子总分架构上线"一节最后的待办）。
v2.2 把 SKIP 和 PARTIAL 之间的死区拆成一个新的 OBSERVATION 状态（40-59分），
定位是"技术面弱信号观察仓"而不是直接放弃交易；真正的一票否决（黑天鹅
新闻/天气state0/total<40）仍然是 SKIP。OBSERVATION 属于 Decision Layer，
只产生一个状态标签，不决定具体仓位百分比——具体的3%仓位比例由 Position
Layer（risk/sizing.py::calculate() 的 score_label 分支）读取
`config.OBSERVATION_POSITION_PCT` 决定，这里保持 position_scale=1.0（不做
总分模型的额外折算），职责边界跟 PARTIAL/FULL 的现有设计一致。

准入规则（基于加权/归一化后的总分 `total`）：
  weather_code == 0  或  总分 < 40                 -> SKIP     position_scale = 0（彻底放弃开仓）
  40 <= 总分 < 60                                   -> OBSERVATION  position_scale = 1.0（具体3%仓位由 Position Layer 决定）
  60 <= 总分 < 80  且 fund/news 至少一项真实参与    -> PARTIAL  position_scale = 总分/100
  60 <= 总分 < 80  且 fund/news 都缺失(None)        -> PARTIAL  position_scale = 1.0（等同旧medium档满仓）
  总分 >= 80                                         -> FULL     position_scale = 1.0（满仓，等同旧行为）
"""
from typing import NamedTuple, Optional

TREND_WEIGHT       = 0.4
FUNDAMENTAL_WEIGHT = 0.2
NEWS_WEIGHT        = 0.2
WEATHER_WEIGHT      = 0.2

OBSERVATION_THRESHOLD = 40.0   # 总分低于此值，彻底放弃开仓（SKIP）
PARTIAL_THRESHOLD     = 60.0   # 总分达到此值，进入部分仓位区间（PARTIAL）
FULL_THRESHOLD         = 80.0   # 总分达到此值，视为强烈信号，满仓放行（FULL）

LABEL_SKIP        = "SKIP"
LABEL_OBSERVATION = "OBSERVATION"
LABEL_PARTIAL     = "PARTIAL"
LABEL_FULL        = "FULL"


class TotalScore(NamedTuple):
    total: float
    position_scale: float
    label: str
    trend_score: float
    fundamental_score: Optional[float]   # None = 该维度缺失数据，未参与加权
    news_score: Optional[float]          # None = 该维度缺失数据，未参与加权
    weather_score: float


def compute_total_score(trend_strength: float,
                         weather_code: int,
                         fundamental_score: Optional[float] = None,
                         news_score: Optional[float] = None) -> TotalScore:
    """
    trend_strength    : 0-1，策略产出的 signal strength。
    weather_code      : 2/1/0，engine.market_weather.market_weather() 的
                         最新状态码。永远参与加权，不会缺失。
    fundamental_score : 0-100，或 None 表示当前没有真实基本面数据（回测里
                         始终是 None；paper/live 只有 engine.fundamental.
                         score() 真正跑通判定时才是数值，API失败/无数据仍是
                         None）。None 时从加权平均里剔除，不参与、不拖累。
    news_score        : 0-100，或 None 表示当前没有真实新闻数据（回测在没有
                         --news-file 时是 None；paper/live 永远有真实抓取
                         能力，classify_code() 不会返回 None）。
    """
    trend_score   = max(0.0, min(1.0, trend_strength)) * 100.0
    weather_score = 0.0 if weather_code == 0 else 100.0

    weighted_sum = trend_score * TREND_WEIGHT + weather_score * WEATHER_WEIGHT
    total_weight = TREND_WEIGHT + WEATHER_WEIGHT

    if fundamental_score is not None:
        weighted_sum += fundamental_score * FUNDAMENTAL_WEIGHT
        total_weight += FUNDAMENTAL_WEIGHT
    if news_score is not None:
        weighted_sum += news_score * NEWS_WEIGHT
        total_weight += NEWS_WEIGHT

    # 重新归一化到 0-100。四舍五入到小数点后6位再做阈值比较——权重相加
    # （如0.4+0.2）在浮点数下不精确等于0.6（实测59.99999999999999），会让
    # 恰好落在60/80边界的分数被误判成低一档，这里先消掉这个浮点噪音。
    total = round(weighted_sum / total_weight, 6)

    if weather_code == 0 or total < OBSERVATION_THRESHOLD:
        return TotalScore(round(total, 2), 0.0, LABEL_SKIP,
                           trend_score, fundamental_score, news_score, weather_score)
    if total < PARTIAL_THRESHOLD:
        # OBSERVATION：技术面弱信号观察仓。不在这里决定具体仓位比例——
        # position_scale=1.0 表示"不做总分模型的额外折算"，3%观察仓比例由
        # risk/sizing.py::calculate() 读取 config.OBSERVATION_POSITION_PCT
        # 决定（Decision Layer 只判断状态，Position Layer 决定仓位%）。
        return TotalScore(round(total, 2), 1.0, LABEL_OBSERVATION,
                           trend_score, fundamental_score, news_score, weather_score)
    if total < FULL_THRESHOLD:
        # PRD 的"60-79按总分比例折算"这条连续折扣，只在基本面/新闻至少有
        # 一项真实参与加权时才生效——如果两项都缺失（回测没有--news-file的
        # 常态），total 在这个区间纯粹是 trend_strength 的线性变换，继续按
        # total/100 打折等于又把trend自己打了一次折，不是"总分模型综合
        # 判断后决定轻仓试错"，只是"trend不够强所以打折trend"，这不是
        # PRD想要的东西。此时视同旧medium档满仓（position_scale=1.0），
        # 只有真的有非趋势信息参与、且总分因此被拉到PARTIAL区间时，才是
        # 真正意义上的"信号带瑕疵、轻仓试错"（2026-07-04，用户确认采纳）。
        has_real_signal = fundamental_score is not None or news_score is not None
        scale = round(total / 100.0, 4) if has_real_signal else 1.0
        return TotalScore(round(total, 2), scale, LABEL_PARTIAL,
                           trend_score, fundamental_score, news_score, weather_score)
    return TotalScore(round(total, 2), 1.0, LABEL_FULL,
                       trend_score, fundamental_score, news_score, weather_score)
