"""
portfolio/replacement_stabilizer.py -- v2.4 Replacement Stabilization Layer (RSL).

背景：见 config.py "v2.4 Replacement Stabilization Layer" 配置块注释。v2.3的
主动置换（portfolio/capacity_manager.py::find_replaceable_position()）收益
集中在少数极端分差事件里（Replacement Success Rate仅35.5%），且没有频率/
预算/稳定性约束。RSL在v2.3之外加一层更严格的复核，本模块同样是纯函数，
无副作用、无I/O，**不修改capacity_manager.py一行代码**（严格遵守PRD"不改变
v2.3主动置换机制逻辑"的约束——decide()对Tier1只会拒绝v2.3本来会同意的置换，
不会反向放宽）。

职责边界：
  - Tier 1 (OBSERVATION_EVICT)：直接调用capacity_manager.find_replaceable_position()
    做v2.3原有判断，拿到候选后用本模块的自适应门槛（adaptive_threshold，始终
    >= config.REPLACEMENT_MARGIN）做更严格的复核。
  - Tier 2 (LOW_PARTIAL_EVICT) / Tier 3 (FULL_DOWNGRADE_REPLACE)：v2.4新增的
    两档更低优先级候选，仅在更高优先级的Tier找不到victim时才会被考虑，均受
    Stability Score保护（防止"刚涨就被换掉"），除非分数差距达到"极端"覆盖线。
  - Cooldown / Budget / 同日链式阻断：调用方（backtest_portfolio.py）负责维护
    cooldown_until / replaced_today / recent_replacement_count 状态并传入，
    本模块只做判断，不持有任何跨调用状态。

统一解释层（2026-07-04，纯文档/诊断层面的收敛，不改变上面任何一条执行逻辑）：
OBS_EVICT/LOW_PARTIAL_EVICT/FULL_DOWNGRADE_REPLACE三档、cooldown、两级budget、
Stability Score，本质上都是同一个决策的不同侧面——是否把资本从当前持仓挪到
新信号上。可以收敛成一个概念模型（仅用于解释/诊断，见下面的 explain()，
decide()的控制流完全不读取这三个变量，只是读同样的底层输入算出跟今天
完全一样的判断）：

    reallocate ⟺ Signal_Strength > Temporal_Friction_Cost
                 且 Capital_Velocity_Pressure > 0（预算未耗尽）
                 且 未被cooldown/同日链式阻断
                 且（Tier2/3）未被Stability Score保护，或分数差达到极端覆盖线

三个变量的具体定义（映射到已有代码里的量，不引入新指标）：
  Signal_Strength = incoming_score - victim_score（跟threshold同一把尺子，
      三档统一，就是各自gate里比较的那个delta）。
  Temporal_Friction_Cost = REPLACEMENT_MARGIN + volatility_factor +
      crowding_factor（= adaptive_threshold()，市场越乱/最近换仓越密集，
      摩擦越高、越不该现在动）+（仅Tier2/3）Stability Score的保护效应
      （持仓越稳，动它的隐性成本越高）。
  Capital_Velocity_Pressure = 预算/子预算的**剩余额度**（不是已用量）——
      `1 - recent_replacement_count/REPLACEMENT_BUDGET_PER_100_DAYS`
      （Tier2/3再额外乘上子预算的剩余额度）。剩余额度越大，系统越有
      "余量"去移动资本；耗尽后（<=0）直接硬拦截，不是连续摩擦。

**重要澄清（避免以后又被同一个坑绊倒）**：最初有一版草案把
volatility_factor/crowding_factor直接当成"Capital_Velocity_Pressure"的
正向压力项，但代码里这两个量实际是让threshold变高、让置换更难发生——
方向是阻力/摩擦，不是推力。已改为上面的定义：
crowding_factor（20日短窗口，连续摩擦，进adaptive_threshold）跟budget
剩余额度（100日长窗口，硬拦截，进Capital_Velocity_Pressure）是同一大类
"资金流速约束"在不同时间尺度/不同生效方式（连续 vs 硬性）上的两个分量，
语义上有重叠是刻意保留的，不是建模失误。

**再收敛（2026-07-04，同一天第二次修订）：Latent Constraint Model**——
Temporal_Friction_Cost和Capital_Velocity_Pressure不是两个独立的驱动项，
而是同一个潜变量"Constraint_State"（系统当前有多难移动资本）的两个观测
投影，分别对应连续摩擦和硬性余量两种不同的"约束生效方式"：

    Constraint_State = Temporal_Friction_Cost      （若 Capital_Velocity_Pressure > 0）
                      = +∞                          （若 Capital_Velocity_Pressure <= 0）

    reallocate ⟺ Signal_Strength > Constraint_State

这一条不等式合并了原来"threshold gate"和"budget硬拦截"两条独立判断——
预算耗尽时约束状态是无穷大，不管信号多强都过不了，跟decide()里
`if recent_replacement_count >= BUDGET: return blocked`（无条件拦截，不看
incoming_score）的真实行为完全一致；预算充足时约束状态就是连续的
Temporal_Friction_Cost，信号够强就能越过——这也是decide()的真实行为。
两个分量变量Temporal_Friction_Cost/Capital_Velocity_Pressure的定义完全
没变，只是新增了`constraint_state`/`would_reallocate`这两个组合视图
（见CapitalFlowVariables），threshold的计算逻辑、budget的判断逻辑都
原封不动。**cooldown/同日链式阻断（二元资格过滤，不是连续变量）目前仍未
纳入这个潜变量**，只隐含在"候选进不进eligible池"这一步——如果以后要
更细的解释粒度可以再补，属于已知局限，不是遗漏。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION, LABEL_PARTIAL
from portfolio.capacity_manager import find_replaceable_position

REPL_OBSERVATION_EVICT      = "OBSERVATION_EVICT"
REPL_LOW_PARTIAL_EVICT      = "LOW_PARTIAL_EVICT"
REPL_FULL_DOWNGRADE_REPLACE = "FULL_DOWNGRADE_REPLACE"


@dataclass
class ReplacementDecision:
    victim_code:      Optional[str]   = None
    replacement_type: Optional[str]   = None
    stability_score:  Optional[float] = None
    cooldown_blocked: bool            = False
    budget_blocked:   bool            = False
    # 诊断字段（不影响任何交易决策）：本次审查时，held_positions里是否存在
    # 一个score_label==OBSERVATION的持仓，但被cooldown_until/replaced_today
    # 排除出了eligible池——如果是，且最终victim_code来自Tier2/3，说明这次
    # 置换是"OBS victim被冷却挡住后，落到了另一个持仓身上"的真实spillover，
    # 而不是"根本没有OBS可换"的独立新增置换。实测（82只池全历史）：218次
    # 成功决策里这个字段0次为True——LOW_PARTIAL/WEAK_FULL不是spillover，
    # 是结构性独立的新增通道，见 REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS。
    observation_blocked_by_cooldown: bool = False
    new_tier_budget_blocked: bool = False
    # Shadow mode（config.REPLACEMENT_WEAK_FULL_ENABLED=False时）：Tier3
    # 依然正常计算，只是不会被采纳为victim_code——这里记录"如果放行，本来
    # 会换出谁"，供调用方做ablation+shadow对比分析（不影响任何交易决策，
    # 该字段非None时victim_code本身一定是None，因为没有真正执行）。
    shadow_weak_full_victim:           Optional[str]   = None
    shadow_weak_full_stability_score:  Optional[float] = None
    # 同款shadow字段，Tier1(OBSERVATION)/Tier2(LOW_PARTIAL)专用——config.
    # REPLACEMENT_OBSERVATION_TIER_ENABLED / REPLACEMENT_LOW_PARTIAL_TIER_
    # ENABLED 为False时才会非None。OBSERVATION本身不算Stability Score，
    # 所以只有LOW_PARTIAL带stability字段。
    shadow_observation_victim:         Optional[str]   = None
    shadow_low_partial_victim:         Optional[str]   = None
    shadow_low_partial_stability_score: Optional[float] = None


def adaptive_threshold(volatility_factor: float, crowding_factor: float) -> float:
    """REPLACEMENT_MARGIN(10) + 波动率缓冲 + 拥挤度缓冲。两个factor都
    是>=0的加分项，所以这个阈值永远 >= v2.3的固定10分门槛，只会更严不会更松。
    """
    return config.REPLACEMENT_MARGIN + volatility_factor + crowding_factor


def stability_score(position: dict) -> float:
    """Stability Score = holding_component x score_component x
    vol_penalty_component，落在[0,1]。三者都是"这个仓位值不值得被保护"的
    独立维度——持仓越久、分数越高、自身波动率(entry_atr/avg_cost)越低，越
    应该被保护，不轻易换出。position需要额外携带"_holding_days"（调用方在
    排序阶段算出的当前持仓天数，见_rank_candidates）。
    """
    holding_days = position.get("_holding_days", 0)
    holding_component = min(1.0, holding_days / config.REPLACEMENT_STABILITY_HOLDING_NORM_DAYS)

    total_score = position.get("total_score") or 0.0
    score_component = max(0.0, min(1.0, total_score / 100.0))

    avg_cost  = position.get("avg_cost") or 0.0
    entry_atr = position.get("entry_atr") or 0.0
    atr_pct = (entry_atr / avg_cost) if avg_cost > 0 else 0.0
    vol_penalty_component = 1.0 - min(1.0, atr_pct / config.REPLACEMENT_STABILITY_VOL_NORM_ATR_PCT)

    return holding_component * score_component * vol_penalty_component


@dataclass
class CapitalFlowVariables:
    """Capital Flow Decision Model（纯解释层，见模块docstring）——把OBS/LOW/
    FULL三档的判断逻辑统一表述成三个变量，方便跨档位比较"这次候选的信号
    强度/资金流速余量/时间摩擦成本各是多少"。只用于报表/诊断，不参与
    decide()的任何判断分支。"""
    signal_strength:           float
    capital_velocity_pressure: float
    temporal_friction_cost:    float

    @property
    def constraint_state(self) -> float:
        """Latent Constraint Model（见模块docstring）：Temporal_Friction_Cost
        和Capital_Velocity_Pressure合并成的单一潜变量。额度耗尽
        （capital_velocity_pressure<=0）时约束状态是+∞（硬拦截，任何信号
        强度都过不去，对应decide()里budget/子预算耗尽的无条件拦截）；
        额度充足时约束状态就是Temporal_Friction_Cost本身（连续摩擦，信号
        够强就能越过）。"""
        if self.capital_velocity_pressure <= 0:
            return float("inf")
        return self.temporal_friction_cost

    @property
    def would_reallocate(self) -> bool:
        """统一判断：Signal_Strength > Constraint_State——一条不等式合并
        还原decide()里"threshold gate"+"budget硬拦截"两件独立判断的组合
        效果（不含cooldown，见模块docstring的已知局限）。"""
        return self.signal_strength > self.constraint_state

    @property
    def would_pass_score_gate(self) -> bool:
        """仅看连续的Signal_Strength vs Temporal_Friction_Cost这一条，
        忽略budget硬拦截——诊断"如果不考虑预算，分数本身够不够"时用，
        比would_reallocate更细粒度（能区分"分数不够"和"预算用完"两种
        不同的"不换仓"原因）。"""
        return self.signal_strength > self.temporal_friction_cost


def explain_capital_flow(incoming_score: float, victim_score: float,
                          volatility_factor: float, crowding_factor: float,
                          recent_replacement_count: int,
                          recent_new_tier_replacement_count: Optional[int] = None,
                          victim_stability_score: Optional[float] = None) -> CapitalFlowVariables:
    """纯函数，把decide()内部已经在算的量重新表述成三变量模型——所有数值
    都能独立复算验证，不是凭空定义的新指标；decide()本身不调用这个函数，
    控制流完全不受它影响（见portfolio/test_replacement_stabilizer.py里
    explain_capital_flow的还原度测试）。

    recent_new_tier_replacement_count : 仅Tier2/3有意义（Tier1不受子预算
        约束），传None表示不考虑子预算这道栏杆。
    victim_stability_score : 仅Tier2/3有意义（Tier1不计算Stability Score），
        传None表示这个候选不受Stability Score保护。
    """
    signal_strength = incoming_score - victim_score

    main_headroom = max(0.0, 1.0 - recent_replacement_count / config.REPLACEMENT_BUDGET_PER_100_DAYS)
    if recent_new_tier_replacement_count is not None:
        sub_headroom = max(0.0, 1.0 - recent_new_tier_replacement_count
                            / config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS)
        capital_velocity_pressure = min(main_headroom, sub_headroom)
    else:
        capital_velocity_pressure = main_headroom

    temporal_friction_cost = adaptive_threshold(volatility_factor, crowding_factor)
    if (victim_stability_score is not None
            and victim_stability_score >= config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD):
        # Stability保护在真实判断里是一个阶跃：一旦达到保护门槛，有效门槛
        # 直接跳到EXTREME_ADVANTAGE_OVERRIDE（不管stability_score具体是
        # 0.5还是1.0，保护力度一样），不是连续叠加——这里用max()还原这个
        # 阶跃，而不是线性插值，避免解释层跟实际判断的形状不一致。
        temporal_friction_cost = max(temporal_friction_cost,
                                      config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE)

    return CapitalFlowVariables(
        signal_strength=signal_strength,
        capital_velocity_pressure=capital_velocity_pressure,
        temporal_friction_cost=temporal_friction_cost,
    )


def _eligible(held_positions: Dict[str, dict], cooldown_until: Dict[str, int],
              replaced_today: Set[str], current_day_idx: int) -> Dict[str, dict]:
    return {
        code: pos for code, pos in held_positions.items()
        if cooldown_until.get(code, -1) <= current_day_idx and code not in replaced_today
    }


def _rank_candidates(pool: Dict[str, dict], current_day_idx: int) -> List[Tuple[str, float, int]]:
    """(code, total_score, holding_days)，按 (分数低->高, 打平时持仓久->短) 排序
    —— 跟 capacity_manager.find_replaceable_position 同款排序哲学。"""
    ranked = [
        (code, pos["total_score"], current_day_idx - pos.get("entry_day_idx", current_day_idx))
        for code, pos in pool.items() if pos.get("total_score") is not None
    ]
    ranked.sort(key=lambda c: (c[1], -c[2]))
    return ranked


def _try_tier(incoming_score: float, pool: Dict[str, dict], current_day_idx: int,
              threshold: float, replacement_type: str) -> Optional[ReplacementDecision]:
    """在给定候选池里按排序逐个尝试：跳过被Stability Score保护、且分数差距
    未达"极端覆盖线"的候选，返回第一个通过自适应门槛的候选；全部不通过则
    返回None（调用方应继续尝试下一档，而不是直接判定为无解）。"""
    for code, victim_score, holding_days in _rank_candidates(pool, current_day_idx):
        pos = dict(pool[code])
        pos["_holding_days"] = holding_days
        score = stability_score(pos)
        delta = incoming_score - victim_score
        protected = (score >= config.REPLACEMENT_STABILITY_PROTECT_THRESHOLD
                     and delta < config.REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE)
        if protected:
            continue
        if delta > threshold:
            return ReplacementDecision(victim_code=code, replacement_type=replacement_type,
                                        stability_score=score)
    return None


def decide(incoming_code: str,
           incoming_score: float,
           held_positions: Dict[str, dict],
           current_day_idx: int,
           volatility_factor: float,
           crowding_factor: float,
           full_score_history: List[float],
           cooldown_until: Dict[str, int],
           replaced_today: Set[str],
           recent_replacement_count: int,
           recent_new_tier_replacement_count: int = 0) -> ReplacementDecision:
    """
    incoming_code/incoming_score : 触发容量审查的新FULL信号。
    held_positions   : code -> position dict（调用方已排除core_etf）。
    current_day_idx  : 当前Decision Tick的交易日序号。
    volatility_factor/crowding_factor : 调用方按当日市场状态算好传入（见
                        config.REPLACEMENT_VOL_* / REPLACEMENT_CROWDING_*），
                        adaptive_threshold()会把两者跟REPLACEMENT_MARGIN
                        相加。
    full_score_history : 滚动历史FULL标签总分列表，用于算WEAK FULL百分位
                        cutoff；样本不足REPLACEMENT_WEAK_FULL_MIN_HISTORY时
                        Tier3直接跳过，不触发。
    cooldown_until    : code -> 冷却到期day_idx（<= current_day_idx 视为已解冻）。
    replaced_today    : 今天已经参与过置换的code集合（同日链式阻断，防止
                        A->B->C->D同一天连环换仓）。
    recent_replacement_count : 滚动100交易日窗口内已发生的置换次数（任意
                        tier，调用方按 replacements 列表的 day_idx 字段统计）。
    recent_new_tier_replacement_count : 滚动窗口内Tier2(LOW_PARTIAL)+
                        Tier3(WEAK_FULL)置换次数——独立于上面的总预算，
                        专门约束这两个v2.4新增通道，不共享Tier1(OBSERVATION)
                        的额度（实测这两个新通道是纯增量而非OBS被压制后的
                        spillover，见 observation_blocked_by_cooldown 字段
                        docstring，所以需要单独的子预算而不是复用主预算）。
                        默认0（不传时不额外约束，向后兼容旧调用方）。

    返回 ReplacementDecision：victim_code为None表示本次不置换（调用方回退到
    Capacity Block）；cooldown_blocked/budget_blocked仅用于日志，不影响
    调用方的行为分支。
    """
    if recent_replacement_count >= config.REPLACEMENT_BUDGET_PER_100_DAYS:
        return ReplacementDecision(budget_blocked=True)

    # 冷却/同日阻断对beneficiary（incoming_code）同样生效——一个刚被换出或
    # 刚作为受益方进入的code，在冷却期内/当天都不能再次成为任何一次置换的
    # 受益方，防止同一资产被反复用作置换链的"进口"。
    if (cooldown_until.get(incoming_code, -1) > current_day_idx
            or incoming_code in replaced_today):
        return ReplacementDecision(cooldown_blocked=True)

    threshold = adaptive_threshold(volatility_factor, crowding_factor)
    eligible = _eligible(held_positions, cooldown_until, replaced_today, current_day_idx)
    filtered_out_by_cooldown = len(eligible) < len(held_positions)

    # 诊断：held_positions里有OBSERVATION持仓，但被cooldown/replaced_today
    # 排除出了eligible——用于区分"真spillover"（本来该换OBS，但OBS被冷却
    # 挡住，落到了Tier2/3的另一个持仓上）和"独立新增置换"（当天根本没有
    # OBS可换，Tier2/3是唯一选项，跟冷却无关）。不影响任何交易决策。
    obs_blocked = any(
        p.get("score_label") == LABEL_OBSERVATION and code not in eligible
        for code, p in held_positions.items()
    )

    # 闭包变量，Tier1/2的shadow候选算出来后统一在_tag()里贴到最终返回值上
    # （不管最后是哪个Tier真正执行了置换，都不影响shadow观测——shadow只是
    # "记录本来会发生什么"，跟真正采纳哪个Tier的结果是两件独立的事）。
    _shadow_state = {"obs_victim": None, "low_partial_victim": None,
                      "low_partial_stability": None}

    def _tag(decision: ReplacementDecision) -> ReplacementDecision:
        decision.observation_blocked_by_cooldown = obs_blocked
        if decision.shadow_observation_victim is None:
            decision.shadow_observation_victim = _shadow_state["obs_victim"]
        if decision.shadow_low_partial_victim is None:
            decision.shadow_low_partial_victim = _shadow_state["low_partial_victim"]
            decision.shadow_low_partial_stability_score = _shadow_state["low_partial_stability"]
        return decision

    # ── Tier 1: OBSERVATION —— v2.3原判断 + 自适应门槛复核 ───────────────────
    # Shadow mode (config.REPLACEMENT_OBSERVATION_TIER_ENABLED=False)：跟
    # Tier3同款模式——依然照常判断，只是不采纳，记录would-be victim后落到
    # Tier2/3继续尝试（因为这个名额事实上没被这次审查腾出来）。
    obs_pool = {c: p for c, p in eligible.items() if p.get("score_label") == LABEL_OBSERVATION}
    # 2026-07-06: 传incoming_code让Tier1也吃到v2.4阶段三转正的new_score/
    # same_sector消融门（find_replaceable_position()现在是evaluate_
    # replacement()的薄封装，见capacity_manager.py）——这两个门是全局
    # config值，不是"只给非RSL路径用"的特例，RSL Tier1复用同一份基础
    # OBSERVATION判断，理应跟着生效，再叠加自己的自适应门槛。
    victim_code = find_replaceable_position(incoming_score, obs_pool, current_day_idx,
                                             incoming_code=incoming_code)
    if victim_code is not None:
        victim_score = eligible[victim_code].get("total_score") or 0.0
        if incoming_score > victim_score + threshold:
            if config.REPLACEMENT_OBSERVATION_TIER_ENABLED:
                return _tag(ReplacementDecision(victim_code=victim_code,
                                                 replacement_type=REPL_OBSERVATION_EVICT))
            _shadow_state["obs_victim"] = victim_code

    # ── Tier2/3 子预算：两个新增通道共享一个独立于Tier1的预算，纯增量的
    # 通道不能无限叠加在Tier1之上（见 config.REPLACEMENT_NEW_TIER_BUDGET_
    # PER_100_DAYS docstring）。耗尽时直接回退，不再尝试Tier2/3。
    if recent_new_tier_replacement_count >= config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS:
        return _tag(ReplacementDecision(new_tier_budget_blocked=True))

    # ── Tier 2: LOW PARTIAL —— 新增，受Stability Score保护 ──────────────────
    # 同款shadow模式：REPLACEMENT_LOW_PARTIAL_TIER_ENABLED=False时只记录不执行。
    low_partial_pool = {
        c: p for c, p in eligible.items()
        if p.get("score_label") == LABEL_PARTIAL
        and (p.get("total_score") or 0.0) < config.REPLACEMENT_LOW_PARTIAL_THRESHOLD
    }
    decision = _try_tier(incoming_score, low_partial_pool, current_day_idx, threshold,
                          REPL_LOW_PARTIAL_EVICT)
    if decision is not None:
        if config.REPLACEMENT_LOW_PARTIAL_TIER_ENABLED:
            return _tag(decision)
        _shadow_state["low_partial_victim"]    = decision.victim_code
        _shadow_state["low_partial_stability"] = decision.stability_score

    # ── Tier 3: WEAK FULL —— 新增，需要足够历史样本才启用 ───────────────────
    # 依然照常计算候选（哪怕 REPLACEMENT_WEAK_FULL_ENABLED=False）——ablation
    # +shadow实验需要知道"如果放行，本来会换出谁"，而不是让这一档直接消失。
    if len(full_score_history) >= config.REPLACEMENT_WEAK_FULL_MIN_HISTORY:
        cutoff = np.percentile(full_score_history, config.REPLACEMENT_WEAK_FULL_PERCENTILE)
        weak_full_pool = {
            c: p for c, p in eligible.items()
            if p.get("score_label") == LABEL_FULL and (p.get("total_score") or 0.0) < cutoff
        }
        decision = _try_tier(incoming_score, weak_full_pool, current_day_idx, threshold,
                              REPL_FULL_DOWNGRADE_REPLACE)
        if decision is not None:
            if config.REPLACEMENT_WEAK_FULL_ENABLED:
                return _tag(decision)
            # Shadow mode: 不采纳这次置换，组合按"这一档不存在"往下走，只把
            # 本来会发生的事记录下来供离线分析。
            shadow = ReplacementDecision(cooldown_blocked=filtered_out_by_cooldown)
            shadow.shadow_weak_full_victim = decision.victim_code
            shadow.shadow_weak_full_stability_score = decision.stability_score
            return _tag(shadow)

    return _tag(ReplacementDecision(cooldown_blocked=filtered_out_by_cooldown))


def explain(incoming_code: str,
            incoming_score: float,
            held_positions: Dict[str, dict],
            current_day_idx: int,
            volatility_factor: float,
            crowding_factor: float,
            full_score_history: List[float],
            cooldown_until: Dict[str, int],
            replaced_today: Set[str],
            recent_replacement_count: int,
            recent_new_tier_replacement_count: int = 0) -> dict:
    """JSON可序列化的诊断输出。"decision"块是同一组输入下真实decide()调用
    的原样结果（不是重新推导的近似值——保证100%一致，因为就是直接调用
    decide()本身，见portfolio/test_replacement_stabilizer.py::TestExplainJSON
    验证）。"tiers"块补充展示每一档**排名第一的候选**（v2.3/RSL实际会
    考虑的那个，排序规则跟decide()内部完全一样）算出的Signal_Strength/
    Constraint_State，用于人工核对"为什么是这个结论"，本身不改变、也不
    参与decide()的判断。

    没有"OBSERVE/ELIGIBLE/REPLACE/COOLDOWN"状态机、没有对进来的incoming
    信号本身分类成"LOW_PARTIAL"——这两个概念在真实代码里不存在（tier是
    "被换出对象的标签"，不是"这次判断的输出分类"；持仓被换出后就卖掉了，
    不会循环回到某个"OBSERVE"状态；cooldown是二元资格过滤，不折算进
    capital_velocity_pressure的数值），故意不在这里编出来。
    """
    ground_truth = decide(incoming_code, incoming_score, held_positions, current_day_idx,
                           volatility_factor, crowding_factor, full_score_history,
                           cooldown_until, replaced_today, recent_replacement_count,
                           recent_new_tier_replacement_count)

    eligible = _eligible(held_positions, cooldown_until, replaced_today, current_day_idx)
    incoming_eligible = not (cooldown_until.get(incoming_code, -1) > current_day_idx
                              or incoming_code in replaced_today)
    budget_ok = recent_replacement_count < config.REPLACEMENT_BUDGET_PER_100_DAYS
    new_tier_budget_ok = (recent_new_tier_replacement_count
                           < config.REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS)

    def _tier_breakdown(pool: Dict[str, dict], stability_applies: bool) -> dict:
        candidates = _rank_candidates(pool, current_day_idx)
        if not candidates:
            return {"candidate": None}
        code, victim_score, holding_days = candidates[0]
        s = None
        if stability_applies:
            pos = dict(pool[code])
            pos["_holding_days"] = holding_days
            s = stability_score(pos)
        v = explain_capital_flow(
            incoming_score, victim_score, volatility_factor, crowding_factor,
            recent_replacement_count,
            recent_new_tier_replacement_count=(
                recent_new_tier_replacement_count if stability_applies else None),
            victim_stability_score=s)
        return {
            "candidate": code,
            "signal_strength": v.signal_strength,
            "temporal_friction_cost": v.temporal_friction_cost,
            "capital_velocity_pressure": v.capital_velocity_pressure,
            "constraint_state": ("inf" if v.constraint_state == float("inf") else v.constraint_state),
            "would_reallocate": v.would_reallocate,
            "stability_score": s,
        }

    obs_pool = {c: p for c, p in eligible.items() if p.get("score_label") == LABEL_OBSERVATION}
    low_partial_pool = {
        c: p for c, p in eligible.items()
        if p.get("score_label") == LABEL_PARTIAL
        and (p.get("total_score") or 0.0) < config.REPLACEMENT_LOW_PARTIAL_THRESHOLD
    }
    weak_full_pool: Dict[str, dict] = {}
    if len(full_score_history) >= config.REPLACEMENT_WEAK_FULL_MIN_HISTORY:
        cutoff = np.percentile(full_score_history, config.REPLACEMENT_WEAK_FULL_PERCENTILE)
        weak_full_pool = {
            c: p for c, p in eligible.items()
            if p.get("score_label") == LABEL_FULL and (p.get("total_score") or 0.0) < cutoff
        }

    return {
        "incoming_code": incoming_code,
        "incoming_score": incoming_score,
        "eligibility": {
            "budget_ok": budget_ok,
            "new_tier_budget_ok": new_tier_budget_ok,
            "incoming_code_eligible": incoming_eligible,
        },
        "tiers": {
            REPL_OBSERVATION_EVICT:      _tier_breakdown(obs_pool, stability_applies=False),
            REPL_LOW_PARTIAL_EVICT:      _tier_breakdown(low_partial_pool, stability_applies=True),
            REPL_FULL_DOWNGRADE_REPLACE: _tier_breakdown(weak_full_pool, stability_applies=True),
        },
        # 直接原样透传decide()的真实返回值，不重新推导——这是"decision"块
        # 100% == decide()真实输出 的保证来源。
        "decision": {
            "action":                  "REPLACE" if ground_truth.victim_code is not None else "NO_ACTION",
            "tier":                    ground_truth.replacement_type,
            "victim_code":             ground_truth.victim_code,
            "stability_score":         ground_truth.stability_score,
            "budget_blocked":          ground_truth.budget_blocked,
            "new_tier_budget_blocked": ground_truth.new_tier_budget_blocked,
            "cooldown_blocked":        ground_truth.cooldown_blocked,
        },
    }
