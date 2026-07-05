"""
portfolio/capacity_manager.py -- v2.3 Portfolio Capacity Manager
（+ 2026-07-05 独立 WEAK_FULL 通道，见下方 find_weak_full_replaceable_
position() / evaluate_replacement()）。

背景：[[analytics/portfolio_gap_study.py]] 定位到82只纯美股窄池上v2.2落后
v2.0约55pp的主因是"仓位名额竞争"（Slot Capacity Competition）——
MAX_POSITIONS 限制的是"仓位数量"而不是"资金占用比例"，一个只占3%资金的
OBSERVATION持仓，跟一个占20-30%资金的FULL持仓，占用的是完全相同的一个
名额。NVDA/TSLA/AMD/AMZN等超级牛股在2015-2016年（回测早期、名额最紧张
的窗口）的首次FULL建仓机会，就是被OBSERVATION占坑的名额挡掉的。

职责边界：本模块**只做容量调度判断**，不参与任何因子评分/信号生成/风险
控制/仓位计算——`find_replaceable_position()` 是一个纯函数，给定当前持仓
快照和新信号的 Fused Score，判断"是否应该换出某个OBSERVATION持仓来腾出
名额"，返回该持仓的 code 或 None；调用方（backtest_portfolio.py /
engine/runner.py）负责在同一个 Decision Tick 内原子地执行
close(观察仓)->open(FULL信号)，本模块不执行任何交易。

触发范围（按PRD严格限定）：
  - 只有当新信号 score_label == FULL 时才会调用容量置换判断。
  - 只允许置换 score_label == OBSERVATION 的持仓（未来若要扩展到PARTIAL等
    更多低优先级仓位，在这里加，不要在调用方临时特判）。
  - 必须同时满足 incoming_score > victim_score + REPLACEMENT_MARGIN
    才允许置换，避免分数差距很小时的频繁换仓。

WEAK_FULL 独立通道（config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED，
2026-07-05新增）：v2.4曾把"换出评分转弱的FULL持仓"这个概念做进RSL框架
（portfolio/replacement_stabilizer.py 的 Tier3/FULL_DOWNGRADE_REPLACE），
带cooldown/预算/自适应门槛/Stability Score一整套约束，但那套组合PRD五项
目标全部落空已默认关闭。这里是一次独立、不带那些约束的更简单复测：
find_weak_full_replaceable_position() 判断标准与OBSERVATION完全同款
（排序规则、REPLACEMENT_MARGIN门槛都一致），唯一区别是候选池换成"总分低于
历史FULL分数分布某百分位"的FULL持仓，且仅在没有OBSERVATION候选时才会被
考虑（优先级低于OBSERVATION，不跟OBSERVATION竞争同一次置换机会）。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import config
from engine.scoring import LABEL_FULL, LABEL_OBSERVATION

REPL_OBSERVATION_EVICT = "OBSERVATION_EVICT"
REPL_WEAK_FULL_EVICT   = "WEAK_FULL_EVICT"


def _rank_candidates(pool: Dict[str, dict], current_day_idx: int):
    """(code, total_score, holding_days)，排序规则：
      1) Fused Score 由低到高——分数最弱的候选最该被换出。
      2) 分数打平时，Holding Time 由长到短——持仓最久的候选优先释放（它已
         经有过足够时间证明/证伪自己，比刚建仓的同分候选更适合让位）。
    """
    ranked = [
        (code, pos["total_score"], current_day_idx - pos.get("entry_day_idx", current_day_idx))
        for code, pos in pool.items() if pos.get("total_score") is not None
    ]
    ranked.sort(key=lambda c: (c[1], -c[2]))
    return ranked


def find_replaceable_position(incoming_score: float,
                               held_positions: Dict[str, dict],
                               current_day_idx: int) -> Optional[str]:
    """
    incoming_score  : 新FULL信号的 total_score（Fused Score，0-100）。
    held_positions  : code -> position dict。每个 dict 至少需要
                       "score_label" 和 "total_score"（Fused Score）两个键
                       才有资格被考虑；"entry_day_idx" 缺失时 Holding Time
                       按0处理（不会报错，但也不会被优先换出）。调用方应
                       预先排除不参与容量竞争的仓位（例如 core_etf 大盘
                       Beta底仓）。
    current_day_idx : 当前 Decision Tick 的交易日序号，需要跟
                       held_positions 里的 entry_day_idx 用同一个坐标系
                       （backtest_portfolio.py 里就是逐日循环的 day_idx）。

    返回：应被置换让出名额的持仓 code；如果没有满足条件的候选（不存在
    OBSERVATION持仓，或分数优势不够），返回 None——调用方此时应该回退到
    原有的 Capacity Block（放弃本次开仓）行为，不做任何仓位变动。
    """
    pool = {code: pos for code, pos in held_positions.items()
            if pos.get("score_label") == LABEL_OBSERVATION}
    candidates = _rank_candidates(pool, current_day_idx)
    if not candidates:
        return None

    victim_code, victim_score, _holding_days = candidates[0]
    if incoming_score > victim_score + config.REPLACEMENT_MARGIN:
        return victim_code
    return None


def find_weak_full_replaceable_position(incoming_score: float,
                                         held_positions: Dict[str, dict],
                                         current_day_idx: int,
                                         full_score_history: Optional[List[float]]) -> Optional[str]:
    """独立 WEAK_FULL 通道——判断标准与 find_replaceable_position 完全同款
    （排序规则、REPLACEMENT_MARGIN门槛一致），候选池换成 score_label==FULL
    且 total_score 低于历史FULL分数分布第 config.REPLACEMENT_WEAK_FULL_
    PERCENTILE 百分位的持仓。历史样本不足 config.REPLACEMENT_WEAK_FULL_
    MIN_HISTORY（或 full_score_history 为 None/空）时直接返回 None，不触发。

    调用方应仅在 find_replaceable_position() 找不到候选时才调用本函数
    （OBSERVATION优先级高于WEAK_FULL，两者不竞争同一次置换机会）。
    """
    if not full_score_history or len(full_score_history) < config.REPLACEMENT_WEAK_FULL_MIN_HISTORY:
        return None

    cutoff = np.percentile(full_score_history, config.REPLACEMENT_WEAK_FULL_PERCENTILE)
    pool = {code: pos for code, pos in held_positions.items()
            if pos.get("score_label") == LABEL_FULL and (pos.get("total_score") or 0.0) < cutoff}
    candidates = _rank_candidates(pool, current_day_idx)
    if not candidates:
        return None

    victim_code, victim_score, _holding_days = candidates[0]
    if incoming_score > victim_score + config.REPLACEMENT_MARGIN:
        return victim_code
    return None


@dataclass
class ReplacementEvaluation:
    """完整决策记录，供 Explain Layer / 回测统计使用——不管最终是否真正
    换仓，只要存在候选（不论是否满足margin）就会产出一条记录。"""
    incoming_code:    Optional[str] = None
    incoming_score:   Optional[float] = None
    victim_code:      Optional[str] = None
    replacement_type: Optional[str] = None   # REPL_OBSERVATION_EVICT / REPL_WEAK_FULL_EVICT
    old_score:        Optional[float] = None
    new_score:        Optional[float] = None
    score_difference: Optional[float] = None
    margin:           float = 0.0
    margin_satisfied: bool = False
    decision:         str = "NO_CANDIDATE"   # "REPLACE" / "KEEP" / "NO_CANDIDATE"


def evaluate_replacement(incoming_code: str,
                          incoming_score: float,
                          held_positions: Dict[str, dict],
                          current_day_idx: int,
                          full_score_history: Optional[List[float]] = None) -> ReplacementEvaluation:
    """Explain Layer 入口：完整复现一次容量置换审查的决策过程。优先看
    OBSERVATION候选，找不到（候选池为空，不是margin不够）且
    config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED 时才看 WEAK_FULL候选。
    找到候选后不论margin是否满足都会返回一条完整记录（decision="REPLACE"
    表示满足margin可以真正执行置换，"KEEP"表示候选存在但优势不够，
    "NO_CANDIDATE"表示两个通道都没有可换的持仓）。
    """
    margin = config.REPLACEMENT_MARGIN
    no_candidate = ReplacementEvaluation(incoming_code=incoming_code, incoming_score=incoming_score,
                                          margin=margin, decision="NO_CANDIDATE")

    def _record(victim_code, victim_score, replacement_type) -> ReplacementEvaluation:
        diff = incoming_score - victim_score
        satisfied = diff > margin
        return ReplacementEvaluation(
            incoming_code=incoming_code, incoming_score=incoming_score,
            victim_code=victim_code, replacement_type=replacement_type,
            old_score=victim_score, new_score=incoming_score,
            score_difference=round(diff, 4), margin=margin,
            margin_satisfied=satisfied,
            decision="REPLACE" if satisfied else "KEEP")

    # v2.4优化阶段三消融实验（2026-07-05）：三个候选变量的可选前置过滤，
    # 全部默认关闭（config.py里None/False），开启前不影响任何现有行为。
    # 见 config.py "v2.4优化阶段三" 注释块 + analytics/replace_ablation_
    # report.py 的验证结论。
    if (config.REPLACEMENT_MIN_NEW_SCORE is not None
            and incoming_score < config.REPLACEMENT_MIN_NEW_SCORE):
        return no_candidate

    obs_pool = {code: pos for code, pos in held_positions.items()
                if pos.get("score_label") == LABEL_OBSERVATION}

    if (config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE is not None
            and len(obs_pool) > config.REPLACEMENT_MAX_OBSERVATION_POOL_SIZE):
        return no_candidate

    if config.REPLACEMENT_BLOCK_SAME_SECTOR:
        incoming_sector = config.SECTOR_MAP.get(incoming_code, "other")
        obs_pool = {code: pos for code, pos in obs_pool.items()
                    if config.SECTOR_MAP.get(code, "other") != incoming_sector}

    obs_candidates = _rank_candidates(obs_pool, current_day_idx)
    if obs_candidates:
        victim_code, victim_score, _ = obs_candidates[0]
        return _record(victim_code, victim_score, REPL_OBSERVATION_EVICT)

    if config.REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED:
        if full_score_history and len(full_score_history) >= config.REPLACEMENT_WEAK_FULL_MIN_HISTORY:
            cutoff = np.percentile(full_score_history, config.REPLACEMENT_WEAK_FULL_PERCENTILE)
            wf_pool = {code: pos for code, pos in held_positions.items()
                       if pos.get("score_label") == LABEL_FULL
                       and (pos.get("total_score") or 0.0) < cutoff}
            wf_candidates = _rank_candidates(wf_pool, current_day_idx)
            if wf_candidates:
                victim_code, victim_score, _ = wf_candidates[0]
                return _record(victim_code, victim_score, REPL_WEAK_FULL_EVICT)

    return no_candidate
