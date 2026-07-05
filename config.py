"""
kabu — configuration
All tuneable parameters in one place.
Override any value via environment variable or edit this file directly.
"""
import os

# ── Connection ────────────────────────────────────────────────────────────────
OPEND_HOST: str  = os.getenv("FUTU_HOST",          "127.0.0.1")
OPEND_PORT: int  = int(os.getenv("FUTU_PORT",      "11111"))
TRD_ENV:    str  = os.getenv("FUTU_TRD_ENV",       "SIMULATE").upper()
SECURITY_FIRM: str = os.getenv("FUTU_SECURITY_FIRM", "FUTUSECURITIES").upper()
ACC_ID:     int  = int(os.getenv("FUTU_ACC_ID",    "0"))

# ── Watchlist ─────────────────────────────────────────────────────────────────
# Split into two regions per user request (2026-07-03). WATCHLIST itself stays
# a flat list (many call sites — backtest_portfolio.py, runner.py, etc. —
# assume config.WATCHLIST is a flat list of codes) but is now assembled from
# the two region lists below so the regional grouping is still visible/usable.
#
# NOTE on Taiwan / Korea: user asked for 50 Taiwan + 50 Korea stocks too.
# Verified empirically against live moomoo OpenD — TW.* and KR.* codes are
# REJECTED outright ("format of code ... is wrong"); moomoo's Market enum
# only has AU/CA/HK/JP/MY/SG/SH/SZ/US, no TW or KR. This is a hard platform
# limitation, not a lookup gap — moomoo does not offer Taiwan Stock Exchange
# or Korea Exchange market data/trading access at all. Not added; see chat
# for alternatives discussed (US-listed ADRs cover only a handful of the
# requested names, e.g. TSM/UMC/ASX for Taiwan).

# NASDAQ-100 constituents (approximate 2024-2025 composition).
# The index is rebalanced quarterly — verify at https://www.nasdaq.com/nasdaq-100
WATCHLIST_EUROPE_US: list = [
    # Technology
    "US.AAPL",  # Apple
    "US.MSFT",  # Microsoft
    "US.NVDA",  # NVIDIA
    "US.AVGO",  # Broadcom
    "US.AMD",   # Advanced Micro Devices
    "US.AMAT",  # Applied Materials
    "US.ADI",   # Analog Devices
    "US.MU",    # Micron Technology
    "US.LRCX",  # Lam Research
    "US.KLAC",  # KLA Corporation
    "US.SNPS",  # Synopsys
    "US.CDNS",  # Cadence Design
    "US.NXPI",  # NXP Semiconductors
    "US.MRVL",  # Marvell Technology
    "US.ON",    # ON Semiconductor
    "US.ASML",  # ASML Holding
    "US.QCOM",  # Qualcomm
    "US.TXN",   # Texas Instruments
    "US.CSCO",  # Cisco

    # Software & Cloud
    "US.ADBE",  # Adobe
    "US.INTU",  # Intuit
    "US.PANW",  # Palo Alto Networks
    "US.CRWD",  # CrowdStrike
    "US.FTNT",  # Fortinet
    "US.ZS",    # Zscaler
    "US.TEAM",  # Atlassian
    "US.WDAY",  # Workday
    "US.DDOG",  # Datadog
    "US.SNOW",  # Snowflake
    "US.OKTA",  # Okta
    # "US.ANSS",  # ANSYS (removed — code inactive)
    "US.VRSK",  # Verisk Analytics
    "US.ADP",   # Automatic Data Processing
    "US.PAYX",  # Paychex

    # Internet & E-Commerce
    "US.AMZN",  # Amazon
    "US.META",  # Meta Platforms
    "US.GOOGL", # Alphabet Class A
    "US.GOOG",  # Alphabet Class C
    "US.NFLX",  # Netflix
    "US.TSLA",  # Tesla
    "US.BKNG",  # Booking Holdings
    "US.MELI",  # MercadoLibre
    "US.ABNB",  # Airbnb
    "US.TTWO",  # Take-Two Interactive
    "US.EA",    # Electronic Arts
    "US.TTD",   # The Trade Desk
    "US.PDD",   # PDD Holdings

    # AI & Emerging Tech
    "US.PLTR",  # Palantir
    "US.APP",   # AppLovin
    "US.ARM",   # ARM Holdings

    # Biotech & Healthcare
    "US.AMGN",  # Amgen
    "US.GILD",  # Gilead Sciences
    "US.VRTX",  # Vertex Pharmaceuticals
    "US.REGN",  # Regeneron
    "US.ISRG",  # Intuitive Surgical
    "US.IDXX",  # IDEXX Laboratories
    "US.DXCM",  # DexCom
    "US.ALGN",  # Align Technology
    "US.GEHC",  # GE HealthCare

    # Consumer
    "US.COST",  # Costco
    "US.SBUX",  # Starbucks
    "US.MDLZ",  # Mondelez
    "US.PEP",   # PepsiCo
    "US.MNST",  # Monster Beverage
    "US.ORLY",  # O'Reilly Automotive
    "US.ROST",  # Ross Stores
    "US.KDP",   # Keurig Dr Pepper

    # Telecom & Media
    "US.TMUS",  # T-Mobile
    "US.CMCSA", # Comcast

    # Industrials & Energy
    "US.HON",   # Honeywell
    "US.PCAR",  # PACCAR
    "US.FAST",  # Fastenal
    "US.CPRT",  # Copart
    "US.CTAS",  # Cintas
    "US.BKR",   # Baker Hughes
    "US.FANG",  # Diamondback Energy
    "US.LIN",   # Linde

    # Utilities
    "US.CEG",   # Constellation Energy
    "US.EXC",   # Exelon
    "US.AEP",   # American Electric Power
    "US.XEL",   # Xcel Energy

    # ETF (index tracking)
    "US.QQQ",   # NASDAQ-100 ETF
    "US.SPY",   # S&P 500 ETF
]

# 亚太地区 — 日本50只（本地代码）+ 台湾/韩国10只（moomoo 不支持本地 TW./KR.
# 代码，用美股上市的 ADR/直接上市代替，代码仍是 US. 前缀、走美股交易时段，
# 但按公司归属分类放在这里而不是欧美地区列表）。
# 日本50只验证于 moomoo OpenD 2026-07-03；请求的东芝(6502)、丰田自动织机(6201)
# 未添加 — API 返回"未知股票"，均已因私有化交易从东京证券交易所退市。
# 台湾/韩国请求的100只里，只有这10只有 moomoo 能买到的美股对应标的，其余
# 90只（联发科/鸿海/三星电子/SK海力士/现代汽车等）无可用 ADR 或 ADR 太不
# 活跃，未添加。
WATCHLIST_ASIA_PACIFIC: list = [
    # 半导体 & 电子设备
    "JP.8035",  # 东京电子 Tokyo Electron
    "JP.285A",  # 铠侠 Kioxia
    "JP.6146",  # 迪斯科 Disco
    "JP.6920",  # Lasertec — EUV掩膜检测（原有）
    "JP.6857",  # 爱德万测试 Advantest
    "JP.4063",  # 信越化学 Shin-Etsu Chemical
    "JP.3436",  # 胜高 SUMCO
    "JP.6723",  # 瑞萨电子 Renesas Electronics
    "JP.6963",  # 罗姆半导体 ROHM
    "JP.4062",  # 揖斐电 Ibiden
    "JP.5214",  # 日本电气硝子（原有）
    "JP.6981",  # 村田制作所 Murata
    "JP.6976",  # 太阳诱电 Taiyo Yuden
    "JP.6762",  # TDK
    "JP.6971",  # 京瓷 Kyocera

    # 消费电子 & 精密仪器
    "JP.6758",  # 索尼集团 Sony Group
    "JP.7731",  # 尼康 Nikon
    "JP.7751",  # 佳能 Canon
    "JP.6752",  # 松下控股 Panasonic Holdings
    "JP.6753",  # 夏普 Sharp
    "JP.6861",  # 基恩士 Keyence
    "JP.6954",  # 发那科 Fanuc
    "JP.6645",  # 欧姆龙 Omron

    # 综合电机 & IT
    "JP.6702",  # 富士通 Fujitsu
    "JP.6501",  # 日立 Hitachi
    "JP.6503",  # 三菱电机 Mitsubishi Electric
    "JP.6701",  # NEC
    "JP.6594",  # 日本电产 Nidec
    "JP.9984",  # 软银集团 SoftBank Group

    # 汽车 & 零部件
    "JP.6902",  # 电装 Denso
    "JP.7270",  # 斯巴鲁 Subaru
    "JP.7267",  # 本田技研工业 Honda Motor
    "JP.7201",  # 日产汽车 Nissan Motor

    # 金融
    "JP.8411",  # 瑞穗金融集团 Mizuho Financial Group
    "JP.8316",  # 三井住友金融集团 Sumitomo Mitsui Financial Group
    "JP.8306",  # 三菱UFJ金融集团 Mitsubishi UFJ Financial Group

    # 综合商社
    "JP.8058",  # 三菱商事 Mitsubishi Corp
    "JP.8031",  # 三井物产 Mitsui & Co
    "JP.8053",  # 住友商事 Sumitomo Corp
    "JP.8002",  # 丸红 Marubeni
    "JP.8001",  # 伊藤忠商事 Itochu

    # 制药 & 消费
    "JP.4568",  # 第一三共 Daiichi Sankyo
    "JP.4502",  # 武田药品 Takeda Pharmaceutical
    "JP.4519",  # 中外制药 Chugai Pharmaceutical
    "JP.4503",  # 安斯泰来制药 Astellas Pharma
    "JP.4911",  # 资生堂 Shiseido
    "JP.9983",  # 迅销 Fast Retailing (优衣库)

    # 航运
    "JP.9101",  # 日本邮船 NYK Line
    "JP.9104",  # 商船三井 MOL
    "JP.9107",  # 川崎汽船 K Line

    # 台湾 / 韩国 — ADR 或美股直接上市（本地代码 moomoo 不支持，见上方说明）
    "US.TSM",   # 台积电 ADR
    "US.UMC",   # 联电 ADR
    "US.ASX",   # 日月光投控 ADR
    "US.CPNG",  # 酷澎（直接美股上市，非ADR）
    "US.PKX",   # 浦项制铁 ADR
    "US.KB",    # KB金融集团 ADR
    "US.SHG",   # 新韩金融集团 ADR
    "US.WF",    # 我们金融集团 ADR
    "US.KEP",   # 韩国电力 ADR
    "US.LPL",   # LG显示(LG Display) ADR
]

WATCHLIST: list = WATCHLIST_EUROPE_US + WATCHLIST_ASIA_PACIFIC

# ── Capital ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 50_000.0  # total account capital (USD)

# ── Risk management ───────────────────────────────────────────────────────────
MAX_POSITIONS:          int   = 10    # max simultaneous open positions (active signals only, excl. QQQ core)
STOP_LOSS_PCT:          float = 0.05  # 5% drop from entry → forced sell
TAKE_PROFIT_PCT:        float = 0.15  # 15% rise → forced sell
MAX_DRAWDOWN_PCT:       float = 0.20  # 20% portfolio drawdown → halt trading

# P1 — global exposure limits
MAX_TOTAL_EXPOSURE_PCT:  float = 0.95  # QQQ底仓+10只活跃仓位，放宽至95%
MAX_SECTOR_EXPOSURE_PCT: float = 0.50  # 半导体17只，50%允许同时持2-3只龙头

# ── v2.3 Portfolio Capacity Manager（仓位名额主动置换）─────────────────────────
# 背景：MAX_POSITIONS 限的是"仓位数量"不是"资金占用比例"，一个3%的
# OBSERVATION持仓会跟一个20-30%的FULL持仓抢同一个名额——见
# analytics/portfolio_gap_study.py 对82只窄池55pp缺口的诊断（NVDA/TSLA/
# AMD/AMZN等超级牛股的首次FULL建仓机会就是这样被挡掉的）。开启后，当
# MAX_POSITIONS已满且新信号是FULL时，允许 portfolio/capacity_manager.py
# 判断能否换出一个持有中的OBSERVATION持仓腾出名额，而不是直接放弃交易。
ENABLE_ACTIVE_REPLACEMENT: bool = True
REPLACEMENT_MARGIN: float = 10.0   # incoming FULL总分必须比被换出的持仓总分
                                    # 高出这么多，才允许换仓——避免分数差距
                                    # 很小时频繁换仓（churn）。原名
                                    # MIN_REPLACEMENT_ADVANTAGE，2026-07-05
                                    # 改名统一术语，数值/语义不变。可配置，
                                    # 便于回测不同取值（如8/10/12）对比。

# v2.4优化阶段三（2026-07-05）：REPLACEMENT_MARGIN扫描确认单一阈值已到能力
# 边界（6~20区间不生效，27~60区间非单调、无全面占优解）后，改做纯数据分析
# 找候选变量（analytics/replace_decision_feature_analysis.py），三个候选
# 里跨周期最稳健的是n_observation_pool_size(负相关)/new_score(正相关)，
# same_sector分组差异最大。下面三个开关就是这三个候选各自的消融实验开关。
#
# 消融验证结论（analytics/replace_ablation_report.py，8组配置x2个历史
# 窗口2015-2026/2019-2026，要求两个窗口同时"总收益不低于baseline-5pp+
# Sharpe不降+MDD不恶化+Replace Alpha均值不降"才算稳定）：
# pool_size（阶段二相关性最强的候选）消融后两窗口全面变差，彻底否决——
# 提醒了一次"单变量相关系数高不代表消融后有用"。new_score单独已稳定
# 通过；new_score+same_sector组合效果最强且两窗口都通过：全历史总收益
# 251.03%（>baseline 243.49%）、Sharpe 0.926（>baseline 0.876，首次
# 超过v2.4目标0.90）、MDD-18.20%（>baseline -20.45%，首次跌破v2.4目标
# 20%以内）、交易数837（落进v2.4目标800~900区间）、Replace Alpha均值
# 从-0.814%翻转为+1.770%；近期窗口2019-2026同样四项全部达标。用户
# 2026-07-05拍板采纳这个组合，正式转正为默认值（REPLACEMENT_MAX_
# OBSERVATION_POOL_SIZE维持None，只有下面两个从关闭状态转正）。
REPLACEMENT_MAX_OBSERVATION_POOL_SIZE = None    # int|None，非None时：审查时
                                                  # 持有的OBSERVATION仓位数
                                                  # 超过此值直接放弃本次换仓
                                                  # （不筛选victim，是review
                                                  # 级别的整体前置条件）。
                                                  # 消融验证失败，维持关闭。
REPLACEMENT_MIN_NEW_SCORE = 95.0                # float|None，非None时：
                                                  # incoming信号总分低于此值
                                                  # 不允许换仓。2026-07-05
                                                  # 消融验证通过，转正默认值
                                                  # （原None，阈值取自生产
                                                  # margin=10基线121次换仓
                                                  # new_score分布的25th~50th
                                                  # 分位折中，非拍脑袋）。
REPLACEMENT_BLOCK_SAME_SECTOR: bool = True       # True时：候选池排除跟
                                                  # incoming同板块
                                                  # (config.SECTOR_MAP)的
                                                  # OBSERVATION持仓，若排除后
                                                  # 还有其他跨板块候选则继续
                                                  # 换那个，不是整体放弃。
                                                  # 2026-07-05消融验证通过
                                                  # （需搭配上面REPLACEMENT_
                                                  # MIN_NEW_SCORE一起生效，
                                                  # 单独开启在全历史窗口
                                                  # 不稳定，组合后才稳定，
                                                  # 转正默认值（原False）。

# 独立 WEAK_FULL 换仓通道（不依赖下面的v2.4 RSL框架，无cooldown/budget/
# stability score保护）——当没有OBSERVATION候选可换时，允许换出一个
# score_label==FULL但total_score低于历史FULL分数分布第
# REPLACEMENT_WEAK_FULL_PERCENTILE百分位（且历史样本>=
# REPLACEMENT_WEAK_FULL_MIN_HISTORY，两个阈值定义在下面RSL配置块，标准/
# 独立两条路径共用）的持仓，判断标准与OBSERVATION同款：
# incoming_score > victim_score + REPLACEMENT_MARGIN。用户明确要求不引入
# RSL的cooldown/预算/自适应门槛/Stability Score（那套组合此前已因PRD五项
# 目标全部落空而默认关闭，见下面ENABLE_REPLACEMENT_STABILIZATION注释）——
# 这是一次独立、更简单的复测，只回答"WEAK_FULL这个概念本身（不叠加RSL其余
# 约束）值不值得引入"，不代表RSL整体结论有变化。
#
# 2026-07-05 回归验证结论（82只池2015-2026全历史，analytics/
# weak_full_standalone_report.py）：关闭时（=当前生产v2.3行为）总收益
# 243.49%/Sharpe 0.876/MDD-20.45%/951笔——跟历史记录的v2.3基线逐位一致，
# 验证了本次重构（find_replaceable_position拆出_rank_candidates公共排序
# 逻辑）零行为变化。开启后总收益暴跌到175.66%（-67.83pp）、Sharpe跌到
# 0.780，仅最大回撤有微弱改善（-18.10%，好1.35pp）；537次容量审查里220次
# 真正执行换仓（OBSERVATION_EVICT 174 + WEAK_FULL_EVICT 46），比基线
# 121次几乎翻倍——即使去掉了RSL的cooldown/预算/自适应门槛/Stability Score
# 保护，"允许换出转弱的FULL持仓"这个概念本身仍然净损害整体收益，结论跟
# 带RSL约束时测出的FULL_DOWNGRADE_REPLACE效率垫底（sampling_efficiency
# 0.192/-0.931）方向完全一致，不是RSL框架本身的问题。默认改回False，
# 保持生产锁定的v2.3行为不受影响；代码保留，用户可随时改True重新验证。
REPLACEMENT_STANDALONE_WEAK_FULL_ENABLED: bool = False

# ── v2.4 Replacement Stabilization Layer（RSL，调度稳定性控制层）───────────────
# 背景：v2.3主动置换收益集中在121次置换里的少数极端分差事件（Success Rate仅
# 35.5%），且置换本身没有频率/预算/稳定性约束。v2.4不改变v2.3的判断逻辑本身
# （portfolio/capacity_manager.py零修改），而是在其外层加一层更严格的复核，
# 详见 portfolio/replacement_stabilizer.py。
#
# 统一解释层（2026-07-04，纯文档收敛，不是新执行逻辑）：OBSERVATION_EVICT/
# LOW_PARTIAL_EVICT/FULL_DOWNGRADE_REPLACE三档 + cooldown + 两级budget +
# Stability Score，本质上是同一个问题的不同侧面——这套机制的本质是
# "三层信号强度驱动的资本流速控制系统"，不是三套互相独立的"换仓规则"。
# 统一表述成 Signal_Strength / Capital_Velocity_Pressure /
# Temporal_Friction_Cost 三个变量，具体定义和跟本文件下面各配置项的映射见
# portfolio/replacement_stabilizer.py模块docstring + explain_capital_flow()
# （纯诊断函数，decide()的判断分支不读取它，只是用同样的底层输入重新表述）。
#
# 2026-07-04 生产锁定：暂时关闭RSL，退回纯v2.3行为。82只池2015-2026全历史
# 三方回归（v2.0/v2.3/v2.4当前配置）显示：即使已经冻结了效率最差的
# FULL_DOWNGRADE_REPLACE（见上面REPLACEMENT_WEAK_FULL_ENABLED），v2.4在
# 总收益(185.53%)/Sharpe(0.783)/最大回撤(-23.80%)三项上仍全面输给v2.3
# (243.49%/0.876/-20.45%)，换手还更高(1027 vs 951笔)。用户拍板暂时使用
# v2.3——RSL的全部代码（三档+cooldown+两级budget+shadow+Latent Constraint
# Model解释层）保留不删除，只是通过这个总开关设False不生效，
# 100%复现v2.3发布时的行为（已用集成测试验证）。
ENABLE_REPLACEMENT_STABILIZATION: bool = False

# 冷却：一次置换后，victim/beneficiary双方N天内都不能再参与任何置换。
REPLACEMENT_COOLDOWN_DAYS: int = 3

# 预算：滚动100个交易日内最多允许的置换次数——15次约为v2.3实测历史基线
# （121次/11年≈每100日4~5次）的3倍留白，不是PRD字面值(80~120)，字面值相对
# 真实频率形同虚设起不到约束作用（已跟用户确认）。
REPLACEMENT_BUDGET_PER_100_DAYS: int = 15

# v2.4实测发现：LOW_PARTIAL/WEAK_FULL不是"OBS被压制后绕路"（已用spillover
# 诊断字段observation_blocked_by_cooldown验证：218次成功决策里0次命中），
# 而是两个结构性独立的新增通道，会在共享预算之外纯增量地叠加置换次数
# （82只池实测：Replacement Count 121->156，+28.9%，直接违背PRD"降低换手"
# 的初衷）。新增一个只约束这两个新档位的独立子预算，跟主预算
# REPLACEMENT_BUDGET_PER_100_DAYS分开算，不共享额度——8次约为这两档位
# 实测有机增长速率（49次/11年≈100日1.7次）的~4.7倍留白。
REPLACEMENT_NEW_TIER_BUDGET_PER_100_DAYS: int = 8
REPLACEMENT_NEW_TIER_BUDGET_WINDOW_DAYS: int = 100

# 置信缓冲区：adaptive_threshold = REPLACEMENT_MARGIN + volatility_factor
# + crowding_factor，三层（OBSERVATION/LOW_PARTIAL/WEAK_FULL）复核统一用这个
# 自适应阈值，而不是v2.3那个固定10分门槛。
REPLACEMENT_VOL_LOOKBACK_DAYS: int = 5          # 市场（QQQ）已实现波动率回看窗口
REPLACEMENT_VOL_FACTOR_MAX: float = 10.0        # 波动率百分位排名(0-100)映射到的最大缓冲分
REPLACEMENT_CROWDING_WINDOW_DAYS: int = 20      # 统计"最近置换频率"的回看窗口
REPLACEMENT_CROWDING_FACTOR_PER_EVENT: float = 2.0  # 回看窗口内每发生一次置换，缓冲分+2
REPLACEMENT_CROWDING_FACTOR_MAX: float = 10.0   # crowding_factor封顶值

# 层级替换优先级新增的两档（OBSERVATION维持v2.3原样，不受下面两个阈值影响）：
REPLACEMENT_LOW_PARTIAL_THRESHOLD: float = 70.0     # PARTIAL总分<70视为LOW PARTIAL
                                                      # （PARTIAL区间[60,80)的中点）
REPLACEMENT_WEAK_FULL_PERCENTILE: float = 70.0      # FULL持仓总分低于历史FULL分数分布
                                                      # 的第70百分位，视为WEAK FULL——
                                                      # 上面REPLACEMENT_STANDALONE_
                                                      # WEAK_FULL_ENABLED（独立通道）和
                                                      # 下面RSL Tier3共用同一对阈值。
REPLACEMENT_WEAK_FULL_MIN_HISTORY: int = 20         # 历史FULL分数样本不足此数时，
                                                      # WEAK FULL这一档直接跳过不触发

# WEAK FULL（FULL_DOWNGRADE_REPLACE）单独开关，用于ablation+shadow实验：
# 2026-07-04 execution freeze（不是删除）：sampling_efficiency ROI分解
# （analytics/sampling_efficiency_report.py，dedup修复后的干净数字）显示
# FULL_DOWNGRADE_REPLACE在2015-2026和2019-2026两个窗口里效率比分别是
# 0.192和-0.931——持续垫底且方向一致，本质是"提前终止了后续会盈利的FULL
# 持仓换取新信号"，多数情况下这笔交换不划算。但样本量仍偏小（12/6笔真实
# 执行），不足以判定这一档在时间结构里到底该不该存在，所以只冻结执行，
# 不删除判断代码：False时Tier3判断依然正常计算（历史样本够就会算），只是
# 不会真正执行换仓——继续把"本来会换出谁"记进result["shadow_replacements"]
# 供持续积累样本/换窗口复测用，组合按"这一档不存在"往下走。
# OBSERVATION_EVICT(0.945~7.134)/LOW_PARTIAL_EVICT(0.739~1.068)两档效率
# 总体站得住，保持不变（REPLACEMENT_OBSERVATION_TIER_ENABLED /
# REPLACEMENT_LOW_PARTIAL_TIER_ENABLED 仍为默认True）。
REPLACEMENT_WEAK_FULL_ENABLED: bool = False

# 同款shadow开关，Tier1(OBSERVATION)/Tier2(LOW_PARTIAL)——用于给ROI分解/
# sampling_efficiency分析提供跟WEAK FULL一样的"如果放行会换出谁+事后
# 反事实"数据，而不需要另外跑一次全局disable对比（那样会因为路径依赖，
# 从第一次分歧点开始整条模拟路径都不可比，见weak_full_ablation.py同类
# 教训）。默认True（正常执行，不进shadow）。
REPLACEMENT_OBSERVATION_TIER_ENABLED: bool = True
REPLACEMENT_LOW_PARTIAL_TIER_ENABLED: bool = True

# 稳定性评分：Stability Score = holding_component × score_component ×
# vol_penalty_component，只用于保护LOW_PARTIAL/WEAK_FULL两档（OBSERVATION不
# 受此保护，维持v2.3原有"随时可换"语义）。
REPLACEMENT_STABILITY_HOLDING_NORM_DAYS: int = 20   # 持仓天数达到此值时holding_component封顶1.0
REPLACEMENT_STABILITY_VOL_NORM_ATR_PCT: float = 0.08  # ATR/price达到8%时vol_penalty_component归零
REPLACEMENT_STABILITY_PROTECT_THRESHOLD: float = 0.5  # Stability Score≥此值即受保护
REPLACEMENT_EXTREME_ADVANTAGE_OVERRIDE: float = 30.0  # 即使受保护，分数差距≥此值仍可强制换仓

# P3 — per-trade risk budget
RISK_PER_TRADE_PCT: float = 0.02  # max 2% of total capital loss per trade

# ── Take-profit policy ────────────────────────────────────────────────────────
# Trend-following strategies rely on riding momentum — a hard take-profit cap
# kills the high-reward tail.  Set TAKE_PROFIT_TREND to None to let the ATR
# trailing stop decide when to exit trending positions.
# Mean-reversion strategies keep the hard take-profit (they trade to a target).
TAKE_PROFIT_TREND: float = None   # atr_breakout / ema_rsi: no cap, ATR trail exits
# TAKE_PROFIT_PCT (0.15) still applies to ranging / mean-reversion strategies.

# ── Signal quality gate ───────────────────────────────────────────────────────
# Only open new positions when signal strength meets this floor.
# Raising from ~0 to 0.60 cuts false entries and commission drag.
MIN_ENTRY_STRENGTH: float = 0.0   # 不过滤，动态仓位自动区分强弱信号

# ── ATR 跟踪止损 ──────────────────────────────────────────────────────────────
# 2026-07-04 曾锁定为 5.5×ATR（原 4.0）——固定ATR全历史梯度对比
# （2015-01-01~2026-07-03，142只股票池，见 _atr_fixed_comparison.py /
# _atr_fixed_comparison_result.csv）在 {4.0,5.0,5.5,6.0,6.5} 五个点里，5.5
# 曾是干净的单点最优（peak）。**这次验证是在没有保本止损锁的假设下做的**
# ——backtest_portfolio.py当时完全没实现下面的ATR_BREAKEVEN_TRIGGER逻辑，
# 只在实盘/模拟盘（risk/guard.py::update_trailing_stop()）里生效，是一处
# 真实的行为漂移，不是数值巧合掩盖，2026-07-06做v2.4 Replace决策SSoT审计
# 时才发现。
#
# 2026-07-06修复：backtest_portfolio.py补上保本锁（共享
# risk/guard.py::breakeven_lock_floor()，见该文件的注释），82只池全历史
# （v2.4当前配置：REPLACEMENT_MIN_NEW_SCORE=95+REPLACEMENT_BLOCK_SAME_
# SECTOR=True）在旧值5.5下从251.03%/Sharpe0.926/-18.20%/837笔暴跌到
# 180.49%/Sharpe0.762/-21.60%/1318笔——证明5.5这个值是在错误的止损模型
# 下选出来的，保本锁生效后需要重新扫描。重新扫描
# （_atr_resweep_with_breakeven.py，{3.0~10.0}十二点+{11,12,14,16,20,25}
# 六点补充网格）发现随着ATR变宽整体呈上升趋势但明显非单调（路径依赖），
# 且ATR>=16之后交易数不再随ATR增大而明显下降（850~950笔区间打平），说明
# 那个区间ATR跟踪止损已经名存实亡（几乎不会被触发，退出主要靠硬止损/
# 保本锁/策略信号），继续追更高的数字（比如ATR=25时总收益436.71%）大概率
# 是在拟合这一条历史路径的巧合，不是真的止损设计更优。用户拍板选择
# **ATR=14.0**——该区间仍在"止损机制还有意义"的范围内，且是局部Sharpe/
# MDD/Calmar三项都最优的点：总收益353.08%，Sharpe0.995，MDD-16.39%
# （历史上第一次同时满足v2.4最初PRD设的Sharpe>=0.90和MDD<=20%两项目标），
# Calmar0.857，交易954笔。backtest_portfolio.py::ATR_TRAIL_MULT必须跟这里
# 保持同步（两处目前是独立的硬编码值，SSoT审计已记录为已知问题，尚未
# 合并成一处，改一边记得改另一边）。
ATR_MULT_BASE:   float = 14.0   # 趋势策略跟踪止损距离：entry - 14.0×ATR
ATR_MULT_MID:    float = 14.0   # 保持不变（不再分档收紧）
ATR_MULT_TIGHT:  float = 14.0
ATR_BREAKEVEN_TRIGGER: float = 1.0  # 价格 > 入场价 + 1×ATR 时止损上移至成本价

# ── Time-frame note ──────────────────────────────────────────────────────────
# Strategies use K_DAY (daily bars) by default — indicators (ADX, ATR,
# Donchian channel) are all daily-level.  The 5-minute runner loop is simply
# a polling interval to catch same-day bar updates; it does NOT use intraday
# K-lines unless --ktype is explicitly changed.

# ── Earnings blackout ─────────────────────────────────────────────────────────
EARNINGS_BLACKOUT_DAYS: int = 1   # ban new positions ±N days around earnings

# ── Half-Kelly (equity curve state) ──────────────────────────────────────────
# When realized drawdown from peak ≥ threshold → multiply all sizing by 0.5
HALFKELLY_DRAWDOWN_THRESHOLD: float = 0.05   # 5% realized drawdown → headwind
HALFKELLY_RECOVERY_THRESHOLD: float = 0.02   # within 2% of peak → tailwind restored

# ── Pyramiding (right-side scale-in) ─────────────────────────────────────────
PYRAMID_ENABLED: bool = True
# Only add if price is within this % above avg cost (prevent chasing)
PYRAMID_MAX_ADD_ABOVE_COST: float = 0.08
# Signal strength must improve by at least this much to trigger scale-in
PYRAMID_STRENGTH_UPGRADE_MIN: float = 0.20

# ── Position sizing ───────────────────────────────────────────────────────────
# Methods: "fixed_amount" | "fixed_percent" | "fixed_qty" | "signal_based"
# signal_based: tiered % of available cash, hard-capped by RISK_PER_TRADE_PCT
SIZING_METHOD: str   = "signal_based"
SIZING_AMOUNT: float = 10_000.0   # USD per position (fixed_amount mode)
SIZING_PCT:    float = 0.10       # 10% of cash (fixed_percent mode)
SIZING_QTY:    int   = 10         # fixed share count (fixed_qty mode)

# signal_based tiers (% of available cash, before risk cap and strategy cap)
SIZING_PCT_STRONG: float = 0.20   # strength >= 0.7, base allocation
SIZING_PCT_MEDIUM: float = 0.10   # strength >= 0.4
SIZING_PCT_WEAK:   float = 0.03   # strength <  0.4  (legacy strength-threshold
                                   # fallback tier, used only when score_label
                                   # is not passed — e.g. PROMOTE/pyramid calls)

# v2.2: OBSERVATION 档（engine.scoring 总分40-59分）的独立仓位比例——不复用
# SIZING_PCT_WEAK（那是给不传 score_label 的旧路径用的），单独开一个配置项
# 是因为PRD明确要求"不得硬编码"，需要能独立调优而不影响旧路径。默认3%，
# 跟SIZING_PCT_WEAK当前数值恰好相同只是历史延续，不代表两者必须联动。
OBSERVATION_POSITION_PCT: float = 0.03

# Dynamic sizing: when fewer open positions than this threshold, a strong
# signal is allowed a larger slice (SIZING_PCT_STRONG_DYNAMIC) so idle cash
# works harder without breaching RISK_PER_TRADE_PCT discipline.
DYNAMIC_SIZING_MAX_OPEN: int   = 3      # use expanded size when ≤ N positions open
SIZING_PCT_STRONG_DYNAMIC: float = 0.30 # expanded strong allocation (when slots are free)

# Momentum-weighted sizing (RSI-14): scales the tier % above, NOT the risk
# cap or strategy cap — a strong-momentum entry gets a bigger slice of the
# SAME risk budget, it never risks more than RISK_PER_TRADE_PCT per trade.
# atr_breakout + RSI>threshold  → bigger bullet (real breakout, not noise)
# boll/boll_mr + RSI<threshold  → smaller bullet (left-side bounce, size down)
DYNAMIC_SIZING_RSI_STRONG: float      = 60.0
DYNAMIC_SIZING_RSI_WEAK:   float      = 45.0
DYNAMIC_SIZING_MULT_STRONG: float     = 1.5
DYNAMIC_SIZING_MULT_WEAK:   float     = 0.6
DYNAMIC_SIZING_STRONG_STRATEGIES: set = {"atr_breakout"}
DYNAMIC_SIZING_WEAK_STRATEGIES:   set = {"boll", "boll_mr"}

# Per-strategy position cap — overrides tier sizing when lower.
# atr_breakout raised to 30% (from 20%) to match expanded sizing ceiling.
# Mean-reversion strategies (boll) are left-side trades: cap at 10%.
STRATEGY_MAX_SIZE_PCT: dict = {
    "boll":         0.10,
    "boll_mr":      0.10,
    "ma":           0.15,
    "ma_rsi":       0.15,
    "ema_rsi":      0.20,
    "atr_breakout": 0.30,   # ← raised from 0.20
    "atr_breakout_early": 0.30,  # same cap as atr_breakout — the trial-size
                                  # trim to 30% of normal is applied via
                                  # TRENDING_EARLY_POSITION_SCALE below, not
                                  # via a separate cap, so "normal position
                                  # size" (the promotion target) is well-defined.
    "macd":         0.15,
    "combined":     0.15,
    "rsi":          0.10,
}

# Stop loss distance caps per tier — if technical stop exceeds cap, auto-downgrade
STOP_MAX_STRONG: float = 0.10   # strong tier: stop must be ≤ 10% from entry
STOP_MAX_MEDIUM: float = 0.12   # medium tier: stop must be ≤ 12% from entry

# ── Dynamic Strategy Routing (Market Regime Switching) ───────────────────────
# Maps detected regime → strategy name.  Set to None to skip that regime.
REGIME_STRATEGY_MAP: dict = {
    "TRENDING_UP":    "atr_breakout",       # ride breakout momentum
    "TRENDING_EARLY": "atr_breakout_early", # same breakout, pre-ADX, light size
    "TRENDING_DOWN":  None,                 # no long entries into downtrend
    "RANGING":        "boll",               # mean reversion, 60%+ win rate
    "VOLATILE":       None,                 # high-churn noise — skip, no alpha here
}

# ── TRENDING_EARLY trial-position controls ────────────────────────────────────
# Trial entry: only open TRENDING_EARLY_POSITION_SCALE of what a normal
# atr_breakout entry would size to. If the regime later confirms to
# TRENDING_UP (ADX crosses 25) while the trial is still held, top it up to
# full size (see engine/runner.py "promote" step / backtest_portfolio.py
# equivalent). If it gets stopped out (-STOP_LOSS_PCT) within
# TRENDING_EARLY_STOPOUT_LOOKBACK_DAYS trading days of entry — still
# unpromoted — the stock is blocked from new entries for
# TRENDING_EARLY_COOLDOWN_DAYS trading days (prevents repeated whipsaw
# re-entries at the top of a range in choppy markets).
TRENDING_EARLY_POSITION_SCALE:        float = 0.3
TRENDING_EARLY_STOPOUT_LOOKBACK_DAYS: int   = 5
TRENDING_EARLY_COOLDOWN_DAYS:         int   = 10

# ── QQQ macro technical halt (config side) ────────────────────────────────────
# See engine/regime.py::qqq_macro_halt_series for the detection logic (below
# MA200 + steep MA20 downslope) — hard-blocks new entries, locked as of
# v1.0-RELEASE-FINAL. Position-scale and dual-watchlist alternatives were
# tried and reverted (both tested worse on the full 2015-2026 sample); see
# project memory before reintroducing either.

# ── Market Weather regime classifier（模块一，2026-07-04 新增）──────────────
# 见 engine/market_weather.py。只喂给下面的风险乘数，不替换/不修改
# engine.regime.qqq_macro_halt_series（v1.0-RELEASE-FINAL 锁定不动）。
# 状态0 并入 qqq_macro_halt 已有的 buy_candidates=[]/macro_block 同一个拦截
# 开关，不新建平行拦截逻辑。
MARKET_WEATHER_MA_SHORT: int = 20
# 长期均线复用 QQQ_MA_PERIOD(200)，不单独开常量，避免两个"200"各自漂移。
MARKET_WEATHER_VOL_WINDOW: int = 20
MARKET_WEATHER_VOL_BASELINE_WINDOW: int = 60
MARKET_WEATHER_VOL_EXPANSION_MULT: float = 1.3
MARKET_WEATHER_VOL_EXTREME_MULT:   float = 2.0

# ── 动态大盘风险系数（模块二，2026-07-04 重构）───────────────────────────────
# 第一版实现曾对 atr_breakout/ma_rsi 这类本来就用 2×ATR 做止损的策略，另外
# 叠加一个独立的"1%风险/(ATR×2)"上限——这跟现有 RISK_PER_TRADE_PCT(2%) 用
# 同一个止损距离算出来的风险上限完全重复，只是比例更严，等于无条件砍半仓位，
# 全历史回测验证：总收益277.18%→206.93%，最大回撤还从-19.85%恶化到-20.79%
# （见 _market_weather_ab.py，V0/V1两组数字完全相同也印证了状态0拦截本身
# 没有额外代价，代价全部来自这个重复约束）。已改为直接把 RISK_PER_TRADE_PCT
# 本身按大盘状态动态缩放，不再新增平行公式：
#   风险预算 = total_capital * RISK_PER_TRADE_PCT * MARKET_WEATHER_RISK_MULTIPLIER[code]
# 这个风险预算除以策略自己的止损距离（stop_loss_pct，趋势策略已经是2×ATR/price，
# 均值回归策略是固定5%）就是 risk/sizing.py::calculate() 里已有的 qty_by_risk
# 公式——不需要额外的 ATR/1% 参数，天然对不同策略的真实止损类型正确。
#
# 状态2(1.0)=正常2%风险预算；状态1(1.0)=全历史回测验证减半(0.5)不划算——
# 收益240.91%→218.45%、Sharpe 0.891→0.88都更差，最大回撤只从-20.37%微改善
# 到-19.9%，用户拍板状态1不减仓，维持满额2%；状态0(0.0)=禁止新开仓（已持
# 仓位仍按自己的止损/策略SELL规则退出）——状态0触发条件复用
# engine.regime.qqq_macro_halt_series 而不是单纯"跌破MA200"，原因见
# engine/market_weather.py 顶部注释（单纯跌破MA200触发拦截会多损失~13%收益、
# 回撤却没有改善）。以上均见 _market_weather_ab_result.csv 最终版数字。
MARKET_WEATHER_RISK_MULTIPLIER: dict = {2: 1.0, 1: 1.0, 0: 0.0}

# 单股仓位硬顶（模块二）：无论风险预算算出多少股，都不能超过总资产的这个比例。
# 独立于上面的风险预算缩放，是资金集中度上限，不是风险倒推的一部分。
MARKET_WEATHER_MAX_POSITION_PCT: float = 0.20

# ── Sector map (for correlation / concentration limits) ───────────────────────
SECTOR_MAP: dict = {
    # Semiconductors
    "US.NVDA": "semiconductor", "US.AMD":  "semiconductor",
    "US.AVGO": "semiconductor", "US.AMAT": "semiconductor",
    "US.ADI":  "semiconductor", "US.MU":   "semiconductor",
    "US.LRCX": "semiconductor", "US.KLAC": "semiconductor",
    "US.NXPI": "semiconductor", "US.MRVL": "semiconductor",
    "US.ON":   "semiconductor", "US.ASML": "semiconductor",
    "US.QCOM": "semiconductor",
    "US.TXN":  "semiconductor", "US.SNPS": "semiconductor",
    "US.CDNS": "semiconductor",
    "JP.285A": "semiconductor", "JP.6146": "semiconductor",
    "JP.8035": "semiconductor", "JP.6920": "semiconductor",
    "JP.6857": "semiconductor", "JP.4062": "semiconductor",
    "JP.5214": "semiconductor",
    # Software & Cloud
    "US.MSFT": "software", "US.ADBE": "software", "US.INTU": "software",
    "US.PANW": "software", "US.CRWD": "software", "US.FTNT": "software",
    "US.ZS":   "software", "US.TEAM": "software", "US.WDAY": "software",
    "US.DDOG": "software", "US.SNOW": "software", "US.OKTA": "software",
    "US.VRSK": "software", "US.ADP":  "software", "US.PAYX": "software",
    # Internet & Consumer Tech
    "US.AAPL": "internet", "US.AMZN": "internet", "US.META": "internet",
    "US.GOOGL":"internet", "US.GOOG": "internet", "US.NFLX": "internet",
    "US.TSLA": "internet", "US.BKNG": "internet", "US.MELI": "internet",
    "US.ABNB": "internet",
    "US.TTWO": "internet", "US.EA":   "internet", "US.TTD":  "internet",
    "US.PDD":  "internet",
    # AI & Emerging
    "US.PLTR": "ai", "US.APP": "ai", "US.ARM": "ai",
    # Biotech
    "US.AMGN": "biotech", "US.GILD": "biotech", "US.VRTX": "biotech",
    "US.REGN": "biotech", "US.ISRG": "biotech", "US.IDXX": "biotech",
    "US.DXCM": "biotech", "US.ALGN": "biotech", "US.GEHC": "biotech",
    # Consumer
    "US.COST": "consumer", "US.SBUX": "consumer", "US.MDLZ": "consumer",
    "US.PEP":  "consumer", "US.MNST": "consumer", "US.ORLY": "consumer",
    "US.ROST": "consumer", "US.KDP":  "consumer",
    # Telecom
    "US.TMUS": "telecom", "US.CMCSA": "telecom",
    # Industrial & Energy
    "US.HON":  "industrial", "US.PCAR": "industrial", "US.FAST": "industrial",
    "US.CPRT": "industrial", "US.CTAS": "industrial",
    "US.BKR":  "industrial", "US.FANG": "industrial", "US.LIN":  "industrial",
    # Utilities
    "US.CEG": "utility", "US.EXC": "utility",
    "US.AEP": "utility", "US.XEL": "utility",
    # ETF
    "US.QQQ": "etf", "US.SPY": "etf",
}

# ── QQQ 大盘 Beta 动态垫底（2026-07-03 改造，按用户PRD）───────────────────────
# 固定划出 QQQ_CORE_TARGET_PCT（20%~30%区间，默认25%）长期持有 QQQ，是"死仓"
# 不是"资金不足时的兜底填充"——不再因为个股信号出现就卖出腾资金（旧版本会,
# 已移除该逻辑）。唯一进出场条件：QQQ 收盘跌破/站上 MA200（大趋势走坏/修复），
# 只要在 MA200 上方就一直持有，不因为浮盈浮亏做任何调整。个股引擎实际能用的
# 仓位空间 = MAX_TOTAL_EXPOSURE_PCT − QQQ_CORE_TARGET_PCT（约70%），因为
# exposure_pct() 统计口径包含 QQQ 底仓成本，两者共享同一个 95% 硬顶。
QQQ_CORE_CODE:         str   = "US.QQQ"
QQQ_CORE_TARGET_PCT:   float = 0.25   # 固定目标仓位占比，20%~30%区间内可调
QQQ_MA_PERIOD:         int   = 200    # 跌破即清仓（大趋势走坏），重新站上再买回

# ── Data cache ────────────────────────────────────────────────────────────────
CACHE_DIR:          str = os.path.join(os.path.expanduser("~"), ".kabu_cache")
CACHE_TTL_DAILY:    int = 3600 * 6   # 6 h for 1d / 1w / 1M bars
CACHE_TTL_INTRADAY: int = 300        # 5 min for 1m–60m bars

# ── Notifications ─────────────────────────────────────────────────────────────
# Set env var KABU_DINGTALK_WEBHOOK to receive DingTalk alerts.
DINGTALK_WEBHOOK: str = os.getenv("KABU_DINGTALK_WEBHOOK", "")
