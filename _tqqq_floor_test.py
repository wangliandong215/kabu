from datetime import datetime
import backtest_portfolio as bp

# 只把"QQQ底仓"实际持有/交易的标的换成TQQQ，宏观/大盘天气信号继续用真实QQQ
# 判断（因为 benchmark_df 在源码里是按字面量 "US.QQQ" 赋值的，不受 QQQ_CODE
# 影响）——这样隔离出"只换持仓标的、信号不变"的杠杆效应，不牵动整个引擎的
# 宏观熔断判断。
bp.QQQ_CODE = "US.TQQQ"

stocks = list(bp.BACKTEST_STOCKS)
if "US.TQQQ" not in stocks:
    stocks.append("US.TQQQ")
if "US.QQQ" not in stocks:
    stocks.append("US.QQQ")

r = bp.run_portfolio_backtest(
    stocks=stocks,
    start="2015-01-01",
    end=datetime.now().strftime("%Y-%m-%d"),
    cash=7_000_000.0,
    news_feed=None,
    usd_to_jpy=140.0,
)

trades = [t for t in r["trade_log"] if t.get("strategy") == "core_etf"]
print(f"\n=== core_etf (TQQQ底仓) 交易记录，共 {len(trades)} 条 ===")
for t in trades:
    print(t)

sells = [t for t in trades if t["side"] == "SELL" and t.get("pnl") is not None]
buys = [t for t in trades if t["side"] == "BUY"]
wins = [t for t in sells if t["pnl"] > 0]
print(f"\n买入触发次数: {len(buys)}")
print(f"卖出(MA200跌破)次数: {len(sells)}")
if sells:
    print(f"胜率: {len(wins)/len(sells)*100:.1f}%")
    print(f"平均盈亏: {sum(t['pnl'] for t in sells)/len(sells):,.0f}")
    print(f"总盈亏: {sum(t['pnl'] for t in sells):,.0f}")
