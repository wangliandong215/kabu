from datetime import datetime
from backtest_portfolio import run_portfolio_backtest, BACKTEST_STOCKS

r = run_portfolio_backtest(
    stocks=BACKTEST_STOCKS,
    start="2015-01-01",
    end=datetime.now().strftime("%Y-%m-%d"),
    cash=7_000_000.0,
    news_feed=None,
    usd_to_jpy=140.0,
)

trades = [t for t in r["trade_log"] if t.get("strategy") == "core_etf"]
print(f"\n=== core_etf (QQQ Beta底仓) 交易记录，共 {len(trades)} 条 ===")
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
else:
    print("胜率: N/A（没有卖出记录 —— 说明QQQ底仓一旦买入后再没跌破过MA200，一直持有到回测结束）")

# still-open core_etf position at end of backtest?
