"""
notify/telegram_bot.py — two-way Telegram bot backed by the Claude API.

Long-polls Telegram for messages from config.TELEGRAM_CHAT_ID only, forwards
each one to Claude (with read-only tools for live portfolio/quote/analysis
data), and replies with Claude's answer. Runs as a background thread inside
main.py — separate from the trading loop, read-only, never places orders.

Requires:
  - config.TELEGRAM_BOT_TOKEN / config.TELEGRAM_CHAT_ID (already used for
    outbound alerts in notify/alert.py)
  - ANTHROPIC_API_KEY environment variable (console.anthropic.com)

If either is missing, start() logs a warning and does nothing — the trading
loop is unaffected.
"""
import json
import threading
import time
from pathlib import Path

import requests

import config
from notify import alert

_OFFSET_PATH = Path(__file__).parent / "_telegram_offset.json"
_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = """你是kabu量化交易系统的Telegram助手，供系统的开发者/使用者查询持仓、行情和股票分析。

背景：kabu是一个基于moomoo OpenD的量化交易系统，当前运行在SIMULATE（模拟盘）环境。
用户在东京（JST时区）。用中文简洁回答，适合在Telegram里阅读（不要用markdown表格，
可以用简单的换行和数字列表）。

你可以调用工具获取实时数据（持仓状况、任意股票行情、单股技术指标分析）。
如果问题和kabu项目/持仓/股票无关，就正常按你自己的知识回答，不必强行调用工具。
如果调用工具失败（比如moomoo OpenD没开），如实告知用户，不要编造数据。
涉及买卖建议时提醒这只是技术指标层面的参考，不构成投资建议。"""


def _load_offset() -> int:
    if _OFFSET_PATH.exists():
        try:
            return json.loads(_OFFSET_PATH.read_text()).get("offset", 0)
        except Exception:
            return 0
    return 0


def _save_offset(offset: int) -> None:
    try:
        _OFFSET_PATH.write_text(json.dumps({"offset": offset}))
    except Exception:
        pass


def _send_message(text: str) -> None:
    url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN, method="sendMessage")
    for i in range(0, len(text), 4000):
        chunk = text[i:i + 4000]
        try:
            requests.post(
                url,
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": chunk},
                timeout=10,
            )
        except Exception as e:
            alert.log(f"telegram_bot: send failed: {e}")


def _get_updates(offset: int) -> list:
    url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN, method="getUpdates")
    resp = requests.get(
        url,
        params={"offset": offset, "timeout": 30, "allowed_updates": '["message"]'},
        timeout=35,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(body)
    return body["result"]


# ── Tools (read-only — never place orders) ────────────────────────────────────

def _build_tools():
    from anthropic import beta_tool

    @beta_tool
    def get_portfolio_positions() -> str:
        """获取当前模拟盘持仓明细：每只股票的成本价、数量、实时价格、浮动盈亏、
        策略、止损位，以及总市值/总浮盈/已实现盈亏/距历史峰值的差距。数据来自
        moomoo OpenD实时行情快照，需要OpenD正在运行并已登录。"""
        import common
        from portfolio.tracker import Portfolio

        portfolio = Portfolio()
        positions = portfolio.data.get("positions", {})
        if not positions:
            return "当前没有持仓。"

        codes = list(positions.keys())
        try:
            ctx = common.make_quote_ctx()
        except SystemExit:
            return "无法连接moomoo OpenD，请确认OpenD正在运行并已登录。"
        try:
            ret, df = ctx.get_market_snapshot(codes)
        finally:
            common.safe_close(ctx)
        if ret != 0:
            return f"获取实时行情失败：{df}"

        price_map = dict(zip(df["code"], df["last_price"]))
        lines = []
        total_cost = 0.0
        total_value = 0.0
        for code, pos in positions.items():
            entry = pos["avg_cost"]
            qty = pos["qty"]
            cost = entry * qty
            total_cost += cost
            last = price_map.get(code)
            name = config.NAME_MAP.get(code, code)
            if last:
                value = last * qty
                total_value += value
                pnl_pct = (last / entry - 1) * 100
                pnl_amt = value - cost
                trail = pos.get("trail_stop")
                line = (
                    f"{code}（{name}）数量{qty} 成本{entry:.2f} 现价{last:.2f} "
                    f"浮盈{pnl_pct:+.2f}%（{pnl_amt:+,.0f}） 策略{pos['strategy']}"
                )
                if trail:
                    line += f" 止损{trail:.2f}"
                lines.append(line)
            else:
                total_value += cost
                lines.append(f"{code}（{name}）数量{qty} 成本{entry:.2f} 现价获取失败")

        realized = portfolio.data.get("realized_pnl", 0.0)
        peak_equity = portfolio.data.get("peak_equity", 0.0)
        summary = (
            f"持仓{len(positions)}只，总成本{total_cost:,.0f}，总市值{total_value:,.0f}，"
            f"浮动盈亏{total_value - total_cost:+,.0f}，已实现盈亏{realized:+,.0f}，"
            f"历史峰值权益{peak_equity:,.0f}"
        )
        return summary + "\n" + "\n".join(lines)

    @beta_tool
    def get_stock_quote(codes: list[str]) -> str:
        """查询任意股票代码的实时行情快照（最新价）。代码格式为"市场.代码"，
        例如 US.AAPL、US.NVDA、JP.7203、HK.00700。可一次查询多只。"""
        import common

        try:
            ctx = common.make_quote_ctx()
        except SystemExit:
            return "无法连接moomoo OpenD，请确认OpenD正在运行并已登录。"
        try:
            ret, df = ctx.get_market_snapshot(codes)
        finally:
            common.safe_close(ctx)
        if ret != 0:
            return f"查询失败：{df}"

        lines = []
        for _, row in df.iterrows():
            code = row["code"]
            name = config.NAME_MAP.get(code, code)
            price = row.get("last_price") or row.get("cur_price")
            lines.append(f"{code}（{name}）最新价 {price}")
        return "\n".join(lines) if lines else "未查询到数据。"

    @beta_tool
    def analyze_stock(code: str, strategy: str = "combined") -> str:
        """对单只股票运行kabu项目的技术指标策略分析（日线K线），返回当前信号
        （BUY/SELL/HOLD）、信号强度、各子指标明细。strategy可选：ma/rsi/macd/
        boll/combined（默认combined，四指标投票）。用于回答"某只股票现在
        技术面建议买卖吗"这类问题。代码格式同get_stock_quote，例如US.NVDA。"""
        from analyze import analyze

        try:
            result = analyze(code, strategy_name=strategy, output_json=True)
        except SystemExit:
            return f"无法获取{code}的行情数据（可能是OpenD未连接，或该代码无行情权限）。"
        except Exception as e:
            return f"分析{code}失败：{e}"

        signal = result.get("signal", "HOLD")
        strength = result.get("signal_strength", 0) or 0
        price = result.get("current_price")
        pct = result.get("price_change_pct", 0) or 0
        lines = [
            f"{code} [{strategy}] 现价{price}（{pct:+.2f}%） "
            f"信号:{signal} 强度{strength:.0%}"
        ]
        sub = result.get("sub_indicators") or {}
        if sub:
            for name, ind in sub.items():
                lines.append(f"  {name}: {ind.get('signal')} {ind.get('detail', '')}")
        elif result.get("detail"):
            lines.append(f"  {result['detail']}")
        return "\n".join(lines)

    return [get_portfolio_positions, get_stock_quote, analyze_stock]


def _ask_claude(client, tools, user_text: str) -> str:
    runner = client.beta.messages.tool_runner(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": user_text}],
    )
    last_message = None
    for message in runner:
        last_message = message
    if last_message is None:
        return "（没有收到回复）"
    parts = [b.text for b in last_message.content if b.type == "text"]
    return "\n".join(parts) if parts else "（没有文本回复）"


def _handle_message(client, tools, text: str) -> None:
    if text.strip() in ("/start", "/help"):
        _send_message(
            "kabu助手已上线。直接用中文提问就行，比如"
            "「持仓状况和理由」「US.NVDA现在技术面怎么样」「随便问点别的」。"
        )
        return
    try:
        reply = _ask_claude(client, tools, text)
    except Exception as e:
        alert.log(f"telegram_bot: Claude call failed: {e}")
        reply = f"抱歉，调用Claude失败了：{e}"
    _send_message(reply)


def _run_loop() -> None:
    import anthropic

    client = anthropic.Anthropic()
    tools = _build_tools()
    offset = _load_offset()
    alert.log("telegram_bot: started, polling for messages")

    while True:
        try:
            updates = _get_updates(offset)
        except Exception as e:
            alert.log(f"telegram_bot: getUpdates failed: {e}")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or {}
            chat_id = str(message.get("chat", {}).get("id", ""))
            text = message.get("text")
            if not text or chat_id != str(config.TELEGRAM_CHAT_ID):
                continue
            _handle_message(client, tools, text)

        if updates:
            _save_offset(offset)


def start() -> None:
    """Spawn the polling loop as a daemon thread. No-op if not configured."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        alert.log("telegram_bot: KABU_TELEGRAM_BOT_TOKEN/CHAT_ID not set, skipping")
        return
    try:
        import anthropic  # noqa: F401
    except ImportError:
        alert.log("telegram_bot: `anthropic` package not installed, skipping (pip install anthropic)")
        return
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        alert.log("telegram_bot: ANTHROPIC_API_KEY not set, skipping")
        return

    thread = threading.Thread(target=_run_loop, name="telegram_bot", daemon=True)
    thread.start()
