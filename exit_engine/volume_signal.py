# -*- coding: utf-8 -*-
"""
exit_engine/volume_signal.py — V3.2-A Volume Exhaustion / Heavy Selling
signal.

Two independently-gated checks on context.bars_5m (5-minute bars: fine
enough to react, coarse enough to not be dominated by tick noise):

  Exhaustion    — price still rising but the latest bar's volume has
                  fallen well below its recent average (momentum losing
                  participation).
  Heavy Selling — price falling on volume well above its recent average
                  (active distribution, not a quiet drift down).

Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine import indicators
from exit_engine.models import SignalReading

MODULE_NAME = "volume"

STATE_NORMAL = "NORMAL"
STATE_EXHAUSTION = "EXHAUSTION"
STATE_HEAVY_SELLING = "HEAVY_SELLING"
STATE_UNKNOWN = "UNKNOWN"


def evaluate(context) -> SignalReading:
    if not (config.ENABLE_VOLUME_EXHAUSTION or config.ENABLE_HEAVY_SELLING):
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    df = context.bars_5m
    if (df is None or "volume" not in df or "close" not in df
            or len(df) < config.EXIT_VOLUME_LOOKBACK + 2):
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "insufficient bars for volume analysis", state=STATE_UNKNOWN)

    close = df["close"].astype(float)
    prev_close = close.iloc[-2]
    if prev_close == 0:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "prior close is zero — cannot compute return", state=STATE_UNKNOWN)

    last_vol = float(df["volume"].astype(float).iloc[-1])
    avg_vol = indicators.avg_volume(df, lookback=config.EXIT_VOLUME_LOOKBACK)
    last_return = (close.iloc[-1] - prev_close) / prev_close

    points = 0.0
    states = []
    if avg_vol:
        if (config.ENABLE_VOLUME_EXHAUSTION and last_return > 0
                and last_vol < avg_vol * config.EXIT_VOLUME_EXHAUSTION_RATIO):
            points += config.EXIT_VOLUME_EXHAUSTION_POINTS
            states.append(STATE_EXHAUSTION)
        if (config.ENABLE_HEAVY_SELLING and last_return < 0
                and last_vol > avg_vol * config.EXIT_HEAVY_SELLING_RATIO):
            points += config.EXIT_HEAVY_SELLING_POINTS
            states.append(STATE_HEAVY_SELLING)

    triggered = points > 0
    state = "+".join(states) if states else STATE_NORMAL
    detail = (f"last_vol={last_vol:.0f} avg_vol={(avg_vol or 0):.0f} "
              f"last_return={last_return:+.2%}")
    return SignalReading(MODULE_NAME, True, triggered, points, detail, state=state,
                          extra={"last_volume": last_vol, "avg_volume": avg_vol,
                                 "last_return": last_return})
