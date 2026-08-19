"""
research/ — Mean Reversion Research / Observation Layer (v2.9.x).

Independent, side-effect-only package: it records candidate "extreme
oversold" events for later offline analysis (see analyze_mean_reversion.py).
Nothing in this package is imported by engine/runner.py, strategies/*,
risk/*, or portfolio/* — so there is no code path by which anything here can
affect a trading decision. Same discipline as engine/entry_quality.py and
engine/hmm_shadow.py.
"""
