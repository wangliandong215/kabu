"""
config_binance.py — Binance connection config.

Named config_binance.py rather than config/binance.py: kabu already has a
top-level config.py (imported as `import config` all over the codebase), and
a config/ package would collide with it. This file is the Binance equivalent
of that existing config.py, kept separate on purpose.

Phase 1 only needs BINANCE_BASE_URL (public endpoints, no auth). API_KEY /
SECRET_KEY are reserved for the trading phase and stay blank until then.
"""
import os

BINANCE_BASE_URL: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

# TODO(phase 2 — trading): populate via env var once order placement lands.
API_KEY: str = os.getenv("BINANCE_API_KEY", "")
SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
