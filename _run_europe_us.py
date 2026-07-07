"""Ad-hoc launcher: auto-route loop restricted to WATCHLIST_EUROPE_US only."""
import config
from engine.runner import run_loop

run_loop(
    strategy_name="combined",
    codes=config.WATCHLIST_EUROPE_US,
    ktype="K_DAY",
    bars=120,
    interval_seconds=300,
    confirmed=True,
    use_real=False,
    auto_route=True,
)
