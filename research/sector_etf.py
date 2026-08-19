"""
research/sector_etf.py — sector-label -> representative-ETF reverse lookup.

config.SECTOR_MAP maps individual stocks to a sector label string (used for
concentration/replacement logic); there is no existing reverse map onto a
tradeable ETF. This one is used ONLY to fetch a comparison return series for
Category 6 (Market Environment) — "Stock Return - Sector ETF Return" — never
for any trading decision.

Only labels with a genuinely representative ETF already present in
config.WATCHLIST are mapped; everything else is None rather than a guessed
fit (e.g. "materials" spans uranium/copper/aluminum miners with no single
honest ETF match). ai/quantum_space mappings are approximate best-effort
(small/niche ETFs) — noted inline.
"""

SECTOR_ETF_MAP: dict = {
    "semiconductor": "US.SMH",
    "software":      "US.IGV",
    "industrial":    "US.XLI",
    "defense":       "US.PPA",
    "ai":            "US.CHAT",  # approximate — Generative AI & Technology ETF, not AI-hardware specific
    "quantum_space":  "US.UFO",  # approximate — space-themed ETF; doesn't cover quantum computing names well

    # No honest single-ETF match in the current watchlist — never guess:
    "datacenter": None,
    "internet":   None,
    "consumer":   None,
    "biotech":    None,
    "materials":  None,
    "utility":    None,
    "automation": None,
    "telecom":    None,
    "etf":        None,
}


def sector_etf_for(sector_label: str):
    """None for an unmapped/unknown label — never fabricates an ETF."""
    return SECTOR_ETF_MAP.get(sector_label)
