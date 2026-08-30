"""
research/market_features_health.py — Research Data Health Check (v2.11.1
Priority 2).

Answers "did the daily collection task really succeed for every code/
indicator it was supposed to cover, or did it just exit 0 while quietly
dropping some items?" — reads research/collect_market_features.py's
collection_log rows (SUCCESS/NO_DATA/FAILED/SKIPPED per attempted item) and
rolls them up per category and overall.

Read-only and side-effect-free: does not import portfolio/, risk/,
strategies/, or engine/runner.py, and never affects a trading decision —
Research data being unhealthy must never block or alter live trading; this
tool only checks, logs, and reports (see module-level CLI below).

Usage:
  python -m research.market_features_health                # today's market_date
  python -m research.market_features_health --market-date 2026-08-28
"""
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import engine.market_hours as market_hours
from research.market_features_store import MarketFeaturesStore

_CATEGORIES = ("news", "options", "macro", "fedwatch")

_STATUS_SEVERITY = {"FAILED": 3, "PARTIAL": 2, "NO_DATA": 1, "SUCCESS": 0}


@dataclass
class CategoryHealth:
    status: str
    checked: int = 0
    succeeded: int = 0
    no_data: int = 0
    failed: int = 0
    skipped: int = 0
    details: List[dict] = field(default_factory=list)   # failed/no_data items


@dataclass
class HealthReport:
    market_date: str
    categories: Dict[str, CategoryHealth]
    overall: str


def _category_health(rows: List[dict]) -> CategoryHealth:
    if not rows:
        return CategoryHealth(status="SKIPPED", details=[
            {"target": "ALL", "detail": "no collection_log rows found for this market_date"}])

    counts = {"SUCCESS": 0, "NO_DATA": 0, "FAILED": 0, "SKIPPED": 0}
    details = []
    for row in rows:
        status = row.get("status", "FAILED")
        counts[status] = counts.get(status, 0) + 1
        if status in ("FAILED", "NO_DATA"):
            details.append({"target": row.get("target"), "status": status,
                             "detail": row.get("detail", "")})

    checked = len(rows)
    non_skipped = checked - counts["SKIPPED"]

    if non_skipped == 0:
        status = "SKIPPED"
    elif counts["FAILED"] == non_skipped:
        status = "FAILED"
    elif counts["FAILED"] > 0:
        status = "PARTIAL"
    elif counts["SUCCESS"] == 0 and counts["NO_DATA"] == non_skipped:
        status = "NO_DATA"
    else:
        status = "SUCCESS"

    return CategoryHealth(
        status=status, checked=checked, succeeded=counts["SUCCESS"],
        no_data=counts["NO_DATA"], failed=counts["FAILED"],
        skipped=counts["SKIPPED"], details=details,
    )


def _overall_status(categories: Dict[str, CategoryHealth]) -> str:
    non_skipped = [c for c in categories.values() if c.status != "SKIPPED"]
    if not non_skipped:
        return "SKIPPED"
    worst = max(non_skipped, key=lambda c: _STATUS_SEVERITY.get(c.status, 0))
    return worst.status


def check_daily_market_features(market_date: Optional[str] = None,
                                 store: Optional[MarketFeaturesStore] = None) -> HealthReport:
    """Build a HealthReport for `market_date` (default: today's US trading
    day per engine.market_hours.market_date_for("US") — the same value
    research/collect_market_features.py stamps its own rows with)."""
    market_date = market_date or market_hours.market_date_for("US").isoformat()
    owns_store = store is None
    store = store or MarketFeaturesStore()
    try:
        categories = {}
        for category in _CATEGORIES:
            rows = store.collection_log_rows(market_date, category)
            categories[category] = _category_health(rows)
        overall = _overall_status(categories)
        return HealthReport(market_date=market_date, categories=categories, overall=overall)
    finally:
        if owns_store:
            store.close()


def print_report(report: HealthReport) -> None:
    print(f"Market Date: {report.market_date}\n")
    for name in _CATEGORIES:
        cat = report.categories[name]
        print(f"{name.capitalize()}: {cat.status}"
              f"  (checked={cat.checked} success={cat.succeeded} "
              f"no_data={cat.no_data} failed={cat.failed} skipped={cat.skipped})")
        for d in cat.details:
            print(f"    - {d['target']}: {d.get('status', '')} {d.get('detail', '')}")
    print(f"\nOverall: {report.overall}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Research Data Health Check — reports whether the daily "
                     "collection task actually succeeded per item. Read-only, "
                     "never affects trading. See module docstring.")
    parser.add_argument("--market-date", default=None,
                         help="US trading day to check (default: today's, per "
                              "engine.market_hours.market_date_for).")
    args = parser.parse_args(argv)

    report = check_daily_market_features(market_date=args.market_date)
    print_report(report)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
