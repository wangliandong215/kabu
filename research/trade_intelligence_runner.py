"""
research/trade_intelligence_runner.py — V3.6-A Trade Intelligence
orchestration entry point (spec section 十七/十八, V3.6-A only).

Pure research/shadow-mode pipeline: trade_intelligence_data.load_dataset()
-> run the four in-scope analysis modules (Early Failure, Exit,
Signal/Confidence, Position Size) -> each analyze() already gates its own
findings through trade_intelligence_patterns.classify_state() -> upsert the
kept patterns into the V3.6-A research DB -> generate + write a report.

NEVER imported by engine/runner.py or main.py — invoke manually or via an
external Windows Task Scheduler entry you set up yourself, same convention
as research/early_failure_monitor.py (see that module's docstring for the
rationale: wiring an automatic call into the live trading pass would mean
editing engine/runner.py, which this round of work was explicitly told to
avoid).

Usage:
    python -m research.trade_intelligence_runner [--execution PAPER]
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from research import trade_intelligence_data as data
from research import trade_intelligence_early_failure as early_failure
from research import trade_intelligence_exit as exit_analysis
from research import trade_intelligence_signal_confidence as signal_confidence
from research import trade_intelligence_position_size as position_size
from research import trade_intelligence_report as report
from research.trade_intelligence_narrative import get_narrative_provider
from research.trade_intelligence_store import TradeIntelligenceStore

_GENERATOR_VERSION = "trade_intelligence_runner.v1"
_ANALYSIS_MODULES = (early_failure, exit_analysis, signal_confidence, position_size)


def refresh(source_db_path: Optional[Path] = None,
            research_db_path: Optional[Path] = None,
            report_dir: Optional[Path] = None,
            execution: Optional[str] = "REAL") -> dict:
    """Run one full V3.6-A analysis pass. source_db_path/research_db_path/
    report_dir let tests point at temp paths instead of the production
    trade_history.db / trade_intelligence.db / reports dir — see
    research/test_trade_intelligence_runner.py."""
    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now().isoformat()
    resolved_source = str(source_db_path or data._DB_PATH)
    store = TradeIntelligenceStore(db_path=research_db_path)
    try:
        df = data.load_dataset(source_db_path, execution=execution)

        patterns = []
        for module in _ANALYSIS_MODULES:
            patterns.extend(module.analyze(df))

        for candidate in patterns:
            store.upsert_pattern(candidate, run_id=run_id,
                                  generator_version=_GENERATOR_VERSION)

        n_candidate = sum(1 for p in patterns if p.state == "CANDIDATE_PATTERN")
        n_observation = sum(1 for p in patterns if p.state == "OBSERVATION")
        finished_at = datetime.now().isoformat()
        run_summary = {
            "run_id": run_id, "started_at": started_at, "finished_at": finished_at,
            "source_db_path": resolved_source,
            "n_trades_analyzed": int(len(df)) if df is not None else 0,
            "n_patterns_total": len(patterns),
            "n_candidate_patterns": n_candidate, "n_observations": n_observation,
        }

        provider = get_narrative_provider()
        payload = report.build_report(patterns, run_summary, provider)
        paths = report.write_report(payload, report_dir=report_dir)
        run_summary["report_path"] = str(paths.get("run_md"))

        store.record_run(
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            source_db_path=resolved_source,
            n_trades_analyzed=run_summary["n_trades_analyzed"],
            n_patterns_total=run_summary["n_patterns_total"],
            n_candidate_patterns=n_candidate, n_observations=n_observation,
            report_path=run_summary["report_path"], status="OK",
        )
        return run_summary
    except Exception as exc:
        store.record_run(
            run_id=run_id, started_at=started_at, finished_at=datetime.now().isoformat(),
            source_db_path=resolved_source, n_trades_analyzed=0, n_patterns_total=0,
            n_candidate_patterns=0, n_observations=0, report_path=None,
            status="ERROR", error_message=f"{exc.__class__.__name__}: {exc}",
        )
        raise
    finally:
        store.close()


if __name__ == "__main__":
    import sys
    execution = "REAL"
    if "--execution" in sys.argv:
        execution = sys.argv[sys.argv.index("--execution") + 1]
    summary = refresh(execution=execution)
    print(f"V3.6-A Trade Intelligence refresh complete: {summary}")
